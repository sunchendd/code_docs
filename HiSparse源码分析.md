# HiSparse 源码分析

> 基于 SGLang 源码：sgl-project/sglang

---

## 一、整体架构：三层结构 + 单 Kernel

HiSparse 代码分布在三个层次，核心是一个单 kernel 的 CUDA swap-in：

| 层次 | 关键文件 | 职责 |
|------|---------|------|
| **内存池层** | `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | 双层分配器逻辑（Logical + HiSparse）、Page Table 映射 |
| **调度/编排层** | `python/sglang/srt/managers/hisparse_coordinator.py` | 准入、LRU 管理、Staging→Decode 转换、Swap-in 派发 |
| **CUDA Kernel 层** | `python/sglang/jit_kernel/csrc/hisparse.cuh` | 单 kernel 完成 hash 查找 + LRU 驱逐 + Host→Device 搬运 |

其余关键文件：

| 文件 | 职责 |
|------|------|
| `python/sglang/jit_kernel/hisparse.py` | JIT 编译绑定，`load_cache_to_device_buffer_mla` / `dsv4_mla` |
| `python/sglang/srt/mem_cache/memory_pool_host.py` | `MLATokenToKVPoolHost`, `HiSparseHostPoolMixin` |
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | `HiSparseC4DevicePool` (DSv4 device buffer) |
| `python/sglang/srt/layers/attention/dsv4/indexer.py` | C4Indexer 接入 HiSparse 的 swap-in 入口 |
| `python/sglang/srt/managers/schedule_batch.py` | 每个 decode step 的 `map_last_loc_to_buffer` 调用 |
| `python/sglang/srt/managers/scheduler.py` | `_build_hisparse_decode_batch`, staging→decode 过渡 |

---

## 二、核心数据结构：双层 Page Table

HiSparse 最关键的设计是**每个 request 维护两份 page table**：

### 2.1 逻辑→HiSparse 物理映射

```
full_to_hisparse_device_index_mapping: [logical_size + page_size + 1, ]    int64
  - 将"模型眼中的逻辑 KV index"映射到"HiSparse device buffer 物理 index"
  - 最后一个 entry 恒为 -1 (哨兵)
  - 在 alloc_device_buffer 时建立映射
  - 在 map_last_loc_to_buffer 每步更新最新 token 的映射
  - 在 request_finished / free_hisparse 时清零
```

### 2.2 逻辑→Host 物理映射

```
req_to_host_pool: [max_req_slots, max_compressed_context_len + page_size]  int64
  - 将 req/compressed_token_pos 映射到 host memory 物理地址
  - -1 表示未分配
  - 按 page 粒度分配，惰性增长
```

### 2.3 Device Buffer 内层 (每层独立)

```
req_device_buffer_tokens:   [layer_num, max_req_slots, padded_buffer_size]  int32
  - 记录每个 buffer slot 当前存放的是什么 token position (-1 = 空)

req_device_buffer_token_locs: [layer_num, max_req_slots, padded_buffer_size]  int32
  - 记录每个 buffer slot 在物理 device KV cache 中的地址

lru_slots: [layer_num, max_req_slots, device_buffer_size]  int16
  - LRU 排序的 slot 索引
  - 布局: [stale_evictables, ..., miss_replacements, ..., active_hits]
  - 前 = LRU (最可驱逐), 后 = MRU (最近命中)
  - 初始化为 [0, 1, ..., buffer_size-1] (插入序 = LRU 序)
```

### 2.4 Reserved Slot

Buffer 额外保留一页 `page_size` 个 slot：
- `padded_buffer_size = device_buffer_size + page_size`
- `buffer[device_buffer_size]` 专门存放最新 token
- 不参与 LRU 追踪，swap-in kernel 中直接命中

---

## 三、Swap-in Kernel 详解

`load_cache_to_device_buffer_kernel` — 一个 kernel 完成 hash 查找 + LRU 驱逐 + H2D 搬运。每个 block 处理一个 request。

### Phase 1: Hash Table 构建 (共享内存)

- 表大小 = `2 × NUM_TOP_K`，开放寻址，Knuth 乘法 hash
- 最新 token (`seq_len - 1`) 特殊处理：标记为 `TOKEN_HIT`，直接使用 reserved slot
- 其余 token 插入共享内存 hash 表

### Phase 2: LRU 扫描 & 分类 (warp 级)

- 遍历当前 `lru_slots` 中每个 buffer slot，warp 级并行
- 每个线程处理一个 slot：在 hash 表中查找其 token position
- **Hit**: compact 到 MRU 方向
- **Evictable**: compact 到 LRU 方向
- 通过 `__ballot_sync` + `warp_inclusive_scan` 实现无锁并行

### Phase 3: Miss 分配 (warp 级)

- 对每个未命中的 top_k token，从 LRU 末端偷取 evictable slot
- 将 miss token 写入该 slot 的 `device_buffer_tokens`
- 设置 `top_k_device_locs[token_idx] = device_buffer_locs[evict_slot]`

### Phase 4: LRU 序写回

```
lru_slots 最终布局:
  [0, total_evictable - total_misses):   真正 stale 的 evictable (最 LRU)
  [total_evictable - total_misses, total_evictable): 刚加载的 miss tokens
  [total_evictable, HOT_BUFFER_SIZE):    命中的活跃 tokens (最 MRU)
```

### Phase 5: 数据搬运 (warp 级)

- 每个 warp 负责搬运一个 miss token
- **MLA 路径** (`transfer_item_warp`): 128-bit 批量传输 (paired 64-bit loads)
  - `ld.global.nc.v2.b64` (non-coherent 读) + `st.global.cg.v2.b64` (cache-global 写)
  - 同时搬运 K 和 V
- **DSv4 路径** (`device::hisparse::transfer_item`): 处理 page-padded device layout
  - GPU: 64 items/page, 576B aligned
  - CPU: linear layout
  - K-only (MLA)

### 快速路径

当 `seq_len <= device_buffer_size` 时，所有 token 已在 buffer 中按序排列，跳过 hash/LRU/搬运，直接做内存拷贝。

---

## 四、调度生命周期

### 4.1 准入 (Admission)

```python
admit_request_into_staging(req):
  1. 设 req.hisparse_staging = True
  2. 取 prefill KV 的 logical index → translate_loc_from_full_to_hisparse_device
  3. 分配 host memory page (alloc_paged_token_slots)
  4. 在 staging stream 上发起 async DMA: backup_from_device_all_layer (kernel 模式)
  5. 将 (start_event, finish_event, req) 加入 ack_staging_queue
```

### 4.2 Staging → Decode 转换

```python
collect_ready_reqs():
  1. 检查 ack_staging_queue 中 finish_event 已完成的事件
  2. TP 同步: all_reduce(MIN) 保证所有 TP 卡都完成 staging
  3. 对每个 ready req:
    a. alloc_device_buffer(req)    # 分配 device buffer 并建立映射
    b. _skip_first_backup = True   # 跳过下次 decode 的 backup
    c. req.hisparse_staging = False
```

### 4.3 alloc_device_buffer

```python
alloc_device_buffer(req):
  1. 将逻辑 indices 压缩 (DSv4: compress_ratio=4, DSA: compress_ratio=1)
  2. 在 full_to_hisparse_device_index_mapping 中查找已有映射
  3. 已有映射够用 → 复用 (增量分配)
  4. 不够用 → 从 hisparse_attn_allocator 分配新 page
  5. DSv4 特殊: 将最新 token 的映射交换到 device_buffer_size 位置
```

### 4.4 Decode 每步: map_last_loc_to_buffer

```python
map_last_loc_to_buffer:
  1. _eager_backup_previous_token()  # 备份上一个压缩 token 到 host
  2. _grow_device_buffers()          # 短序列增量分配 device buffer
  3. 将最新 token 映射到 reserved slot
```

### 4.5 _eager_backup_previous_token — 增量备份

- 每 `compress_ratio` 步才产生一个压缩 token
- 将上一个压缩 token 的 KV 从 device buffer → host memory 做一次性备份
- 使用独立的 `decode_backup_stream`，不阻塞主调度流
- 跳过一次 (staging 已全覆盖 prefill)

### 4.6 请求结束

```python
request_finished(req):
  1. 等待 decode_producer_stream 和 pending backup
  2. 释放 hisparse device buffer indices
  3. 释放 host memory
  4. 清除所有映射、LRU 状态、buffer tokens/locs
```

### 4.7 PD Disaggregation 直连路径

```python
admit_request_direct(req):
  - KV 通过 RDMA 直接入 host pool，跳过 staging DMA
  - 仅分配 device buffer (但内容为空)
  - 短序列 (seq_len <= device_buffer_size): preload 全部 token
  - 长序列: reset device_buffer_tokens 为 -1，让 swap-in kernel 全走 miss 路径
  - 注意: DSv4 暂不支持 (NotImplementedError)
```

---

## 五、DSv4 vs DSA 两条路径的差异

| 维度 | DSA (DeepSeek-V3.2, GLM-5) | DSv4 |
|------|------|------|
| compress_ratio | 1 (不压缩) | 4 (每4个token压缩1个) |
| 分配器 | `HiSparseTokenToKVPoolAllocator` | `DeepSeekV4HiSparseTokenToKVPoolAllocator` |
| Device Pool | `HiSparseDSATokenToKVPool` (extends `DSATokenToKVPool`) | `HiSparseC4DevicePool` (extends `DeepSeekV4SingleKVPool`) |
| Host Pool | `MLATokenToKVPoolHost` | `DeepSeekV4SingleKVPoolHost` |
| Swap-in Kernel | `load_cache_to_device_buffer_mla` (linear layout) | `load_cache_to_device_buffer_dsv4_mla` (page-padded device layout) |
| Backup Kernel | `transfer_kv_all_layer_mla` | `hisparse_offload_to_host` (persistent kernel) |
| PD Direct-to-Host | 支持 | `NotImplementedError` |
| Page Table | 1:1 映射 | compressed: `(loc - 3) // 4` |

---

## 六、Page Table 管理链路总结

```
模型 Forward 层使用的 logical index
    │
    ▼
full_to_hisparse_device_index_mapping[logical]
    │
    ▼
HiSparse Device Buffer 物理 index (GPU 上的实际 KV slot)
    │
    ▼ (swap-in miss 时)
host_cache_locs[request, compressed_pos]
    │
    ▼
Host Memory 物理 index (CPU pin_memory 上的实际 KV slot)
```

- **alloc_extend**: 同时为 logical allocator 和 hisparse_attn_allocator 分配 page，建立 `logical → hisparse device` 映射
- **alloc_device_buffer**: 复用或增量分配 hisparse buffer pages，重建映射
- **map_last_loc_to_buffer**: 每步更新最新 token 的映射
- **request_finished**: 清零映射 + 释放所有资源

---

## 七、关键设计决策

1. **单 Kernel 一体化解法**: 把 hash 查找、LRU 驱逐、page table 更新、H2D 搬运全部压缩到一个 kernel 里，避免控制面和数据面的串行化，消除后台线程的延迟抖动。

2. **Warp 级无锁数据结构**: LRU 扫描和 miss 分配都用 warp-level ballot + inclusive scan，完全在寄存器中操作，没有任何 atomic 竞争。

3. **每层独立 LRU**: `lru_slots[layer_id]` 是每层独立的，不同层的 indexer 可能选不同的 top_k tokens，各层 buffer 内容可以不同。

4. **eager_backup_previous_token**: 不像 HiCache 等 request 结束后再备份 — HiSparse 在 decode 每步就实时备份刚产生的新 token，保证 swap-in kernel 永远能看到最新的 KV。

5. **容量收益的本质**: `host_to_device_ratio` (默认 2~10) 意味着在相同 GPU HBM 下，系统可服务的 total token 量是原来的 ratio 倍。这是高并发下吞吐翻倍的来源。

6. **Reserved Slot 设计**: `device_buffer_size` 位置独立于 LRU，永远存放最新 token，因为最新 token 是几乎所有 attention head 的 top_k 命中项。

---

## 八、配置参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `top_k` | 2048 | Indexer 选出的 top token 数 |
| `device_buffer_size` | 4096 (2× top_k) | GPU 上每请求的 buffer 大小 |
| `host_to_device_ratio` | 2 | Host:Device 的 KV 容量比 |
| `compress_ratio` | 1 (DSA) / 4 (DSv4) | Token 压缩率 |
| `page_size` | 64 (CUDA) / 1 (HIP) | KV cache page 大小 |
| `block_size` | 1024 | CUDA kernel block 大小 |

**生产参考配置 (来自 LMSYS):**

| 场景 | top_k | device_buffer_size | host_to_device_ratio |
|------|-------|-------------------|---------------------|
| PD-disaggregation decode | 2048 | 6144 | 10 |
| 单机 8×H200 colocated | 2048 | 4096 | 8 |

---

## 九、SGLang KV Cache 演进

| 技术 | 解决的问题 |
|------|-----------|
| RadixAttention | 共享前缀复用，减少重复计算 |
| HiCache | 历史 KV 扩展到多级存储，跨请求复用 |
| HiSparse | Sparse attention 下完整 KV 常驻 HBM 的浪费 |

三者共同指向一个判断：**KV cache 已经是推理系统的主要资产，不能再被当成 attention kernel 的附属数据结构。**

---

## 十、v0.5.12 中 HiSparse 相关更新

SGLang v0.5.12 (2026-05-16) 对 HiSparse 有一项关键改进和一些能力扩展：

### 10.1 HiSparse FP8 KV Cache 支持 ([#23013](https://github.com/sgl-project/sglang/pull/23013))

**之前**: HiSparse 的 attention backend 只能用 `flashmla_sparse`，该 kernel 不接受 FP8 输入，因此 HiSparse 被锁定在 BF16 KV cache。

**现在**: 新增路由逻辑，按 KV dtype 自动选择：
| KV dtype | Attention backend |
|----------|-------------------|
| `bfloat16` | `flashmla_sparse` (不变) |
| `fp8_e4m3` | `flashmla_kv` (新增) |

**技术细节**: `flashmla_kv` 原生支持 FP8 + sparse attention (`is_fp8_kvcache=True` + `indices=...`)，HiSparse 的 hot-buffer indices 与 `flashmla_kv` 的 indices 合约是 drop-in 兼容的，不需要修改 attention 路径的任何代码，只需在 `hisparse_hook.py` 中增加 backend 选择逻辑。

**使用方式**:
```bash
sglang launch_server --model DeepSeek-V3.2 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-hisparse \
  --disable-radix-cache \
  --hisparse-config '{"top_k": 2048, "device_buffer_size": 4096}'
```

### 10.2 DeepSeek V4 HiSparse (Day-0 能力, [#23882](https://github.com/sgl-project/sglang/pull/23882))

v0.5.12 首次发布了 DeepSeek V4 的完整推理路径，HiSparse 作为 Day-0 特性被包含在其中。这是 HiSparse 从 DSA-only 到 DSv4 (C4 compression 路径) 的首次正式 release。

DSv4 HiSparse 区别于 DSA 的关键点：
- `compress_ratio = 4` (每4个 token 产生1个压缩 token)
- Swap-in kernel 使用 `load_cache_to_device_buffer_dsv4_mla` (page-padded device layout)
- Device pool 使用 `HiSparseC4DevicePool` (DeepSeekV4 特化)
- Host pool 使用 `DeepSeekV4SingleKVPoolHost` (pin_memory, K-only)
- Backup 使用 `hisparse_offload_to_host` persistent kernel
- 每层独立的 `lru_slots` 支持 C4Indexer 在不同层可能选择不同 compressed tokens

### 10.3 DSv4 HiCache 集成 ([#24691](https://github.com/sgl-project/sglang/pull/24691))

v0.5.12 将 HiCache (历史 KV 跨请求复用) 与 HiSparse (active KV 分层存储) 在 DSv4 上首次打通：

- 利用 UnifiedTree 的 SWA HiCache 能力 + Shadow Radix 机制
- 通过 sidecar pool 机制让 memory pool 共享来自 source pool 的 indices
- HiCache 负责跨请求前缀共享 (减少 compute)，HiSparse 负责请求内的稀疏 K V offload (减少 memory)
- 两者可以同时开启，形成「跨请求复用 + 请求内分层」的完整 KV 生命周期管理

### 10.4 其他相关改进

| PR | 描述 |
|----|------|
| [#23316](https://github.com/sgl-project/sglang/pull/23316) | HiCache framework for UnifiedRadixTree — HiCache + Radix tree 统一架构 |
| [#23391](https://github.com/sgl-project/sglang/pull/23391) | SWA HiCache for unified radix cache — sliding window attention 与 HiCache 结合 |
| [#24277](https://github.com/sgl-project/sglang/pull/24277) | SSD offload through Mooncake store — HiCache 扩展到 SSD 存储层 |
| [#24775](https://github.com/sgl-project/sglang/pull/24775) | MHC + DeepGemm pipeline 优化 — fused norm + fused hc_head，进一步降低 HiSparse decode 路径的额外开销 |
| [#25311](https://github.com/sgl-project/sglang/pull/25311) | TMA bulk-store `set_mla_kv_buffer` — DSv4 KV buffer 写入加速 (最高 12×) |

### 10.5 架构演化总结

```
v0.5.11 (上一版本)                 v0.5.12 (当前版本)
─────────────────────              ─────────────────────
HiSparse DSA only                  + DSv4 HiSparse (Day-0)
BF16 KV cache only                 + FP8 KV cache (flashmla_kv)
HiSparse / HiCache 独立             + DSv4 上 HiCache + HiSparse 打通
RadixAttention 独立                 + UnifiedRadixTree 整合 HiCache
GPU + CPU (两层)                    + GPU + CPU + SSD (三层，Mooncake)
```

关键趋势：**KV cache 正在成为一个跨越 GPU HBM、Host RAM、SSD 三层存储的分层生命周期管理系统，HiSparse 负责「请求内」的 sparse attention 冷热分层，HiCache 负责「跨请求」的 prefix 复用，UnifiedRadixTree 负责统一的索引和 eviction 策略。**