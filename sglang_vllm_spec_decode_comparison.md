# SGLang v0.5.12 vs vLLM 投机解码特性对比分析

## 1. 概述

本文档对比分析 SGLang v0.5.12 和 vLLM 在投机解码（Speculative Decoding）方面的实现差异，评估 vLLM 移植 SGLang 特性的可行性和优先级。

---

## 2. SGLang v0.5.12 投机解码特性清单

### 2.1 核心投机算法

| 特性 | SGLang 实现 | vLLM 对应 | 移植难度 | 优先级 |
|------|------------|-----------|----------|--------|
| **EAGLE** | ✅ eagle_worker.py | ✅ v1/spec_decode/eagle.py | 低 | 已支持 |
| **EAGLE-3** | ✅ eagle_worker_v2.py | ✅ v1/spec_decode/eagle.py | 低 | 已支持 |
| **MTP (Multi-Token Prediction)** | ✅ frozen_kv_mtp_worker.py | ✅ 支持多种模型 | 低 | 已支持 |
| **N-gram** | ✅ ngram_worker.py | ✅ v1/spec_decode/ngram_proposer.py | 低 | 已支持 |
| **Standalone Draft** | ✅ standalone_worker.py | ✅ draft_model.py | 低 | 已支持 |
| **DFlash** | ✅ dflash_worker.py | ✅ v1/spec_decode/dflash.py | 低 | 已支持 |
| **Multi-layer EAGLE** | ✅ multi_layer_eagle_worker.py | ❌ 未支持 | 中 | 中 |
| **Medusa** | ❌ 未提及 | ✅ v1/spec_decode/medusa.py | - | vLLM 独有 |
| **Suffix Decoding** | ❌ 未提及 | ✅ v1/spec_decode/suffix_decoding.py | - | vLLM 独有 |

### 2.2 Spec V2 架构优化

| 特性 | SGLang 实现 | vLLM 对应 | 移植难度 | 优先级 |
|------|------------|-----------|----------|--------|
| **Adaptive Speculative Decoding** | ✅ adaptive_spec_params.py + adaptive_runtime_state.py | ❌ 未支持 | 高 | 高 |
| **Runtime State Management** | ✅ SpecRuntimeState 抽象 | ❌ 未支持 | 高 | 中 |
| **Two-Batch Overlap** | ✅ TboAttnBackend | ⚠️ 部分支持 | 中 | 中 |
| **Breakable CUDA Graph** | ✅ enable_breakable_cuda_graph | ✅ 支持 | 低 | 已支持 |
| **Piecewise CUDA Graph** | ✅ piecewise_cuda_graph | ✅ CudaGraphDispatcher | 低 | 已支持 |

### 2.3 模型特定支持

| 特性 | SGLang 实现 | vLLM 对应 | 移植难度 | 优先级 |
|------|------------|-----------|----------|--------|
| **DeepSeek V4 投机解码** | ✅ 完整支持 | ✅ 基础支持 | 中 | 高 |
| **DeepSeek V3.2 FP4** | ✅ FP4 + PDL | ⚠️ 部分支持 | 中 | 高 |
| **Kimi K2.5 EAGLE-3 MLA** | ✅ TokenSpeed MLA | ⚠️ 基础MLA | 高 | 高 |
| **Gemma 4 MTP** | ✅ gemma4_mtp | ✅ v1/spec_decode/gemma4.py | 低 | 已支持 |
| **GLM-5 FP4** | ✅ 支持 | ⚠️ 未完整 | 中 | 中 |

### 2.4 高级特性

| 特性 | SGLang 实现 | vLLM 对应 | 移植难度 | 优先级 |
|------|------------|-----------|----------|--------|
| **SWA (Sliding Window Attention)** | ✅ 投机解码集成 | ❌ 未支持投机场景 | 中 | 中 |
| **HiCache + UnifiedRadixTree** | ✅ 完整支持 | ❌ 架构差异大 | 高 | 低 |
| **HiSparse** | ✅ 内存卸载 | ⚠️ v1/kv_offload 基础版 | 高 | 低 |
| **TokenSpeed MLA** | ✅ Backend集成 | ⚠️ FlashMLA | 高 | 中 |
| **Tree Mask 优化** | ✅ TreeMaskMode 枚举 | ⚠️ 基础tree attn | 中 | 中 |

---

## 3. 详细特性分析

### 3.1 Adaptive Speculative Decoding（自适应投机解码）

#### SGLang 实现
```python
# adaptive_spec_params.py
class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    """
```

**核心机制：**
1. **EMA 跟踪**：使用指数移动平均跟踪接受率
2. **动态步数调整**：根据接受率动态调整 `speculative_num_steps`
3. **滞后机制**：使用 `down_hysteresis` 和 `up_hysteresis` 避免震荡
4. **候选步数**：支持配置多个候选步数 (1, 3, 7)
5. **预热期**：`warmup_batches` 期间不调整

**运行时状态管理：**
```python
# adaptive_runtime_state.py
@dataclass
class SpecRuntimeState:
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None"
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"
```

#### vLLM 现状
- **当前支持**：固定 `num_speculative_tokens`，不支持运行时调整
- **缺失组件**：
  1. 无 EMA 接受率跟踪
  2. 无动态步数切换机制
  3. 无运行时状态管理抽象
  4. CUDA Graph 和 Attention Backend 无法动态切换

#### 移植建议
1. **Phase 1**：添加 `AdaptiveSpeculativeParams` 类用于参数计算
2. **Phase 2**：实现简单的 EMA 跟踪和步数调整
3. **Phase 3**：重构 CUDA Graph 和 Backend 以支持运行时切换（复杂）

### 3.2 EAGLE-3 V2 架构

#### SGLang 改进
- **分离的 Draft/Verify/Extend 阶段**：每个阶段有独立的 Attention Backend
- **重叠执行**：支持 `enable_two_batch_overlap` 并行处理
- **SWA 支持**：EAGLE-3 与滑动窗口注意力集成
- **多 Token Map 支持**：`speculative_token_map` 配置

#### vLLM 现状
- 基础 EAGLE-3 已实现 (`eagle3`, `extract_hidden_states`)
- 支持 `pass_hidden_states_to_model`
- 缺少 V2 架构的细粒度阶段分离

#### 移植建议
- 优先级较低，vLLM 当前实现已能满足大部分场景

### 3.3 Tree Mask 优化

#### SGLang TreeMaskMode
```python
# eagle_utils.py
class TreeMaskMode(Enum):
    FULL_MASK = auto()           # 完整注意力掩码
    QLEN_ONLY = auto()           # 仅 query 长度
    QLEN_ONLY_BITPACKING = auto()  # bit 压缩版本
```

**优势：**
- 减少内存占用（bitpacking 版本）
- 支持不同的树结构验证策略
- 灵活应对不同的 batch size

#### vLLM 现状
- 基础 Tree Attention 已实现
- 无多种掩码模式选择
- 内存优化空间有限

### 3.4 Sliding Window Attention (SWA) 集成

#### SGLang 实现
- `swa_radix_cache.py`：SWA 感知的 Radix Cache
- `unified_radix_cache.py`：统一树 + SWA 支持
- EAGLE-3 + SWA 完整集成

#### vLLM 现状
- SWA 基础支持存在
- 与投机解码集成不完整
- 缺少 `HiCache` 框架

### 3.5 Multi-layer EAGLE

#### SGLang 实现
```python
# multi_layer_eagle_worker.py
class MultiLayerEagleWorker(BaseSpecWorker):
    """支持多层 EAGLE draft 模型"""
```

**特点：**
- 多层 draft 模型结构
- 层间 hidden state 传递
- 更准确的 token 预测

#### vLLM 现状
- 不支持多层 EAGLE 结构
- 仅支持单层 draft head

---

## 4. 移植可行性矩阵

### 4.1 高优先级可移植特性

| 特性 | 移植难度 | 预估工作量 | 依赖项 | 建议版本 |
|------|---------|-----------|--------|---------|
| **Adaptive Spec Decoding** | 高 | 2-3 周 | CUDA Graph 重构 | v0.8.x |
| **TreeMaskMode** | 中 | 1 周 | Attention Backend | v0.7.x |
| **SWA + Spec Decode** | 中 | 1-2 周 | HiCache 简化版 | v0.7.x |
| **Multi-layer EAGLE** | 中 | 1-2 周 | Model 架构调整 | v0.7.x |

### 4.2 低优先级/高难度特性

| 特性 | 移植难度 | 预估工作量 | 阻塞因素 |
|------|---------|-----------|---------|
| **HiCache 完整框架** | 很高 | 1-2 月 | 架构差异大 |
| **HiSparse** | 很高 | 1 月 | 内存管理重构 |
| **TokenSpeed MLA Backend** | 高 | 2-3 周 | CUDA 内核开发 |
| **PD Disaggregation + Spec** | 高 | 2-3 周 | PD 架构不稳定 |

### 4.3 已对齐特性

| 特性 | SGLang | vLLM | 状态 |
|------|--------|------|------|
| EAGLE | ✅ | ✅ | 功能对等 |
| EAGLE-3 | ✅ | ✅ | 功能对等 |
| MTP (DeepSeek) | ✅ | ✅ | 功能对等 |
| N-gram | ✅ | ✅ | 功能对等 |
| Draft Model | ✅ | ✅ | 功能对等 |
| DFlash | ✅ | ✅ | 功能对等 |
| Gemma 4 MTP | ✅ | ✅ | 功能对等 |
| CUDA Graphs | ✅ | ✅ | 功能对等 |
| Piecewise CUDA Graph | ✅ | ✅ | 功能对等 |

---

## 5. 代码结构对比

### 5.1 SGLang 投机解码目录结构
```
python/sglang/srt/speculative/
├── adaptive_runtime_state.py      # 自适应运行时状态管理 ⭐
├── adaptive_spec_params.py        # 自适应参数计算 ⭐
├── base_spec_worker.py            # 基础 worker 抽象
├── dflash_info.py                 # DFlash 元数据
├── dflash_worker.py               # DFlash 实现
├── draft_utils.py                 # Draft 工具函数
├── eagle_draft_cuda_graph_runner.py
├── eagle_draft_extend_cuda_graph_runner.py
├── eagle_info.py                  # EAGLE 输入/输出结构
├── eagle_info_v2.py               # EAGLE V2 扩展 ⭐
├── eagle_utils.py                 # EAGLE 工具函数
├── eagle_worker.py                # EAGLE 基础实现
├── eagle_worker_v2.py             # EAGLE V2 实现 ⭐
├── external_corpus_manager.py     # 外部语料管理
├── frozen_kv_mtp_worker.py        # MTP 实现
├── multi_layer_eagle_utils.py     # 多层 EAGLE 工具
├── multi_layer_eagle_worker.py    # 多层 EAGLE ⭐
├── multi_layer_eagle_worker_v2.py # 多层 EAGLE V2 ⭐
├── ngram_info.py                  # N-gram 元数据
├── ngram_worker.py                # N-gram 实现
├── spec_info.py                   # 投机算法元数据
├── spec_utils.py                  # 通用工具
└── standalone_worker.py           # Standalone draft
```

### 5.2 vLLM 投机解码目录结构
```
vllm/v1/spec_decode/
├── __init__.py
├── custom_class_proposer.py       # 自定义投机类
├── dflash.py                      # DFlash 实现 ✅
├── draft_model.py                 # Draft model ✅
├── eagle.py                       # EAGLE 入口 ✅
├── extract_hidden_states.py       # EAGLE-3 hidden states ✅
├── gemma4.py                      # Gemma 4 MTP ✅
├── llm_base_proposer.py           # 基础 proposer ✅
├── medusa.py                      # Medusa (vLLM 独有) ✅
├── metadata.py                    # 元数据
├── metrics.py                     # 指标收集
├── mtp.py                         # MTP 通用实现 ✅
├── ngram_proposer.py              # N-gram CPU ✅
├── ngram_proposer_gpu.py          # N-gram GPU ✅
├── suffix_decoding.py             # Suffix decoding (vLLM 独有) ✅
└── utils.py                       # 工具函数

vllm/v1/worker/gpu/spec_decode/
├── __init__.py
├── eagle/
│   ├── __init__.py
│   ├── cudagraph.py               # CUDA Graph 管理 ✅
│   ├── eagle3_utils.py            # EAGLE-3 工具 ✅
│   ├── speculator.py              # Eagle Speculator ✅
│   └── utils.py                   # 工具函数 ✅
├── rejection_sampler.py           # 拒绝采样器 ✅
├── rejection_sampler_utils.py     # 工具函数 ✅
└── utils.py                       # 通用工具 ✅
```

---

## 6. 移植建议路线图

### Phase 1: 基础设施 (v0.7.x)
- [ ] TreeMaskMode 抽象和多模式支持
- [ ] 基础 Adaptive 参数计算框架
- [ ] SWA + 投机解码集成

### Phase 2: 自适应投机 (v0.8.x)
- [ ] 完整 Adaptive Speculative Decoding
- [ ] EMA 接受率跟踪
- [ ] 运行时步数切换

### Phase 3: 高级特性 (v0.9.x)
- [ ] Multi-layer EAGLE 支持
- [ ] 简化的 HiCache 框架
- [ ] DeepSeek V4 投机优化

### Phase 4: 内核优化 (未来)
- [ ] TokenSpeed MLA Backend
- [ ] HiSparse 内存卸载
- [ ] PD Disaggregation + Spec

---

## 7. 结论

### 7.1 总体评估
- **vLLM 投机解码基础扎实**：EAGLE、EAGLE-3、MTP 等核心算法与 SGLang 功能对等
- **SGLang 在自适应和优化方面领先**：Adaptive Spec、V2 架构、SWA 集成值得移植
- **架构差异影响移植**：HiCache、HiSparse 等深度集成的特性移植成本高

### 7.2 优先建议
1. **高 ROI**：Adaptive Speculative Decoding - 显著提升动态工作负载性能
2. **中 ROI**：TreeMaskMode 优化 - 降低内存占用
3. **战略性**：SWA + Spec Decode - 支持长上下文场景
4. **长期**：Multi-layer EAGLE - 提升预测准确性

### 7.3 风险提示
- CUDA Graph 运行时切换在 vLLM 架构中需重大重构
- HiCache/HiSparse 与 vLLM 的 KV Cache 管理架构差异较大
- SWA 集成需要 Radix Cache 层面的改动
