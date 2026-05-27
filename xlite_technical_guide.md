# Xlite 技术详解

## 目录

- [1. Xlite 是什么](#1-xlite-是什么)
- [2. 核心原理](#2-核心原理)
  - [2.1 图化（Graph）技术](#21-图化graph技术)
  - [2.2 算子融合](#22-算子融合)
  - [2.3 内存管理](#23-内存管理)
- [3. 架构设计](#3-架构设计)
- [4. 两种运行模式](#4-两种运行模式)
- [5. 性能对比](#5-性能对比)
- [6. 技术优势与代价](#6-技术优势与代价)
- [7. 适用场景](#7-适用场景)
- [8. 使用限制](#8-使用限制)
- [9. 配置方式](#9-配置方式)

---

## 1. Xlite 是什么

**Xlite** 是华为昇腾（Ascend）NPU 上的高性能 LLM 推理运行时，是 vLLM-Ascend 项目的核心加速组件。

### 关键信息

| 属性 | 说明 |
|------|------|
| **开发者** | 华为技术有限公司（Huawei Technologies） |
| **定位** | 针对昇腾 NPU 优化的图化推理引擎 |
| **集成方式** | vLLM 插件（vllm-ascend/vllm_ascend/xlite/） |
| **核心语言** | C++（底层）+ Python（接口层） |
| **支持模型** | Llama、Qwen、GLM4、MiniMax 等主流架构 |

### 解决的问题

传统 PyTorch 推理在 LLM Decode 阶段面临严重的 **Host-Device 交互开销**：

```
问题：每次生成 1 个 token，都要经历：
CPU 准备参数 → 下发到 NPU → 执行 kernel → 返回结果 → CPU 处理 → 下一轮

重复 1000 次生成 1000 个 token = 1000 次往返延迟！
```

Xlite 通过**图化技术**将上述过程优化为：

```
优化：第一次记录计算图，后续直接重放
CPU 捕获图（1次）→ NPU 循环执行（1000次，无需 CPU 介入）
```

---

## 2. 核心原理

### 2.1 图化（Graph）技术

#### 非图化 vs 图化对比

```
┌─────────────────────────────────────────────────────────────────┐
│                     非图化执行 (标准 PyTorch)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU:  准备参数 → 下发 kernel → 等待完成 → 处理结果 → 下一层      │
│        [开销]     [开销]      [等待]    [开销]                   │
│                    ▲                                             │
│  NPU:         [执行]                                             │
│                    │                                             │
│        每层都要 CPU-NPU 往返通信                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     图化执行 (Xlite)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CPU:  Capture(记录计算步骤) ───────────────────────────────►    │
│                              Replay(只更新输入指针，重复执行)     │
│                                                                  │
│  NPU:  ┌─────────────────────────────────────────────────────┐  │
│        │  [Kernel1] → [Kernel2] → [Kernel3] → ... → [KernelN] │  │
│        │      ↑__________________________________________↓    │  │
│        │      内部循环，NPU 自主调度，CPU 不参与每层调度         │  │
│        └─────────────────────────────────────────────────────┘  │
│                                                                  │
│  效果：消除 Host-Device 往返开销，kernel launch 开销接近零        │
└─────────────────────────────────────────────────────────────────┘
```

#### 图化决策逻辑

```python
# xlite.py 核心逻辑
with_prefill = attn_metadata.attn_state not in [
    AscendAttentionState.DecodeOnly,
    AscendAttentionState.SpecDecoding,
]

# Full Mode: prefill + decode 都图化
# Decode-Only Mode: 只有 decode 阶段图化
use_xlite_graph = not with_prefill or self.full_mode
```

**为什么 Decode 阶段更适合图化？**

| 阶段 | 特点 | 适合图化？ |
|------|------|-----------|
| **Prefill** | 输入长度变化大（100~8000 tokens） | ⚠️ 需限制最大长度 |
| **Decode** | 固定 1 token/序列，图结构完全静态 | ✅ 完美适配 |

---

### 2.2 算子融合

#### 融合示例

**融合前（6 个独立 kernel）：**
```python
# 1. LayerNorm
hidden = rms_norm(hidden)

# 2. QKV Linear
qkv = linear(hidden, qkv_weight)

# 3. Split
q, k, v = split(qkv, 3)

# 4. RoPE
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)

# 5. Attention
attn = flash_attention(q, k, v, mask)

# 6. Output Linear
output = linear(attn, o_proj_weight)
```

**融合后（1 个超级 kernel）：**
```python
# Xlite Fused Attention
output = fused_mha_layer(
    hidden=hidden,
    qkv_weight=qkv_weight,
    o_proj_weight=o_proj_weight,
    freq_cis=precomputed_rope,  # 预计算，无需重复生成
    block_table=kv_cache_blocks,
)
```

**收益：**
- 减少中间结果内存读写
- 减少 kernel launch 次数
- 更好的数据局部性

---

### 2.3 内存管理

#### 张量池（Tensor Pool）

```python
# 初始化时预分配所有中间内存
rt_pool_size = self.xlite_model.get_tensor_pool_size()
self.xlite_rt.init_tensor_pool(rt_pool_size)  # 一次性分配

# 典型值：几百 MB ~ 几 GB
```

**优势：**
- ✅ 推理时零动态分配
- ✅ 内存地址固定，适合 DMA
- ✅ 无内存碎片

#### RoPE 预计算与驻留

```python
def _precompute_freqs_cis(self) -> torch.Tensor:
    """预计算旋转位置编码，常驻 NPU 显存"""
    base = self.xlite_config.rope_theta
    rotary_dim = self.xlite_config.rope_head_dim
    max_seq_len = self.xlite_config.max_seq_len
    
    # CPU 计算
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    t = torch.arange(max_position_embeddings)
    freqs = torch.outer(t, inv_freq)
    freq_cis = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    
    # 一次性传送到 NPU，后续重复使用
    return freq_cis.to(device="npu")
```

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              vLLM-Ascend                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        XliteWrapper (Python)                             ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  ││
│  │  │  XliteModel  │    │   Runtime    │    │    ModelConfig           │  ││
│  │  │  (C++ Model) │◄──►│  (C++ RT)    │    │    (C++ Config)          │  ││
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘  ││
│  │           ▲                  ▲                    ▲                    ││
│  │           │                  │                    │                    ││
│  │     _build_model()    init_tensor_pool()   _build_model_config()     ││
│  └───────────┼──────────────────┼────────────────────┼────────────────────┘│
│              │                  │                    │                      │
│              ▼                  ▼                    ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      xlite._C (Python Binding)                         ││
│  │  - AttnMeta: Attention 元数据封装                                       ││
│  │  - AttnMHA: 多头注意力类型                                              │││
│  │  - Model: 图化模型容器                                                  ││
│  │  - ModelConfig: 模型配置                                                ││
│  │  - Runtime: 运行时环境                                                  ││
│  │  - ScoringFunc*: MoE 打分函数                                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Huawei Xlite Runtime (C++)                        ││
│  │  - 图化引擎 (Graph Engine)                                              ││
│  │  - 算子库 (Fused Operators)                                             ││
│  │  - 内存池管理 (Tensor Pool)                                             ││
│  │  - 调度器 (Task Scheduler)                                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 职责 | 关键方法 |
|------|------|---------|
| **XliteWrapper** | Python 层封装，负责图/非图路由 | `__call__()`, `register_kv_caches()` |
| **XliteModel** | 模型适配基类，转换 vLLM 模型到 xlite | `_build_model()`, `_build_model_config()` |
| **LlamaXliteModel** | Llama 架构适配 | 权重映射、RoPE 预计算 |
| **QwenMoeXliteModel** | Qwen MoE 架构适配 | 专家权重扁平化 |
| **Runtime** | C++ 运行时，管理内存池和执行 | `init_tensor_pool()`, `forward()` |
| **ModelConfig** | C++ 配置对象，定义模型结构 | 设置 hidden_size, n_layers 等 |

---

## 4. 两种运行模式

### 4.1 Decode-Only 模式（推荐）

```python
additional_config = {
    "xlite_graph_config": {
        "enabled": True
        # full_mode 默认为 False
    }
}
```

**特点：**
- ✅ Prefill 阶段：使用原生 PyTorch（灵活处理变长输入）
- ✅ Decode 阶段：使用 Xlite 图化（高性能生成）
- ✅ 平衡灵活性与性能

**适用：** 大多数生产环境

---

### 4.2 Full Mode 模式

```python
additional_config = {
    "xlite_graph_config": {
        "enabled": True,
        "full_mode": True
    }
}
```

**特点：**
- ✅ Prefill + Decode 全部图化
- ⚠️ 需要预先设定最大序列长度
- ⚠️ 内存占用更大（需分配 max_seq_len 的图）

**适用：** 固定 batch size、固定最大长度的场景

---

### 4.3 模式对比

| 维度 | Decode-Only | Full Mode |
|------|-------------|-----------|
| **Prefill** | PyTorch（灵活） | Xlite 图化（需静态形状） |
| **Decode** | Xlite 图化 | Xlite 图化 |
| **内存占用** | 中等 | 较高 |
| **灵活性** | 高 | 低 |
| **适用场景** | 通用 | 固定配置的生产环境 |

---

## 5. 性能对比

### 实测数据（GLM-4.7 W8A8 on Ascend 910B3）

**测试条件：** 8K input & 1K output

| 并发 | 方案 | TTFT (ms) | TPOT (ms) | QPS (req/s) | 输出速度 (token/s) |
|------|------|-----------|-----------|-------------|-------------------|
| 1 | aclgraph | 701.32 | 55.25 | 0.02 | 17.91 |
| 1 | **xlite-decode-only** | 707.03 | **31.17** | **0.03** | **31.45** |
| 1 | 提升 | +0.8% | **-43.6%** | **+50%** | **+75.6%** |
| | | | | | |
| 16 | aclgraph | 42040.55 | 66.36 | 0.14 | 143.31 |
| 16 | **xlite-decode-only** | 34304.14 | **48.39** | **0.18** | **188.94** |
| 16 | 提升 | **-18.4%** | **-27.1%** | **+28.6%** | **+31.8%** |
| | | | | | |
| 64 | aclgraph | 361231.53 | 66.33 | 0.14 | 145.39 |
| 64 | **xlite-decode-only** | 278208.69 | **49.06** | **0.19** | **190.18** |
| 64 | 提升 | **-23.0%** | **-26.0%** | **+35.7%** | **+30.8%** |

### 关键指标解释

| 指标 | 含义 | xlite 优势来源 |
|------|------|---------------|
| **TTFT** | 首 token 延迟 | 高并发时调度优化 |
| **TPOT** | 每 token 生成时间 | 图化消除 kernel launch 开销 |
| **QPS** | 每秒请求数 | 吞吐量提升 |
| **输出速度** | 每秒生成 token 数 | Decode 阶段加速 |

---

## 6. 技术优势与代价

### 6.1 技术优势 ✅

| 优势 | 实现方式 | 效果 |
|------|---------|------|
| **消除 Host Overhead** | 图捕获与重放 | TPOT 降低 26-43% |
| **算子融合** | Fused MHA/MLP | 减少内存搬移 |
| **预分配内存池** | Tensor Pool | 零动态分配开销 |
| **RoPE 预计算** | 常驻 NPU 显存 | 节省 H2D 传输 |
| **量化感知** | W8A8 融合算子 | 降低带宽压力 |
| **NZ 格式** | Fractal-NZ 排布 | 提高计算单元利用率 |

### 6.2 技术代价 ⚠️

| 代价 | 具体表现 | 缓解方法 |
|------|---------|---------|
| **内存占用增加** | 需预分配 tensor pool + RoPE 缓存 | 合理设置 max_seq_len |
| **启动时间增加** | 初始化 + 图化编译耗时 | 服务预热 |
| **模型支持有限** | 仅支持特定架构 | 使用支持的模型 |
| **灵活性降低** | 不支持动态 batch/变长 | 使用 decode-only 模式 |
| **调试困难** | 图化后错误信息不友好 | 开发阶段用原生 PyTorch |

---

## 7. 适用场景

### 7.1 推荐使用 Xlite ✅

| 场景 | 特征 | 收益 |
|------|------|------|
| **长文本生成** | 输出 > 500 tokens | Decode 占比高，加速明显 |
| **高并发在线服务** | 持续满 batch | 吞吐量提升 30-75% |
| **固定模型部署** | 生产环境长期运行 | 启动开销被摊平 |
| **低延迟要求** | TTFT/TPOT 要求严苛 | 响应更稳定 |

### 7.2 不推荐使用 Xlite ❌

| 场景 | 特征 | 原因 |
|------|------|------|
| **短序列查询** | 输出 < 100 tokens | 启动开销占比高 |
| **动态 Batch** | 请求量波动大 | 内存浪费严重 |
| **频繁切换模型** | A/B 测试/多模型 | 每次初始化开销大 |
| **开发调试** | 频繁改代码 | 调试困难，灵活性差 |
| **Prefill 主导** | RAG/长文档理解 | Prefill 阶段收益小 |

---

## 8. 使用限制

### 8.1 硬性限制

```python
class XliteGraphConfig:
    def __init__(self, xlite_graph_config, vllm_config):
        if self.enabled:
            # ❌ 不支持投机解码
            if bool(vllm_config.speculative_config):
                raise RuntimeError("Xlite not compatible with speculative decoding")
            
            # ❌ 不支持流水线并行
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                raise RuntimeError("Xlite not compatible with pipeline parallelism")
            
            # ⚠️ 推荐使用 block_size=128
            if vllm_config.cache_config.block_size != 128:
                logger.warning("Recommended block size is 128")
```

### 8.2 模型支持列表

```python
strategy_map = {
    "LlamaForCausalLM": LlamaXliteModel,
    "Qwen2ForCausalLM": LlamaXliteModel,
    "Qwen3ForCausalLM": LlamaXliteModel,
    "Qwen3VLForConditionalGeneration": LlamaXliteModel,
    "Qwen3MoeForCausalLM": QwenMoeXliteModel,
    "Glm4MoeForCausalLM": Glm4MoeXliteModel,
    "MiniMaxM2ForCausalLM": MiniMaxM2XliteModel,
}
# 不在列表中的模型 → 不支持
```

### 8.3 配置限制

| 配置项 | 限制 | 说明 |
|--------|------|------|
| max_seq_len | 启动时固定 | 超过会报错 |
| max_batch_size | 启动时固定 | 超过会报错 |
| block_size | 推荐 128 | 其他值可能性能下降 |
| 量化格式 | W8A8/FP16/BF16 | 需适配的量化方案 |

---

## 9. 配置方式

### 9.1 基础配置

```python
from vllm import LLM

llm = LLM(
    model="your-model-path",
    additional_config={
        "xlite_graph_config": {
            "enabled": True,        # 启用 xlite
            "full_mode": False      # False=decode-only, True=full mode
        }
    }
)
```

### 9.2 完整配置示例

```python
additional_config = {
    "xlite_graph_config": {
        "enabled": True,
        "full_mode": False,  # decode-only 模式
    },
    # 其他 ascend 配置
    "weight_nz_mode": 1,  # 0=禁用, 1=量化启用, 2=全部启用
    "enable_async_exponential": False,
}
```

### 9.3 环境变量（遗留方式，不推荐）

```bash
# 某些配置可通过环境变量设置（将被废弃）
export VLLM_ASCEND_ENABLE_NZ=1
```

**推荐：** 使用 `additional_config` 替代环境变量。

---

## 10. 总结

### 核心思想

> **Xlite 通过"预编译计算图 + 运行时重放"的技术路线，将 LLM Decode 阶段的 Host-Device 交互开销降低到接近零。**

### 关键数据

| 指标 | 提升 |
|------|------|
| TPOT（单并发） | **-43.6%** |
| TPOT（高并发） | **-26% ~ -27%** |
| 吞吐量 | **+30% ~ +75%** |

### 使用建议

1. **生产环境** → 启用 xlite-decode-only
2. **开发调试** → 使用原生 PyTorch
3. **固定配置** → 可尝试 full-mode
4. **模型选择** → 优先使用支持的架构

### 一句话概括

**Xlite 是用"启动时间和内存"换取"推理性能"的加速器，适合长期运行的生产服务，不适合短期任务和开发调试。**

---

*文档版本：v1.0*  
*最后更新：2026-05*  
*相关项目：https://github.com/vllm-project/vllm-ascend*
