# vLLM / vllm-ascend 投机解码（Speculative Decoding）深度分析报告

> 分析对象：`/root/vllm/`（上游 vLLM）、`/root/vllm-ascend/`（昇腾 NPU 插件）  
> 分析日期：2026-05-19

---

## 目录

1. [当前支持的投机推理方法](#一当前支持的投机推理方法)
   - [1.1 无模型方法（Model-free）](#11-无模型方法model-free)
   - [1.2 独立草稿模型方法](#12-独立草稿模型方法)
   - [1.3 基于目标模型隐藏状态的方法（主流）](#13-基于目标模型隐藏状态的方法主流)
   - [1.4 训练/辅助方法](#14-训练辅助方法)
2. [后续迭代方向与在推进的工作](#二后续迭代方向与在推进的工作)
   - [2.1 上游 vLLM 核心趋势](#21-上游-vllm-核心趋势)
   - [2.2 vllm-ascend 核心趋势](#22-vllm-ascend-核心趋势)
3. [树形投机推理：现状与优化方案](#三树形投机推理现状与优化方案)
   - [3.1 现状：树形投机已被放弃](#31-现状树形投机已被放弃)
   - [3.2 树形投机推理的潜在优化方案](#32-树形投机推理的潜在优化方案)
4. [附录：关键源码位置速查](#附录关键源码位置速查)

---

## 一、当前支持的投机推理方法

两个代码库支持的投机方法基本对齐，vllm-ascend 通过 `AscendEagleProposer` 等 NPU 适配层进行了移植。当前主流方法可分为四大类：

### 1.1 无模型方法（Model-free）

| 方法 | 核心特点 | 优势场景 |
|------|---------|---------|
| **N-gram (CPU)** | 在 prompt + 已生成 token 中匹配 n-gram 模式，用 Numba JIT 编译加速。无需加载任何草稿模型。 | 文本重复度高的任务（代码生成、结构化输出、模板填充）。**零显存开销**。 |
| **N-gram GPU/NPU** | GPU/NPU 张量并行匹配（`unfold` + `argmax`），含异步 D2H trim。vllm-ascend 额外实现了 **Ascend C 自定义算子** 版本。 | 比 CPU 版本延迟更低，适合对 draft 速度敏感的高并发场景。 |
| **Suffix Decoding** | 基于 `arctic-inference` 构建全局 + prompt 后缀树，利用**频率计数**选择最可能的后续 token，支持自适应推测长度。 | 代码编辑、Agentic 循环（self-reflection、self-consistency）、RL rollouts 等高度重复性任务。 |

**配置示例：**
```python
# N-gram
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_max": 4,
}

# Suffix Decoding
speculative_config={
    "method": "suffix",
    "num_speculative_tokens": 15,
}
```

### 1.2 独立草稿模型方法

| 方法 | 核心特点 | 优势场景 |
|------|---------|---------|
| **Draft Model** | 使用独立的小模型做自回归 draft。要求 draft TP size = target TP size（避免 torch.compile 缓存污染）。 | 通用场景，但需要额外维护一个蒸馏/小参数模型。 |
| **Medusa** | 在目标模型顶部挂载多个预测头，同时预测多个未来 token，用 argmax 采样。 | 不想加载额外模型、可接受修改模型结构的场景；低延迟需求。 |
| **MLP Speculator** | 轻量级 MLP-based 推测器，TP 限制为 1。 | 特定模型架构下的轻量加速。 |

### 1.3 基于目标模型隐藏状态的方法（主流）

| 方法 | 核心特点 | 优势场景 |
|------|---------|---------|
| **EAGLE** | 基于目标模型最后一层 hidden state，接单个 future token prediction head。支持 embedding/LM head 权重共享、CUDA Graph/ACL Graph。 | **生产环境最常用**的通用 draft 方案；接受率高，与目标模型耦合紧密。 |
| **EAGLE-3** | EAGLE 的增强版，引入目标模型**中间层 auxiliary hidden states**（通过 `eagle_aux_hidden_state_layer_ids` 配置）作为 draft 模型输入。 | 对接受率要求极高的生产部署；DeepSeek、Qwen 等大模型场景。 |
| **MTP** | 模型原生多 token 预测层。已支持 DeepSeek V3/V4、Qwen3.5、MiMo、GLM-4、ERNIE、Nemotron-H、ExaOne、Pangu Ultra、Step 3.5 等十余种模型架构。 | **首选方案**（如果模型原生支持 MTP）。无需额外 draft 模型，接受率通常最高。 |
| **DFlash** | 交叉注意力 draft：context K/V 来自目标模型 hidden states，query 来自 draft embedding（bonus + mask tokens）。使用**非因果注意力**（`causal=False`），强制启用 `parallel_drafting`。 | 专为 **Qwen3.5 系列**优化，实测 GSM8K 准确率可达 93.40%，接受率 [0.91, 0.80, 0.69, 0.59]。 |
| **Gemma4 MTP** | Gemma4 的 assistant model，运行所有 decoder layers，支持与目标模型**跨模型 KV cache 共享**（cross-model KV sharing）。 | Gemma4 模型专用。 |
| **PEagle** | Parallel Eagle，刚加入支持（`PeagleLlamaForCausalLM`）。使用 `mask_token_id` 做并行 draft 占位。 | 进一步降低 draft 延迟的并行化 Eagle 变体。 |

**配置示例：**
```python
# EAGLE
speculative_config={
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "draft_tensor_parallel_size": 1,
    "num_speculative_tokens": 2,
}

# EAGLE-3
speculative_config={
    "method": "eagle3",
    "model": "...",
    "num_speculative_tokens": 3,
}

# MTP / DeepSeek MTP
speculative_config={
    "method": "deepseek_mtp",
    "num_speculative_tokens": 2,
}

# DFlash
speculative_config={
    "method": "dflash",
    "num_speculative_tokens": 4,
}
```

### 1.4 训练/辅助方法

| 方法 | 核心特点 |
|------|---------|
| **Extract Hidden States** | 非真正投机。从目标模型指定层提取 hidden states 保存为 `.safetensors`，用于训练 EAGLE 风格 draft 模型。`num_speculative_tokens` 强制为 1。 |
| **Custom Class** | 运行时动态导入用户自定义 proposer 类（实验性）。 |

---

## 二、后续迭代方向与在推进的工作

### 2.1 上游 vLLM 核心趋势

1. **Model Runner V2 全面迁移**  
   投机解码的主战场已转向 V2 model runner。近期大量 commit 围绕 V2 的 attention metadata 重建、CUDA Graph 支持、MTP 权重共享展开。这是 vLLM 未来的主架构演进方向。

2. **可插拔 Proposer 架构**  
   自定义 callable proposer backend（commit `256dbcaab`）已落地，意味着投机解码后端正在**模块化**，用户可更灵活地注入自定义 draft 逻辑，而无需修改核心引擎。

3. **树注意力已被彻底移除（方向性信号）**  
   - 2026-05-08 的 commit `b1728c1e6` **[Attention][Cleanup] Remove tree attention** 一次性删除了 **1399 行代码**，包括 `vllm/v1/attention/backends/tree_attn.py`（488 行）及全部相关测试。  
   - 这是一个强烈的架构决策信号：**vLLM 不再走"树注意力"路线**，而是收敛到链式/并行 draft + 高效验证的范式。

4. **PEagle 与 DFlash 持续演进**  
   PEagle（`6ccb10d79`）刚加入；DFlash 在 ROCm 适配、FP8 KV-Cache 修复、多模态输入等方面持续迭代。

5. **平台一致性**  
   ROCm/AMD 侧投入巨大（AITER MLA + Eagle3、DFlash on ROCm），说明投机解码正在跨平台标准化。

6. **生产级硬化**  
   支持 thinking budget（`68dd7db81`）、CI 增加 MTP correctness 测试、Eagle DP 测试、 nightly B200 测试等。

### 2.2 vllm-ascend 核心趋势

1. **EAGLE3/MTP 生产级稳定性（最高优先级）**  
   大量 PR（#9158、#8718、#7290、#7924 等）修复 eagle3 维度不匹配、MTP batch size 不一致、PCP（context parallel）兼容等问题。目标很明确：**支撑 DeepSeek 系列模型的生产部署**。

2. **DFlash Qwen3.5 专用优化**  
   #9074 merged graph 优化使 TPOT 降低 5~10%；但 `spec_num` 目前限制为 1~7（≥8 会因 GDN 算子变化导致精度严重下降），这是正在攻克的瓶颈。

3. **自定义 NPU 原生算子（性能突破）**  
   #9008 实现了 `ngram_spec_decode` Ascend C 自定义算子，并适配 async scheduler：
   - ngram + sync: TTFT 330ms, TPOT 39ms
   - **ngram + async: TTFT 136ms, TPOT 33ms**
   
   这说明 vllm-ascend 正从"GPU 逻辑移植"走向"NPU 原生优化"。

4. **ACL Graph（CUDA Graph 等价物）全覆盖**  
   MTP、DFlash 均已支持 merged graph（`_run_merged_draft` 纳入 graph），这是 NPU 上降低 draft 开销的关键。

5. **异步调度器 + 投机解码深度融合**  
   Zero bubble async scheduling + spec decode（#7640）正在推进；ngram_npu 已完整通过 async 模式测试。

6. **训练管线闭环**  
   `extract_hidden_states` 方法支持在 Ascend 上直接产出 EAGLE 训练数据，形成"训练 → 部署"的闭环。

7. **QuaRot + Eagle3**  
   支持加载经 QuaRot 旋转量化后的 Eagle3 draft 权重，与量化目标模型配合工作。

---

## 三、树形投机推理：现状与优化方案

### 3.1 现状：树形投机已被放弃

- **vLLM 已移除树注意力**：如上所述，`b1728c1e6` 彻底删除了 tree attention backend 及所有相关测试。
- **当前全是链式/并行**：
  - EAGLE/MTP：**sequential chain drafting**（`for step in range(num_speculative_steps)`，每步生成一个 token）
  - DFlash：**parallel non-causal**（一次 forward 出多个 token，但仍是线性序列，无分支）
  - Medusa：multi-head 并行生成，但验证是线性的
- **代码中的残留痕迹**：
  - `vllm/config/speculative.py` 中 `use_local_argmax_reduction` 注释明确写："Only applies to greedy draft selection in **non-tree speculation**"
  - `llm_base_proposer.py:1449` 有 FIXME："when using tree-based specdec, adjust number of forward-passes according to the depth of the tree"——说明设计者曾预留接口，但未实现。
  - `vllm_ascend/spec_decode/eagle_proposer.py:1059` 有 TODO："get more than one token for tree attention"——同样只是预留想法。

### 3.2 树形投机推理的潜在优化方案

虽然代码中已无实现，但从架构残留和学术前沿（如 EAGLE-2、Medusa 的 tree verification、Sequoia 等）来看，如果未来要重新引入，优化方向包括：

#### 3.2.1 树注意力后端重建

- **FlexAttention 路线**：`FlexAttentionBackend` 的 `mask_mod` + `block_mask` 系统是当前最可行的 hook。需要实现将 per-request 树拓扑（parent pointers 或 depth-level indices）转换为 block-sparse mask。
- **稀疏掩码优化**：利用树结构天然的稀疏性，避免计算无效分支的注意力对，降低验证阶段的 FLOPs。

#### 3.2.2 树构建与剪枝算法

- **Top-k 分支扩展**：从 draft model 的 top-k 输出构建 token tree（类似 EAGLE-2 或 Medusa trees）。
- **动态概率剪枝**：根据分支累积概率或阈值实时剪枝低概率路径，在"接受率"和"验证开销"之间做 Pareto 最优。
- **批量树复用**：利用 batch 内多个请求在语义上的相似性，共享公共树前缀，减少重复 draft 计算。

#### 3.2.3 树验证算法

- **树形拒绝采样（Tree-based Rejection Sampling）**：当前 rejection sampler 是线性遍历（`for i in range(num_tokens - 1)`），树形验证需要支持**祖先条件概率**累积，一次性验证多条路径。
- **路径合并与 KV 去重**：验证后合并公共前缀的 KV cache，避免各分支重复存储相同前缀的 key/value，这对长树深度场景至关重要。

#### 3.2.4 与 Graph 模式的兼容性（最大难点）

树形投机与 CUDA Graph / ACL Graph 的"统一张量形状"假设根本冲突：

- **Padded Tree Shapes**：固定最大树宽和深度，用 padding token 填充，使每轮 graph 输入形状恒定。代价是浪费算力在 pad 上。
- **Depth-based Graph Capture**：`llm_base_proposer.py` 中的 FIXME 已指出——应以 **tree depth** 而非 `num_speculative_tokens` 决定 forward pass 次数和 graph 规模，显著降低 capture 开销。
- **Graph 回退策略**：对深度/宽度超阈值的请求自动回退到 eager 模式，保证功能正确性。

#### 3.2.5 NPU 特定优化（vllm-ascend 视角）

- **自定义树形输入扩展算子**：类似现有的 `copy_and_expand_eagle_inputs` CANN 算子，可扩展为 `copy_and_expand_tree_inputs`，在 host 侧完成树拓扑到张量 layout 的映射。
- **Cube/Core 架构适配**：NPU 的 Cube 单元对矩阵乘法友好，可将树注意力中的多个小分支合并为大的 batch GEMM，提升利用率。
- **异步调度兼容**：树分支验证可以与 async scheduler 的微批次调度结合，不同分支分配到不同 micro-batch 并行验证。

### 3.3 结论

**树形投机推理在当前 vLLM/vllm-ascend 中已无实际实现**，且 upstream 明确删除了相关代码。社区正通过 **PEagle、DFlash、并行 drafting、MTP** 等"宽而浅"的并行/链式方案替代传统的"窄而深"的树形分支验证。

如果业务场景需要树形投机的理论优势，需要自行基于 FlexAttention 或自定义 kernel 重建，且首要解决的是**与 CUDA/ACL Graph 的兼容性**问题。

---

## 附录：关键源码位置速查

### 上游 vLLM

| 组件 | 路径 |
|------|------|
| 配置中心 | `vllm/config/speculative.py` |
| EAGLE Proposer | `vllm/v1/spec_decode/eagle.py` |
| 基础 Proposer | `vllm/v1/spec_decode/llm_base_proposer.py` |
| DFlash Proposer | `vllm/v1/spec_decode/dflash.py` |
| Gemma4 MTP | `vllm/v1/spec_decode/gemma4.py` |
| Medusa Proposer | `vllm/v1/spec_decode/medusa.py` |
| Suffix Decoding | `vllm/v1/spec_decode/suffix_decoding.py` |
| N-gram Proposer | `vllm/v1/spec_decode/ngram_proposer.py` |
| N-gram GPU | `vllm/v1/spec_decode/ngram_proposer_gpu.py` |
| EAGLE Speculator | `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` |
| Rejection Sampler | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` |
| Rejection Sampler Utils | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` |
| Speculator Config | `vllm/transformers_utils/configs/speculators/algos.py` |
| Eagle3 Model | `vllm/model_executor/models/llama_eagle3.py` |

### vllm-ascend

| 组件 | 路径 |
|------|------|
| 方法注册中心 | `vllm_ascend/spec_decode/__init__.py` |
| Ascend 基础 Proposer | `vllm_ascend/spec_decode/eagle_proposer.py` |
| Draft Model Proposer | `vllm_ascend/spec_decode/draft_proposer.py` |
| DFlash Proposer | `vllm_ascend/spec_decode/dflash_proposer.py` |
| N-gram Proposer | `vllm_ascend/spec_decode/ngram_proposer.py` |
| N-gram NPU | `vllm_ascend/spec_decode/ngram_proposer_npu.py` |
| Suffix Proposer | `vllm_ascend/spec_decode/suffix_proposer.py` |
| Medusa Proposer | `vllm_ascend/spec_decode/medusa_proposer.py` |
| Hidden States Proposer | `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` |
| v2 Eagle Speculator | `vllm_ascend/worker/v2/spec_decode/eagle/speculator.py` |
| v2 ACL Graph | `vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py` |
| Rejection Sampler Utils | `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` |
| DeepSeek MTP Patch | `vllm_ascend/patch/worker/patch_deepseek_mtp.py` |
| Qwen3.5 MTP Patch | `vllm_ascend/patch/worker/patch_qwen3_next_mtp.py` |
| QuaRot Eagle3 Patch | `vllm_ascend/patch/worker/patch_draft_quarot.py` |
| 自定义 NPU Op | `vllm_ascend/csrc/moe/copy_and_expand_eagle_inputs/` |
| 用户文档 | `vllm-ascend/docs/source/user_guide/feature_guide/speculative_decoding.md` |

---

*本报告基于对 vLLM 及 vllm-ascend 源码的静态分析生成，部分推断性内容建议结合官方文档或源码进一步确认。*
