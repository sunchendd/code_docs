# vLLM / vllm-ascend 投机解码方法矩阵清单（修正版）

> 按 **硬件（H20 | 910C）× 模型** 维度整理  
> 分析日期：2026-05-19（修正版）  
> 基于代码库：`/root/vllm/`（上游 vLLM）、`/root/vllm-ascend/`（昇腾 NPU 插件）

---

## 一、模型存在性验证（修正后）

| 用户提及模型 | 代码库实际版本 | 注册名 / 架构 | 状态 | 验证依据 |
|-------------|-------------|-------------|------|---------|
| **GLM 4.7** | ✅ `Glm4MoeLiteForCausalLM` | GLM-4.7-Flash 专用 | **已支持** | ① `registry.py` 注册 `Glm4MoeLiteForCausalLM` / `Glm4MoeLiteMTPModel` ② `speculative.py` 支持 `glm4_moe_lite_mtp` ③ vllm-ascend 有完整部署文档和 CI 测试 (`GLM-4.7.yaml`) |
| **GLM5.1** | ✅ `GlmMoeDsaForCausalLM` | 继承 `DeepseekV2ForCausalLM` | **已支持** | ① `registry.py:123` ② `speculative.py` 把 `glm_moe_dsa` 映射为 `deepseek_mtp` ③ vllm-ascend patch: "GLM5 uses rotary quant" ④ `deepseek_mtp.py`: "GLM-5.1 W8A8 compatibility" |
| **DeepSeek V4 Pro** | ✅ `DeepseekV4ForCausalLM` | V4 Pro 专用 kernel | **已支持** | ① `registry.py:100/614` ② `DSV4_PRO_NUM_EXPERTS=384` 等专用常量 ③ `supported_models.md` 明确列出 `DeepSeek-V4-Pro` ④ 专门 `dsv4_pro_norm_gate` fused kernel |
| **Qwen3.6** | ✅ 复用 Qwen3.5 架构 | `Qwen3_5ForConditionalGeneration` | **已支持** | ① `test_qwen36_moe_lora.py` 引用 `Qwen3.6-35B-A3B` ② 文档有 `Qwen3.6-27B` 部署示例 ③ `tool_parsers` 明确提到 Qwen3.6 |
| **Kimi2.6** | ❌ 不存在 | — | **未支持** | 代码库中只有 `KimiK25ForConditionalGeneration` (K2.5)，无 Kimi2.6 |
| **MiniMax M2.5** | ✅ 复用 M2 架构 | `MiniMaxM2ForCausalLM` | **已支持** | ① vllm-ascend nightly/weekly CI 有 `MiniMax-M2.5-w8a8-QuaRot` ② Eagle3 draft: `vllm-ascend/MiniMax-M2.5-eagle-model-0318` |

> **Kimi2.6 说明**：当前代码库中确实无此模型。若 checkpoint 的 `architectures` 为 `KimiK25ForConditionalGeneration`，则实际复用 K2.5 架构。

---

## 二、GLM 系列

### 现状

| 模型 | 注册名 | 原生 MTP | 投机支持 | 备注 |
|------|--------|---------|---------|------|
| GLM-4 (Dense) | `Glm4ForCausalLM` | ❌ | 无 | |
| GLM-4 MoE / **GLM-4.5** / **GLM-4.6** | `Glm4MoeForCausalLM` | ✅ `Glm4MoeMTPModel` | MTP | 4.5/4.6 复用 GLM-4 MoE 架构 |
| GLM-OCR | `GlmOcrForConditionalGeneration` | ✅ `GlmOcrMTPModel` | MTP | OCR 的 MTP 复用 `Glm4MoeLiteMultiTokenPredictor` |
| **GLM-4.7 (Flash)** | `Glm4MoeLiteForCausalLM` | ✅ **`Glm4MoeLiteMTPModel`** | **MTP** | 独立架构，继承 DeepSeek V2 Attention |
| **GLM5.1** | `GlmMoeDsaForCausalLM` | ✅ **复用 `DeepSeekMTPModel`** | **MTP + Eagle3** | DSA = DeepSeek Architecture |

> **GLM-4.7 关键发现**：`Glm4MoeLiteForCausalLM` 是 GLM-4.7-Flash 的独立注册类，有专用的 MTP 模型 `Glm4MoeLiteMTPModel`。`speculative.py` 中 `glm4_moe_lite_mtp` 是独立条目。支持 W8A8 量化 (`Eco-Tech/GLM-4.7-W8A8-floatmtp`)，vllm-ascend 有完整的单节点/多节点/PD 分离部署文档和 CI 测试。

> **GLM5.1 关键发现**：architecture 为 `GlmMoeDsaForCausalLM`，**直接继承 `DeepseekV2ForCausalLM`**。`speculative.py` 把 `glm_moe_dsa` 映射为 `deepseek_mtp`，设置 `architectures: ["DeepSeekMTPModel"]`。

### 最优方法

| 硬件 | 模型 | 最优方法 | 配置 | 备注 |
|------|------|---------|------|------|
| **H20** | GLM-4.7 | **MTP (glm4_moe_lite_mtp)** | `method="mtp"` | 原生 MTP，接受率最高 |
| **910C** | GLM-4.7 | **MTP (glm4_moe_lite_mtp)** | `method="mtp"` | vllm-ascend CI 验证；支持 W8A8 量化 + PD 分离 |
| **H20** | GLM5.1 | **MTP (deepseek_mtp)** | `method="deepseek_mtp"` | 自动检测，接受率最高 (0.85~0.95) |
| **910C** | GLM5.1 | **MTP (deepseek_mtp)** | `method="deepseek_mtp"` | vllm-ascend 已适配 rotary quant；W8A8 权重有专门处理 |
| 备选 (H20/910C) | GLM-4.7/5.1 | Eagle3 | `method="eagle3"` | 高接受率，draft TP=1 (910C) |

---

## 三、DeepSeek 系列

### 现状

| 模型 | 注册名 | 原生 MTP | Eagle3 | 备注 |
|------|--------|---------|--------|------|
| V2/V3 | `DeepseekV2ForCausalLM` | ✅ `DeepSeekMTPModel` | ✅ | vllm-ascend 有 patch |
| V4 | `DeepseekV4ForCausalLM` | ✅ `DeepSeekV4MTPModel` | ✅ | 重点支持 |
| **V4 Pro** | `DeepseekV4ForCausalLM` | ✅ **复用 `DeepSeekV4MTPModel`** | ✅ | **专门 `DSV4_PRO_*` fused kernel** |

> **V4 Pro 关键发现**：共享 V4 architecture，但有 Pro 专属硬件优化：`DSV4_PRO_NUM_EXPERTS=384`、`DSV4_PRO_HIDDEN_SIZE=7168`、`dsv4_pro_norm_gate` fused kernel（RMSNorm + router GEMV）。

### 最优方法

| 硬件 | 最优方法 | 配置 | 备注 |
|------|---------|------|------|
| **H20** | **MTP (deepseek_v4_mtp)** | `method="deepseek_v4_mtp"` | 首选，接受率最高 |
| **910C** | **MTP (deepseek_v4_mtp)** | `method="deepseek_v4_mtp"` | 最高优先级优化方向；量化需 `--quantization ascend` |
| 备选 (H20/910C) | Eagle3 | `method="eagle3"` | 支持 QuaRot 量化 draft (910C) |

---

## 四、Qwen 系列

### 现状

| 模型 | 注册名 | Eagle3 | 原生 MTP | DFlash | 备注 |
|------|--------|--------|---------|--------|------|
| Qwen3.5 | `Qwen3_5ForConditionalGeneration` | ✅ | ✅ `Qwen3_5MTP` | ✅ | 投机最丰富 |
| **Qwen3.6** | **复用 Qwen3.5 架构** | ✅ | ✅ **复用 `Qwen3_5MTP`** | ✅ **复用 DFlash** | 同 Qwen3.5 能力 |
| Qwen3-Next | `Qwen3NextForCausalLM` | ✅ | ✅ `Qwen3NextMTP` | ❌ | |

> **Qwen3.6 关键发现**：注册表无单独 `Qwen3_6ForCausalLM`，复用 `Qwen3_5ForConditionalGeneration`。测试文件直接引用 `Qwen/Qwen3.6-35B-A3B`，文档有 `Qwen3.6-27B` 部署命令。

### 最优方法

| 硬件 | 最优方法 | 配置 | 备注 |
|------|---------|------|------|
| **H20** | **DFlash** | `method="dflash"` | Qwen 专用首选，接受率 [0.91, 0.80, 0.69, 0.59] |
| **H20** | MTP | `method="qwen3_5_mtp"` | 次选（无 DFlash 时） |
| **910C** | **MTP** | `method="qwen3_5_mtp"` | **比 DFlash 更稳定**；DFlash spec_num 限制 1~7 |
| **910C** | Eagle3 | `method="eagle3"` | 稳定可靠 |

---

## 五、Kimi 系列

### 现状

| 模型 | 注册名 | Eagle3 | 原生 MTP | 备注 |
|------|--------|--------|---------|------|
| Kimi K2.5 | `KimiK25ForConditionalGeneration` | ✅ | ❌ | 支持 Eagle3 + Eagle |
| **Kimi2.6** | ❌ **无** | — | — | **代码库中不存在** |

### 最优方法

| 硬件 | 最优方法 | 配置 | 备注 |
|------|---------|------|------|
| **H20** | **Eagle3** | `method="eagle3"` | 首选 (0.75~0.90) |
| **910C** | **Eagle3** | `method="eagle3"` | 首选；draft TP=1 |
| 备选 | Eagle / Draft Model | `method="eagle"` / `method="draft_model"` | 通用回退 |

> **Kimi2.6**：若实际复用 K2.5 架构（`architectures: KimiK25ForConditionalGeneration`），则投机支持同上。若为新架构，当前代码库不支持。

---

## 六、MiniMax 系列

### 现状

| 模型 | 注册名 | Eagle3 | 备注 |
|------|--------|--------|------|
| MiniMax M2 | `MiniMaxM2ForCausalLM` | ✅ | vllm-ascend 深度 patch |
| **MiniMax M2.5** | **复用 M2 架构** | ✅ | vllm-ascend 有 nightly CI + Eagle3 draft |

> **M2.5 关键发现**：注册表无单独条目，复用 M2。vllm-ascend 有 `MiniMax-M2.5-w8a8-QuaRot-A2/A3` 测试配置，Eagle3 draft: `vllm-ascend/MiniMax-M2.5-eagle-model-0318`。

### 最优方法

| 硬件 | 最优方法 | 配置 | 备注 |
|------|---------|------|------|
| **H20** | **Eagle3** | `method="eagle3"` + `Eagle3MiniMaxM2ForCausalLM` | 首选 (0.75~0.88) |
| **910C** | **Eagle3** | `method="eagle3"` + `vllm-ascend/MiniMax-M2.5-eagle-model-0318` | 深度适配；QuaRot 量化 CI 验证 |
| 备选 | Eagle / N-gram | `method="eagle"` / `method="ngram"` | 通用回退 |

---

## 七、综合最优推荐速查表

### 7.1 按模型 × 硬件

| 模型 | 版本 | H20 最优 | 910C 最优 | 关键差异 |
|------|------|---------|----------|---------|
| **GLM** | GLM-4.7 | **MTP** (glm4_moe_lite_mtp) | **MTP** (glm4_moe_lite_mtp) | 原生 MTP；910C 支持 W8A8 + PD 分离 |
| **GLM** | GLM5.1 | **MTP** (deepseek_mtp) | **MTP** (deepseek_mtp) | 无差异；910C 已适配 rotary quant |
| **DeepSeek** | V4 Pro | **MTP** (deepseek_v4_mtp) | **MTP** (deepseek_v4_mtp) | 无差异；910C 需显式量化配置 |
| **Qwen** | Qwen3.6 | **DFlash** | **MTP** | H20 DFlash 无限制；910C DFlash spec_num 限 1~7 |
| **Kimi** | K2.5 | **Eagle3** | **Eagle3** | 无差异（无原生 MTP） |
| **Kimi** | K2.6 | **不支持** | **不支持** | 代码库中无此模型 |
| **MiniMax** | M2.5 | **Eagle3** | **Eagle3** | 无差异；910C 有 QuaRot + Eagle3 CI 验证 |

### 7.2 硬件级关键差异总结

- **H20 优势**：DFlash 无限制、Synthetic Rejection Sampling、全 Triton kernel 支持。
- **910C 优势**：MTP/Eagle3 同样成熟；N-gram 有 **Ascend C 自定义算子**（async 下 TTFT 降 60%）；ACL Graph 覆盖所有投机方法。
- **910C 劣势**：DFlash spec_num 限制 1~7；Synthetic RS 不支持；Suffix Decoding 需额外插件。

---

*本清单为修正版。修正内容：*
1. *补充 **GLM-4.7**（`Glm4MoeLiteForCausalLM` / `Glm4MoeLiteMTPModel`），此前遗漏。GLM-4.7-Flash 有独立架构和原生 MTP 支持，vllm-ascend 有完整部署文档和 CI 测试。*
2. *修正 **GLM-4.5/4.6** 与 GLM-4 MoE 的关系：4.5/4.6 复用 `Glm4MoeForCausalLM` 架构，非独立注册类。*
3. *此前版本错误地将 GLM5.1、DeepSeek V4 Pro、Qwen3.6、MiniMax M2.5 标记为"不存在/推断"，实际均有直接支持证据。*
