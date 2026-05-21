# SGLang v0.5.12 vs vLLM 投机解码特性矩阵

> 最后更新：2025-05-21

## 核心算法支持

| 算法/特性 | SGLang 0.5.12 | vLLM | 状态 | 备注 |
|:----------|:-------------:|:----:|:----:|:-----|
| **EAGLE** | ✅ | ✅ | 🟢 对齐 | 功能对等 |
| **EAGLE-3** | ✅ | ✅ | 🟢 对齐 | vLLM 支持 extract_hidden_states |
| **MTP (DeepSeek)** | ✅ | ✅ | 🟢 对齐 | vLLM 支持多种模型 |
| **MTP (Qwen/GLM/Gemma)** | ✅ | ✅ | 🟢 对齐 | vLLM 支持 15+ 模型 |
| **N-gram** | ✅ | ✅ | 🟢 对齐 | CPU + GPU 版本 |
| **Draft Model** | ✅ | ✅ | 🟢 对齐 | 独立 draft 模型 |
| **DFlash** | ✅ | ✅ | 🟢 对齐 | Qwen3 DFlash |
| **Medusa** | ❌ | ✅ | 🔵 vLLM 独有 | 多 head draft |
| **Suffix Decoding** | ❌ | ✅ | 🔵 vLLM 独有 | 基于后缀树 |
| **Multi-layer EAGLE** | ✅ | ❌ | 🟡 SGLang 独有 | 多层 draft |

## 性能优化特性

| 特性 | SGLang | vLLM | 状态 | 移植难度 |
|:------|:------:|:----:|:----:|:--------:|
| **Adaptive Spec Decode** | ✅ | ❌ | 🔴 需移植 | ⭐⭐⭐⭐ |
| **Spec V2 Architecture** | ✅ | ⚠️ | 🟡 部分 | ⭐⭐⭐ |
| **TreeMaskMode** | ✅ | ⚠️ | 🟡 基础版 | ⭐⭐ |
| **SWA + Spec** | ✅ | ❌ | 🔴 需移植 | ⭐⭐⭐ |
| **Two-Batch Overlap** | ✅ | ⚠️ | 🟡 部分 | ⭐⭐⭐ |
| **Parallel Drafting** | ✅ | ✅ | 🟢 对齐 | - |
| **CUDA Graphs** | ✅ | ✅ | 🟢 对齐 | - |
| **Piecewise CUDA Graph** | ✅ | ✅ | 🟢 对齐 | - |

## 模型特定优化

| 模型/特性 | SGLang | vLLM | 状态 | 备注 |
|:----------|:------:|:----:|:----:|:-----|
| **DeepSeek V4** | ✅ 完整 | ⚠️ 基础 | 🟡 部分 | SGLang 优化更多 |
| **DeepSeek V3.2 FP4** | ✅ PDL | ⚠️ | 🟡 部分 | SGLang FP4 更成熟 |
| **Kimi K2.5 MLA** | ✅ TokenSpeed | ⚠️ | 🟡 部分 | SGLang 专有 kernel |
| **GLM-5 FP4** | ✅ | ❌ | 🔴 需移植 | - |
| **Gemma 4 MTP** | ✅ | ✅ | 🟢 对齐 | - |
| **Intern-S2** | ✅ | ⚠️ | 🟡 未确认 | - |
| **MiniCPM-V 4.6** | ✅ | ⚠️ | 🟡 未确认 | VLM + Spec |

## 内存与缓存

| 特性 | SGLang | vLLM | 状态 | 移植难度 |
|:------|:------:|:----:|:----:|:--------:|
| **HiCache Framework** | ✅ | ❌ | 🔴 架构差异 | ⭐⭐⭐⭐⭐ |
| **UnifiedRadixTree** | ✅ | ❌ | 🔴 架构差异 | ⭐⭐⭐⭐⭐ |
| **HiSparse (CPU offload)** | ✅ | ⚠️ | 🟡 基础版 | ⭐⭐⭐⭐ |
| **SSD Offload** | ✅ | ⚠️ | 🟡 实验性 | ⭐⭐⭐ |
| **KV Cache 量化** | ✅ FP8 | ✅ FP8 | 🟢 对齐 | - |
| **Prefix Caching** | ✅ | ✅ | 🟢 对齐 | - |

## 多 GPU / 分布式

| 特性 | SGLang | vLLM | 状态 | 备注 |
|:------|:------:|:----:|:----:|:-----|
| **TP + Spec** | ✅ | ✅ | 🟢 对齐 | Tensor Parallel |
| **DP + Spec** | ✅ | ✅ | 🟢 对齐 | Data Parallel |
| **EP + Spec** | ✅ | ⚠️ | 🟡 部分 | Expert Parallel |
| **PP + Spec** | ✅ | ✅ | 🟢 对齐 | Pipeline Parallel |
| **PD Disaggregation + Spec** | ✅ | ⚠️ | 🟡 实验性 | Prefill-Decode |

## 内核优化

| Kernel/特性 | SGLang | vLLM | 状态 | 备注 |
|:------------|:------:|:----:|:----:|:-----|
| **FlashMLA** | ✅ | ✅ | 🟢 对齐 | MLA Attention |
| **DeepGemm** | ✅ | ✅ | 🟢 对齐 | MoE Gemm |
| **TokenSpeed MLA** | ✅ | ❌ | 🔴 SGLang 独有 | Blackwell 优化 |
| **MegaMoE Kernels** | ✅ | ⚠️ | 🟡 部分 | DeepSeek V4 |
| **TMA Bulk Store** | ✅ | ❌ | 🟡 未确认 | 内存优化 |
| **Custom AllReduce** | ✅ | ✅ | 🟢 对齐 | - |

## 配置与可观测性

| 特性 | SGLang | vLLM | 状态 | 备注 |
|:------|:------:|:----:|:----:|:-----|
| **Spec Metrics** | ✅ | ✅ | 🟢 对齐 | 接受率等 |
| **Adaptive Config** | ✅ JSON | ❌ | 🔴 需移植 | 动态配置 |
| **Custom Spec Registry** | ✅ | ⚠️ | 🟡 部分 | 自定义算法 |
| **Multi-detokenizer** | ✅ | ✅ | 🟢 对齐 | - |

## 移植路线图

### Phase 1 (v0.7.x) - 基础增强
```
优先级: 高 | 工作量: 2-3 周

□ TreeMaskMode 优化
  - QLEN_ONLY 模式
  - Bit-packing 支持
  - 内存占用 -30%

□ SWA + Spec Decode 基础
  - 滑动窗口感知
  - Draft token 裁剪

□ Adaptive 配置框架
  - 配置类定义
  - 基础 EMA 跟踪
```

### Phase 2 (v0.8.x) - 自适应投机
```
优先级: 高 | 工作量: 3-4 周

□ Adaptive Spec Decode 完整实现
  - 运行时步数切换
  - CUDA Graph 动态调整
  - 性能监控

□ Multi-layer EAGLE
  - 多层 draft 支持
  - Hidden state 传递
```

### Phase 3 (v0.9.x) - 深度优化
```
优先级: 中 | 工作量: 4-6 周

□ DeepSeek V4 投机优化
  - MTP + HiSparse 简化版
  - FP4 + Spec 集成

□ 高级 Tree Attention
  - 稀疏树结构
  - 动态树构建
```

### Phase 4 (未来) - 架构级特性
```
优先级: 低 | 工作量: 2+ 月

□ HiCache 简化版
  - 统一缓存管理
  - 分层 eviction

□ TokenSpeed MLA
  - Blackwell 优化
  - FP8 KV Cache
```

## 快速决策指南

### 如果 vLLM 用户需要...

| 需求 | 当前 vLLM 方案 | 建议 |
|:-----|:--------------|:-----|
| **基础投机加速** | EAGLE/MTP/N-gram | ✅ 已支持，直接使用 |
| **动态调整步数** | 固定 num_spec_tokens | 🟡 等待 Phase 2 |
| **长上下文 + 投机** | 有限支持 | 🟡 等待 SWA + Spec |
| **内存受限场景** | 基础 offload | 🟡 等待 HiSparse |
| **DeepSeek V4 最大性能** | 基础支持 | 🟡 可移植 SGLang 优化 |

## 参考资料

- SGLang Speculative Decoding: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative
- vLLM Speculative Decoding: https://docs.vllm.ai/en/latest/features/spec_decode.html
- SGLang v0.5.12 Release: https://github.com/sgl-project/sglang/releases/tag/v0.5.12
