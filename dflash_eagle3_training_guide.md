# DFlash 和 Eagle-3 模型训练指南

> 基于 vLLM Speculators 框架

## 目录

1. [概述](#概述)
2. [算法对比](#算法对比)
3. [环境准备](#环境准备)
4. [训练 Eagle-3 模型](#训练-eagle-3-模型)
5. [训练 DFlash 模型](#训练-dflash-模型)
6. [部署与测试](#部署与测试)
7. [常见问题](#常见问题)

---

## 概述

**Speculators** 是 vLLM 项目开发的投机解码（Speculative Decoding）草稿模型训练框架。投机解码是一种无损加速技术，通过使用更小、更快的草稿模型（即"speculator"）来预测 token，然后由更大的基础模型进行验证，从而在保持输出质量的同时减少延迟。

### 核心特性

- **Online/Offline 训练支持**：实时生成 hidden states 或预生成后训练
- **多架构支持**：支持 MoE、非 MoE 和视觉语言模型
- **标准化格式**：Hugging Face 兼容格式，易于部署
- **无缝 vLLM 集成**：训练完成的模型可直接部署到 vLLM

---

## 算法对比

| 特性 | **Eagle-3** | **DFlash** |
|------|-------------|------------|
| 架构 | 自回归 (Llama-style) | 块扩散 (Qwen3-style) |
| 注意力机制 | 因果掩码 | 双向非因果掩码 |
| 草稿方式 | 逐步自回归预测 | 单步块并行预测 |
| 典型层数 | 1层 | 5层 |
| 理论加速比 | ~1.5-2x | ~2-3x |
| 适用场景 | 通用场景 | 同步请求场景 |
| 训练难度 | 较简单 | 较复杂 |

### 架构示意图

**Eagle-3 架构**：
```
目标模型 → Hidden States → FC 投影 + Token Embedding → 
Llama 解码层 → LM Head → Draft Logits
```

**DFlash 架构**：
```
目标模型 → Hidden States → Mask Token Embeddings → 
双向注意力层 → LM Head → 块预测 Logits
```

---

## 环境准备

### 系统要求

- **操作系统**：Linux 或 macOS
- **Python**：3.10+
- **CUDA**：支持 CUDA 的 GPU
- **内存**：根据模型大小，建议 32GB+

### 安装步骤

创建两个独立的虚拟环境（避免依赖冲突）：

```bash
# 1. Speculators 环境（数据准备 + 训练）
uv venv speculators_venv
source speculators_venv/bin/activate
uv pip install "speculators>=0.5.0"

# 如果使用实验追踪工具（可选）
uv pip install wandb tensorboard
```

```bash
# 2. vLLM 环境（服务目标模型）
uv venv vllm_venv
source vllm_venv/bin/activate
uv pip install "vllm>=0.18"
```

### 验证安装

```bash
speculators --version
```

---

## 训练 Eagle-3 模型

### 方式一：Online 训练（推荐）

**特点**：训练时实时从 vLLM 获取 hidden states

**适用场景**：GPU 资源充足，需要快速迭代

```bash
#!/bin/bash
set -euo pipefail

# ============ 配置 ============
MODEL="Qwen/Qwen3-8B"              # 目标模型（验证器）
DATASET="sharegpt"                  # 数据集选项
OUTPUT_DIR="./output/eagle3_qwen3_8b"
VLLM_PORT=8000
MAX_SAMPLES=5000                    # 测试用，生产环境建议 100K+
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4
DRAFT_VOCAB_SIZE=32000

# GPU 分配
VLLM_GPUS="0,1"                     # vLLM 使用 GPU 0,1
TRAIN_GPUS="2,3"                    # 训练使用 GPU 2,3
NUM_TRAIN_GPUS=2
# =============================

# Step 1: 数据准备
echo "=== Step 1: 准备数据 ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: 启动 vLLM 服务
echo "=== Step 2: 启动 vLLM 服务 ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "停止 vLLM 服务..."
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 等待服务就绪
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM 服务已就绪"

# Step 3: 训练
echo "=== Step 3: 开始训练 ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing generate \
    --on-generate delete

echo "训练完成！检查点保存在 $OUTPUT_DIR/checkpoints/"
```

### 方式二：Offline 训练

**特点**：预先生成所有 hidden states，然后训练

**适用场景**：GPU 资源有限，需要最大化训练效率

```bash
#!/bin/bash
set -euo pipefail

# ============ 配置 ============
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="ultrachat"
OUTPUT_DIR="./output/eagle3_llama3_8b"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4
CONCURRENCY=32

GPUS="0,1"
NUM_GPUS=2
# =============================

# Step 1: 数据准备
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT_DIR" \
    --seq-length "$SEQ_LENGTH"

# Step 2: 启动 vLLM
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/launch_vllm.py "$MODEL" \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do sleep 2; done

# Step 3: 预生成 hidden states
echo "=== Step 3: 生成 hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

# Step 4: 停止 vLLM 释放 GPU
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true

# Step 5: 使用预生成的 hidden states 训练
echo "=== Step 5: 开始训练 ==="
CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size 32000 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --on-missing raise
```

---

## 训练 DFlash 模型

**关键区别**：DFlash 必须显式指定 `--target-layer-ids` 参数

```bash
#!/bin/bash
set -euo pipefail

# ============ 配置 ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"
OUTPUT_DIR="./output/dflash_qwen3_8b"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# DFlash 特有参数
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8                        # 每块预测的 token 数
MAX_ANCHORS=3072                    # 最大锚点位置数
NUM_LAYERS=5                        # 草稿模型层数
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"          # 必须从目标模型提取的层

VLLM_GPUS="0,1"
TRAIN_GPUS="2,3"
NUM_TRAIN_GPUS=2
# =============================

# Step 1: 数据准备
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: 启动 vLLM（注意：必须指定 --target-layer-ids）
echo "=== Step 2: 启动 vLLM 服务 ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do sleep 2; done

# Step 3: 训练（DFlash 特有参数）
echo "=== Step 3: 开始训练 ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --on-missing generate \
    --on-generate delete
```

---

## 参数详解

### 通用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--verifier-name-or-path` | 目标模型路径 | `Qwen/Qwen3-8B` |
| `--data-path` | 预处理数据路径 | `./output` |
| `--save-path` | 检查点保存路径 | `./output/checkpoints` |
| `--draft-vocab-size` | 草稿词汇表大小 | `32000` |
| `--epochs` | 训练轮数 | `5` |
| `--lr` | 学习率 | `1e-4` (Eagle), `3e-4` (DFlash) |
| `--total-seq-len` | 最大序列长度 | `8192` |
| `--num-layers` | 草稿模型层数 | `1` (Eagle), `5` (DFlash) |

### DFlash 特有参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--speculator-type` | 算法类型 | `dflash` |
| `--block-size` | 每块预测的 token 数 | `8` |
| `--max-anchors` | 最大锚点位置数 | `3072` |
| `--target-layer-ids` | 提取 hidden states 的层 | `2 18 33` |

### Online/Offline 参数

| 参数 | 说明 | 选项 |
|------|------|------|
| `--on-missing` | hidden states 缺失时的处理 | `generate` / `raise` |
| `--on-generate` | 生成后的处理 | `delete` / `cache` |
| `--vllm-endpoint` | vLLM 服务端点 | `http://localhost:8000/v1` |
| `--hidden-states-path` | 预生成 hidden states 路径（Offline） | `./hidden_states` |

---

## 部署与测试

### 部署到 vLLM

```bash
# 使用训练好的模型
vllm serve ./output/eagle3_qwen3_8b/checkpoints/checkpoint_best --port 8000

# 或 DFlash 模型
vllm serve ./output/dflash_qwen3_8b/checkpoints/checkpoint_best --port 8000
```

### 测试对话

```bash
# 启动对话
vllm chat --url http://localhost:8000/v1
```

### 检查点结构

```
checkpoints/
├── 0/                              # Epoch 0
│   ├── config.json                 # 模型配置
│   ├── model.safetensors           # 模型权重
│   ├── optimizer_state_dict.pt     # 优化器状态
│   └── scheduler_state_dict.pt     # 学习率调度器状态
├── 1/                              # Epoch 1
├── ...
└── checkpoint_best -> 4/           # 最佳检查点符号链接
```

---

## 常见问题

### 1. Out of Memory（训练时）

**解决方案**：
```bash
# 减少序列长度
python scripts/train.py --total-seq-len 4096 ...

# 减少草稿模型层数
python scripts/train.py --num-layers 1 ...
```

### 2. Out of Memory（vLLM 时）

**解决方案**：
```bash
# 增加 GPU 内存利用率
python scripts/launch_vllm.py model -- --gpu-memory-utilization 0.95

# 减少最大模型长度
python scripts/launch_vllm.py model -- --max-model-len 4096

# 使用更多 GPU
python scripts/launch_vllm.py model -- --tensor-parallel-size 2
```

### 3. 训练 Loss 不下降

**可能原因和解决方案**：
1. 学习率过高：`--lr 1e-5`
2. 数据质量问题：检查预处理输出
3. 训练时间不足：`--epochs 20`

### 4. GPU 利用率不稳定

**解决方案**：
1. 重新分配 GPU：增加 vLLM 的 GPU 数量，减少训练 GPU 数量
2. 切换到 Offline 训练模式

---

## 性能参考

### 5K 样本测试结果（4x H100）

| 指标 | Eagle-3 | DFlash |
|------|---------|--------|
| 训练时间 | ~17 分钟 | ~25 分钟 |
| 接受率 | 14.88% | 5.90% |
| 接受长度 | 1.45 | 1.47 |
| 输出吞吐 | 143 tok/s | 129 tok/s |

**注意**：5K 样本仅用于验证流程，生产环境建议使用 **100K+** 样本训练。

---

## 相关资源

- **项目主页**：https://github.com/vllm-project/speculators
- **官方文档**：https://docs.vllm.ai/projects/speculators/
- **Eagle-3 论文**：https://arxiv.org/abs/2401.15077
- **DFlash 论文**：https://arxiv.org/abs/2602.06036

---

## 预训练模型

可从 HuggingFace 下载已训练的模型：

| 验证器模型 | Eagle-3 | DFlash |
|-----------|---------|--------|
| Qwen3-8B | [链接](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.eagle3) | [链接](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash) |
| Llama-3.1-8B | [链接](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3) | - |
| Gemma-4-31B | [链接](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3) | [链接](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.dflash) |

---

*文档生成时间：2026-05-26*
*基于 Speculators 项目文档*
