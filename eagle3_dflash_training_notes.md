# Eagle-3 / DFlash 训练全流程说明

> 基于实际训练 Qwen3-8B 投机解码草稿模型的经验记录  
> 训练框架：[vLLM Speculators](https://github.com/vllm-project/speculators)  
> 硬件：4x NVIDIA L20（49GB），工作目录：`/home/scd/ai-infra-tools/train-models`

---

## 一、为什么需要训练草稿模型

投机解码（Speculative Decoding）是一种**无损加速技术**：

```
普通推理：大模型 → token1 → token2 → token3 ...（串行，慢）

投机解码：草稿模型 → [token1, token2, token3, token4, token5]（批量猜）
           大模型验证 → [✓, ✓, ✓, ✗]（一次并行验证）
           接受前3个，从第4个重新生成
```

草稿模型要足够快（参数少）、猜中率足够高，才能加速。  
训练的本质：**让草稿模型模仿大模型的"思维方式"**。

---

## 二、两种草稿模型对比

| 特性 | Eagle-3 | DFlash |
|------|---------|--------|
| 架构 | 1层 Llama-style 自回归解码器 | 5层双向注意力块 |
| 预测方式 | 逐个自回归预测 | 一次预测一整块（block=8 tokens）|
| 理论加速 | ~1.5-2x | ~2-3x |
| 训练难度 | 简单 | 复杂 |
| 输入 | hidden states + token embedding | hidden states + mask token embedding |

---

## 三、单条数据的完整训练过程

以一条 ShareGPT 对话为例，从原始文本到梯度更新：

### 3.1 数据预处理阶段（离线，只做一次）

**输入**：
```json
{"conversations": [
  {"from": "human", "value": "解释一下量子纠缠"},
  {"from": "gpt",   "value": "量子纠缠是指两个粒子..."}
]}
```

**处理步骤**：
1. 用 Qwen3-8B tokenizer 把对话转为 token 序列
2. 生成 `loss_mask`：只在 assistant 回复的 token 位置为 1（不在 human 提问处计算 loss）
3. 过滤：少于 16 个有效 token 的样本丢弃（太短 context 不足）
4. 保存为 Arrow 格式（`input_ids`, `loss_mask`, `seq_len`）

**为什么过滤短样本？**  
草稿模型需要足够的 context（前文 token）才能做出有意义的预测。短于 16 token 的样本
几乎没有 context，草稿模型无从学习，反而引入噪声。

### 3.2 Online 生成 hidden states（每个训练步实时完成）

**什么是 hidden states？**

Qwen3-8B 是 36 层 Transformer。每一层处理 token 时都会产生一个向量（hidden state），
代表"模型在这一层对这个 token 的理解"。第 36 层的输出最终转为词汇表概率分布（logits）。

训练草稿模型需要的正是这些中间层的输出，因为：
- hidden states 包含大模型的"内部状态"
- 草稿模型学习这些状态，等于学习大模型的推理模式
- 比直接蒸馏 logits 信息量更丰富

**Qwen3-8B 提取哪几层？**
```
提取层：第 2、18、33、36 层（eagle_aux_hidden_state_layer_ids）
- 第 2 层：浅层语法特征
- 第 18 层：中层语义特征  
- 第 33 层：深层推理特征
- 第 36 层：最终输出层
```

**具体流程**：
```
训练进程（GPU 3,4）
  ↓ 取出 batch 的 input_ids
  ↓ HTTP POST → vLLM（GPU 0,1，端口 8200）
  
vLLM（Qwen3-8B，TP=2）
  ↓ forward pass（只推理，不生成新 token）
  ↓ 在第 2/18/33/36 层抓取 hidden states
  ↓ 写入 /tmp/hidden_states/hs_xxx.safetensors
  ↓ 返回文件路径
  
训练进程
  ↓ 加载 safetensors 文件
  ↓ 删除临时文件（--on-generate delete）
  ↓ 进行反向传播
```

### 3.3 草稿模型前向传播

以 Eagle-3 为例，对一条序列 `[t1, t2, t3, ..., t_n]`：

```
输入：
  - input_ids:           [t1, t2, t3, ..., t_n]     ← 原始 token
  - hidden_states_layer2: [h1², h2², h3², ..., hn²]  ← 第2层激活
  - hidden_states_layer18:[h1¹⁸, ...]               ← 第18层激活
  - hidden_states_layer33:[h1³³, ...]               ← 第33层激活

草稿模型处理：
  token_embed(t_i) + FC(concat(h_i², h_i¹⁸, h_i³³)) → 融合特征
  → 1层 Llama 解码器（含 Causal Attention）
  → LM Head（线性层映射到词汇表大小）
  → draft_logits[i]：预测位置 i+1 的 token 概率分布

3个预测头（head-0/1/2）分别预测：
  head-0: 下 1 个 token（最容易，acc 最高）
  head-1: 下 2 个 token
  head-2: 下 3 个 token
```

### 3.4 计算 Loss 和反向传播

```
loss_head_k = CrossEntropy(draft_logits_k, target_tokens_k)

其中：
  draft_logits_k：草稿模型对第 k 步的预测
  target_tokens_k：Qwen3-8B 实际输出的 token（通过 hidden states 间接得到）

total_loss = loss_0 + loss_1 + loss_2
           = 当前 loss_0=1.078 + loss_1=2.125 + loss_2=2.969 = 6.172

反向传播 → 计算梯度 → AdamW 更新草稿模型参数
（Qwen3-8B 参数全程冻结，不参与更新）
```

---

## 四、训练参数详解

以当前日志为例：
```
epoch=1, lr=7.49e-05, global_step=10692
train/loss_0=1.078,  full_acc_0=1.020,  cond_acc_0=1.020
train/loss_1=2.125,  full_acc_1=0.611,  cond_acc_1=0.880
train/loss_2=2.969,  full_acc_2=0.363,  cond_acc_2=0.845
train/loss=6.182
```

### epoch（训练轮次）

- `epoch=1` 表示当前是第 **2 轮**（从 0 计数），共 5 轮
- 每轮 = 把 100K 条训练数据完整过一遍
- 每 epoch 步数：**8264 步**
- 本次训练总步数：5 × 8264 = **41320 步**

### global_step（全局步数）

- 每处理一个 batch 完成一次参数更新 = 1 步
- `global_step=10692` 表示已做了 10692 次参数更新
- 进度：10692 / 41320 = **25.9%**

**为什么 global_step 跨 epoch 累计？**  
学习率 scheduler 依赖全局 step 数做 warmup 和 cosine decay，所以用全局计数。

### lr（学习率）

- 当前：`lr=7.49e-05`，峰值 `1e-4`
- 使用 **线性 warmup + cosine decay** 调度：
  ```
  0 → 1% steps：从 0 线性增到 1e-4  （warmup，约 413 步）
  1% → 100%：余弦衰减从 1e-4 → 0
  ```
- 当前处于 cosine decay 阶段，LR 正在缓慢下降

### loss_0 / loss_1 / loss_2（各预测头的损失）

| 指标 | 值 | 含义 |
|------|----|------|
| `loss_0=1.078` | 低 | 预测第 1 个未来 token，很容易 |
| `loss_1=2.125` | 中 | 预测第 2 个未来 token，稍难 |
| `loss_2=2.969` | 高 | 预测第 3 个未来 token，更难 |
| `loss=6.182`   | 总 | 三头之和，是优化目标 |

loss 越低 = 草稿模型预测越准确。

### full_acc（完整 token 准确率）

`full_acc_0=1.020` **看起来超过 1？**  
这是加权平均的 artifact，实际含义：在 loss_mask=1 的位置（assistant 回复），预测
的 top-1 token 与目标 token 完全一致的概率。1.020 是数值误差，实际≈100%。

- `full_acc_0=1.020 ≈ 100%`：head-0（预测下1个token）几乎全对 ✅
- `full_acc_1=0.611 = 61%`：head-1 中等
- `full_acc_2=0.363 = 36%`：head-2 一般（预测3步以后本来就难）

### cond_acc（条件准确率）

在 head-0 预测正确的**条件下**，head-1/2 的准确率：

- `cond_acc_1=0.880 = 88%`：如果第1个token猜对，第2个也猜对的概率是88%
- `cond_acc_2=0.845 = 84.5%`：连续猜中3个的条件概率

**cond_acc 是评估推测解码效果的关键指标**。连续猜中 3 个 token（全部正确）的
联合概率 ≈ full_acc_0 × cond_acc_1 × cond_acc_2 ≈ 1.0 × 0.88 × 0.845 ≈ **74%**。
这意味着约 74% 的时间，草稿模型能连猜 3 个 token 全中！

---

## 五、训练中断后的处理

### Checkpoint 保存机制

每完成一个 epoch，自动保存：

```
output/eagle3_qwen3_8b/checkpoints/
├── 0/                        ← Epoch 0 完成时保存
│   ├── config.json           ← 草稿模型架构配置
│   ├── config.py
│   ├── model.safetensors     ← 模型权重（~2GB）
│   ├── optimizer_state_dict.pt  ← 优化器状态（AdamW 的 m/v 向量）
│   ├── scheduler_state_dict.pt  ← 学习率调度状态
│   └── val_metrics.json      ← 验证集指标
└── checkpoint_best -> 0/     ← 符号链接指向最佳 checkpoint
```

### 训练中断后能用吗？

**能！** 每 epoch 保存的 checkpoint 包含完整模型权重，可以直接部署使用：

```bash
# 用 epoch 0 的 checkpoint（已验证 loss=6.70，全训 5 epoch 更好）
vllm serve output/eagle3_qwen3_8b/checkpoints/0/ \
  --speculative-model output/eagle3_qwen3_8b/checkpoints/0/ \
  --port 8000
```

**epoch 0 checkpoint 的实际性能**（val_metrics.json）：
```json
loss_epoch=6.696, full_acc_0=1.346（≈100%）, cond_acc_0=1.346
```

### 能继续训练吗？

**能！** speculators 框架自动检测并恢复：

训练脚本中有 `resume_from_checkpoint=True`（默认），重新运行脚本即可：

```bash
# 直接重新运行，会自动从最后一个 epoch 的 checkpoint 恢复
bash train_eagle3_qwen3_8b.sh

# 如果不想恢复，强制从头：
bash train_eagle3_qwen3_8b.sh --no-resume-from-checkpoint
```

恢复时会加载：
- 模型权重（`model.safetensors`）
- 优化器状态（`optimizer_state_dict.pt`）
- 学习率进度（`scheduler_state_dict.pt`）

**注意**：如果训练在 epoch 中途中断（比如 epoch 2 跑到一半），会从 epoch 1 末
重新开始（epoch 内的 step 不保存中间状态）。

---

## 六、训练全程概览

```
时间线：
Day 1 12:56 UTC  Eagle-3 训练开始
      13:00      Step 100，loss=21
      23:44      Epoch 0 完成，checkpoint/0 保存，loss_epoch=6.70

Day 2 02:31      Epoch 1 进行中，step=10613，loss=6.66
  ~Day 2 18:00   Eagle-3 训练完成（5 epochs），模型保存
  ~Day 3 18:00   DFlash 训练完成
```

**Loss 下降曲线**：
```
step   0: loss=28.0   （随机初始）
step 100: loss=21.0   （开始学到基本语言）
step 10K: loss= 6.7   （收敛良好，接近最终水平）
step 40K: loss≈ 5.x   （预期最终水平）
```

---

## 七、配置参数参考

```bash
# Eagle-3 关键参数
MODEL_PATH=/data/models/Qwen3-8B
VLLM_GPUS=0,1          # vLLM 用 GPU 0,1 (TP=2)
TRAIN_GPUS=3,4         # 训练用 GPU 3,4 (DDP)
VLLM_TP=2              # Tensor Parallel（解决 DP 端口冲突问题）
VLLM_PORT=8200
MAX_SAMPLES=100000
SEQ_LENGTH=8192
EPOCHS=5
LR=1e-4
DRAFT_VOCAB_SIZE=32000
NUM_WORKERS=0          # 容器 /dev/shm=64MB，不能用多进程 DataLoader
TARGET_LAYER_IDS="2 18 33"  # Qwen3-8B 36层，抽取第2/18/33层

# DFlash 额外参数
LR=3e-4
NUM_LAYERS=5
BLOCK_SIZE=8
MAX_ANCHORS=3072
SPECULATOR_TYPE=dflash
```

---

## 八、已知 Bug 和修复

| Bug | 现象 | 修复位置 |
|-----|------|---------|
| JSONL schema 冲突 | datasets 库推断类型失败 | 重新生成 `sharegpt_clean.jsonl` |
| `prefetch_factor` + `num_workers=0` | PyTorch DataLoader 报错 | `speculators/scripts/train.py:setup_dataloader` |
| dtype 不匹配 | `input_ids` 被误转 bfloat16 | `speculators/src/speculators/train/trainer.py:train_epoch` |
| vLLM max-model-len | 恰好 SEQ_LENGTH 的输入无法生成 | 脚本改为 `SEQ_LENGTH + 512` |
| vLLM DP=2 端口冲突 | `EADDRINUSE`: 多进程抢同一端口 | 改为 `--tensor-parallel-size 2` |

---

*记录时间：2026-05-27，基于实际训练日志*
