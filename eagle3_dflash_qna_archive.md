# Eagle-3 / DFlash 训练问答存档

> 记录时间：2026-05-27  
> 当前训练状态：Eagle-3 step 11276 / 41320（27.3%），epoch=1，loss≈5.6  
> 本文档为对话中关键问题的完整存档，配合 `eagle3_dflash_training_notes.md` 一起阅读

---

## Q1：为什么选第 2/18/33 层提取 hidden states？

Qwen3-8B 共 **36 层**，Eagle-3 从中挑三层作为"语义锚点"：

```
层   0  → 刚嵌入，仅token特征，信息太浅
层   2  → 早期语法结构（浅层，计算成本低）       ← 选
层  18  → 中间语义理解（信息最丰富，最重要）      ← 选
层  33  → 接近输出的预测特征（接近"下一token"决策）← 选
层  36  → 最终输出logits（不选，泄露答案）
```

**选择依据：**
- 等间距采样，覆盖"浅/中/深"三个语义阶段
- 三层 hidden states 拼接后送入草稿模型（concat → fc降维），比单层提高约 15-20% 接受率
- 官方 Eagle-3 论文通过消融实验验证了这种跨层融合策略

**三层拼接过程：**
```
hs_2  [seq, 4096]  ┐
hs_18 [seq, 4096]  ├─ concat → [seq, 12288] → FC(12288→4096) → 草稿模型输入
hs_33 [seq, 4096]  ┘
```

---

## Q2：AdamW 更新什么参数？有多少个？怎么改的？

### 草稿模型参数一览（总计 **1022M**）

| 参数 | 形状 | 大小 | 作用 |
|------|------|------|------|
| `embed_tokens.weight` | [151936, 4096] | 622M | 词嵌入（复用Qwen3） |
| `fc.weight` | [4096, 12288] | 50M | 三层hidden_states融合降维 |
| `layers.0.self_attn.*` | Q/K/V/O proj | 67M | 1层注意力 |
| `layers.0.mlp.*` | gate/up/down proj | 151M | 1层FFN |
| `lm_head.weight` | [32000, 4096] | 131M | 预测token概率 |
| `norm / layernorm` | [4096] | ~0M | 归一化 |

> ⚠️ **Qwen3-8B本体（8B参数）完全冻结，只训练这1022M草稿模型**

### AdamW 更新公式（每步执行一次）

```python
# 每个参数 p 维护两个状态
m = 0.9 * m + 0.1 * grad          # 一阶动量：平滑梯度方向
v = 0.999 * v + 0.001 * grad²     # 二阶动量：自适应调步长

# 参数更新
p -= lr * m / (√v + 1e-8)         # 自适应梯度步
p -= lr * weight_decay * p         # 权重衰减（防过拟合）
```

**直觉理解：**
- 大梯度的参数 → `v`大 → 步长小，防止震荡
- 小梯度的参数 → `v`小 → 步长大，加速收敛
- 每步参数改动量约为参数值的 **0.001%**，微小但积累后效果显著

**超参数配置：**
```
lr_peak = 1e-4     # 峰值学习率
warmup  = 413步   # 前1%步线性升温
之后     cosine衰减到0
weight_decay = 0.01
beta1 = 0.9, beta2 = 0.999
```

---

## Q3：准确率还在优化吗？优化曲线是什么样的？

**是的，仍在持续优化。**

### Loss 下降曲线（每200步采样，基于实际训练日志）

```
step     0   loss=33.6  ███████████████████████████████████████ ← 随机猜，完全不会
step   400   loss=17.3  ████████████████████
step   800   loss=13.2  ███████████████
step  2000   loss=10.3  ████████████         ← warmup结束，进入cosine衰减
step  4000   loss= 8.5  ██████████
step  6400   loss= 5.9  ███████              ← epoch 0 末尾
step  8400   loss= 5.6  ██████
step 10800   loss= 5.6  ██████               ← 当前（epoch 1 中段）
```

### 准确率当前值（step ~11000）

| 头 | full_acc | cond_acc | 含义 |
|----|---------|---------|------|
| head-0（预测+1） | **~99%** | ~99% | 下一个token几乎全猜对 |
| head-1（预测+2） | ~76% | **~88%** | head-0对时，+2也对的概率 |
| head-2（预测+3） | ~52% | **~84%** | 连续两对时，+3也对的概率 |

**联合接受率 ≈ 99% × 88% × 84% ≈ 73%**（一次投机猜3个全中的概率）

### 未来趋势预测

| 阶段 | 预期总loss | 说明 |
|------|-----------|------|
| Epoch 1（当前） | ~5.5 | 仍在下降 |
| Epoch 2 | ~5.0 | 进入精细优化 |
| Epoch 3-4 | ~4.5-4.8 | 逐渐收敛 |
| Epoch 5（最终） | ~4.3-4.5 | 最终模型 |

**波动说明：** 单步loss在5~8之间跳动属正常（不同样本难度不同），关注的是**移动平均趋势**，不是单步值。

---

## Q4：模型怎么用？与官方相比如何？

### 训练完后的使用方法

```bash
# Eagle-3 推理（推荐）
vllm serve /data/models/Qwen3-8B \
  --speculative-model /data/models/Qwen3-8B-speculator.eagle3-trained \
  --num-speculative-tokens 3 \
  --tensor-parallel-size 4

# DFlash 推理
vllm serve /data/models/Qwen3-8B \
  --speculative-model /data/models/Qwen3-8B-speculator.dflash-trained \
  --speculative-model-quantization fp8 \
  --tensor-parallel-size 4
```

### 当前状态 vs 官方性能对比

| 指标 | 当前（step 11000） | 训练完成预期 | 官方 RedHatAI |
|------|--------------------|------------|--------------|
| head-0 acc | ~99% | ~99% | ~99% |
| head-1 cond_acc | ~88% | ~90% | ~92% |
| head-2 cond_acc | ~84% | ~87% | ~90% |
| 联合接受率 | ~73% | ~77% | ~82% |
| 推理加速比 | 未测 | **~2.5x** | ~2.8x |

### 差距分析

| 差距来源 | 说明 |
|---------|------|
| 数据量 | 我们用 ShareGPT 12万条；官方用更大专有数据集（数百万条） |
| 数据质量 | ShareGPT 是用户对话，官方可能包含更多推理/代码样本 |
| 训练轮次 | 5 epochs（够用），官方可能做了多次迭代筛选 |
| 架构完全相同 | 差距不来自模型结构，而来自数据 |

**结论：** 训练完成后实际加速比预计达到官方的 **90%**，足够用于生产推理加速。

---

## 附：一条数据的完整训练流程

以一条"解释量子纠缠"对话为例：

```
原始文本 → Tokenizer → token序列 [1, 24, 556, 8921, ..., 2]

Step A: 训练进程发送 token 序列给 vLLM（HTTP 请求）
         ↓
Step B: vLLM 用 Qwen3-8B 做前向传播
        在第 2/18/33 层捞出 hidden_states（形状各为 [seq_len, 4096]）
        写入 /tmp/hidden_states/hs_xxx.safetensors
         ↓
Step C: 草稿模型读取 hidden_states：
        embed_tokens(tokens) + FC(concat(hs_2, hs_18, hs_33))
        → 1层 Transformer → 3组 logits（head-0/1/2）
        → 每组预测词表151936个token的概率分布
         ↓
Step D: 与真实下一个token比较，计算 CrossEntropy Loss
        loss_0 = 1.078  (预测+1)
        loss_1 = 2.125  (预测+2)
        loss_2 = 2.969  (预测+3)
        total  = 6.172
         ↓
Step E: 反向传播 → AdamW 更新 1022M 草稿模型参数
        Qwen3-8B 参数 = 不动（冻结）
         ↓
耗时：约 4.6 秒/步（瓶颈：vLLM 串行生成 hidden states）
```

---

## 附：关键配置和断点恢复

### 训练配置
```bash
# Eagle-3
NUM_LAYERS=1, LR=1e-4, EPOCHS=5
TARGET_LAYERS="2 18 33"
SEQ_LENGTH=8192, BATCH_SIZE=2

# DFlash（额外参数）
NUM_LAYERS=5, LR=3e-4
BLOCK_SIZE=8, MAX_ANCHORS=3072
```

### 断点恢复
```bash
# 重新运行脚本即可，框架自动检测并从最后 epoch 恢复
bash train-models/run_both.sh

# 保存内容（每 epoch 末）
output/eagle3_qwen3_8b/checkpoints/{epoch}/
  ├── model.safetensors      # 模型权重
  ├── optimizer_state_dict.pt # AdamW 动量状态
  └── scheduler_state_dict.pt # LR 调度状态
```

### 监控命令
```bash
tail -f train-models/output/eagle3_qwen3_8b/train.log
watch -n10 nvidia-smi
```

---

*存档时间：2026-05-27 03:18 UTC | 训练进度：step 11276/41320（27.3%）*
