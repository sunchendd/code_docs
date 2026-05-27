# DFlash 深度解析：块扩散加速推测解码

> **论文**：DFlash: Block Diffusion for Flash Speculative Decoding  
> **作者**：Jian Chen, Yesheng Liang, Zhijian Liu（UC San Diego）  
> **发表**：ICML 2026 Poster | arXiv: 2602.06036  
> **代码**：[github.com/z-lab/dflash](https://github.com/z-lab/dflash)  
> **模型**：[huggingface.co/z-lab](https://huggingface.co/collections/z-lab/dflash)

---

## 目录

1. [概述](#1-概述)
2. [背景：为什么 LLM 推理慢？](#2-背景为什么-llm-推理慢)
3. [投机解码（Speculative Decoding）基础](#3-投机解码speculative-decoding基础)
4. [现有投机解码的瓶颈](#4-现有投机解码的瓶颈)
5. [DFlash 的三大核心创新](#5-dflash-的三大核心创新)
   - 5.1 [Block Diffusion 并行 Draft](#51-block-diffusion-并行-draft)
   - 5.2 [KV Injection 特征注入](#52-kv-injection-特征注入)
   - 5.3 [Flex Attention 掩码](#53-flex-attention-掩码)
6. [Hidden State vs KV Cache vs Context Feature](#6-hidden-state-vs-kv-cache-vs-context-feature)
7. [5 层 Transformer 中的信息流动](#7-5-层-transformer-中的信息流动)
8. [训练过程与推理过程的一致性](#8-训练过程与推理过程的一致性)
9. [完整推理流程图](#9-完整推理流程图)
10. [实验结果与性能对比](#10-实验结果与性能对比)
11. [支持的模型列表](#11-支持的模型列表)
12. [快速上手](#12-快速上手)
13. [总结](#13-总结)

---

## 1. 概述

DFlash 是一种基于**块扩散模型（Block Diffusion）** 的推测解码（Speculative Decoding）框架，用于加速大语言模型（LLM）的推理。

**核心思想**：用一个轻量级的块扩散模型替代传统的自回归 draft 模型，实现**一次前向传播并行生成整个 token 块**（通常 16 个 token），再由大模型一次性验证，从而实现**无损加速**。

**核心成果**：
- 在 Qwen3-8B 上实现 **6.1× 无损加速**
- 比 SOTA 方法 EAGLE-3 快 **2.5×**
- 输出与大模型单独生成**完全一致**（无损）

---

## 2. 背景：为什么 LLM 推理慢？

### 2.1 自回归生成的串行瓶颈

普通 LLM 生成文本是**逐 token** 的——每生成一个 token，都需要完整的前向传播：

```
用户问：1+1=?

LLM 生成过程：

第1步：读完问题 → 所有层计算 → 输出 "答"     ⏱ 50ms
第2步：输入"答" → 所有层计算 → 输出 "案"     ⏱ 50ms
第3步：输入"案" → 所有层计算 → 输出 "是"     ⏱ 50ms
第4步：输入"是" → 所有层计算 → 输出 "2"      ⏱ 50ms
                                                ──────
                                          总计   200ms
```

### 2.2 内存带宽瓶颈

每生成一个 token，都需要把**整个模型的几十亿参数从显存搬到计算核心**走一遍。GPU 的算力很闲，大部分时间在**等数据搬运**。

> **类比**：你有一条 8 车道的高速公路（GPU 算力），但入口只有 1 个收费站（内存带宽）。每次只放 1 辆车进来（1 个 token），8 车道上只跑了 1 辆车，其他 7 条道空着。

---

## 3. 投机解码（Speculative Decoding）基础

### 3.1 核心思想

> **找一个又小又快的"枪手"（draft 模型）先快速猜一串答案，再让大模型一次性验证。猜对的全收，猜错的从错误处重来。**

### 3.2 为什么验证可以并行？

生成 5 个 token 需要 5 次串行前向传播，但**验证 5 个 token 只需要 1 次前向传播**：

```
生成（串行）：          验证（并行）：
t₁ → t₂ → t₃ → t₄ → t₅    [t₁, t₂, t₃, t₄, t₅] → 一次性全部检查
5 次前向传播                  1 次前向传播
```

> **类比**：写作文要一个字一个字写，但老师扫一眼就知道对不对。

### 3.3 验证与接受机制

```
draft 模型猜了 5 个 token：[t₁, t₂, t₃, t₄, t₅]

大模型逐位验证：
  t₁ 正确？ ✅ 接受
  t₂ 正确？ ✅ 接受
  t₃ 正确？ ✅ 接受
  t₄ 正确？ ❌ 拒绝！→ 后面全部丢弃
  
结果：接受 [t₁, t₂, t₃]（3 个）
      + 大模型给出正确的 t₄（bonus token）
      从 t₄ 开始下一轮
```

**无论猜对猜错，最终输出和大模型直接生成完全一致。**

---

## 4. 现有投机解码的瓶颈

EAGLE-3 等现有 SOTA 方法的 draft 模型仍然是**自回归的**：

```
EAGLE-3 的 draft 过程：

猜 t₁ → 猜 t₂ → 猜 t₃ → ... → 猜 t₁₆     ⏱ 16 × t_step
                                                （串行，和 token 数成正比）

总时间 = 16 × t_step + t_verify
```

**核心矛盾**：

| 矛盾点 | 说明 |
|--------|------|
| 想猜更多 token | draft 时间线性增长 → 收益递减 |
| 想 draft 更快 | 模型必须做得极浅（EAGLE-3 只用 1 层）→ 猜不准 |
| 实际加速上限 | **2-3×** |

---

## 5. DFlash 的三大核心创新

### 5.1 Block Diffusion 并行 Draft

#### 核心思想

用**填空题**代替**听写**：

```
自回归（听写 / EAGLE-3）：

  一个一个来：
  第1步：[？] → 猜出 "认为"
  第2步：[认为, ？] → 猜出 "是"
  第3步：[认为, 是, ？] → 猜出 "毕达"
  第4步：[认为, 是, 毕达, ？] → 猜出 "哥拉"
  
  共 4 步，串行


Block Diffusion（填空 / DFlash）：

  一口气来：
  一步搞定：[？, ？, ？, ？] → 同时猜出 ["认为", "是", "毕达", "哥拉"]
  
  共 1 步，并行
```

#### 为什么并行时间"几乎恒定"?

因为 GPU 本质上是**矩阵运算**——同时处理 4 个 token 和 16 个 token，对 GPU 来说差别很小（都是一次矩阵乘法，只是矩阵宽了一点）：

```
自回归 draft 的时间：

  猜 4 个 token  →  4 次前向传播 → 4t
  猜 16 个 token → 16 次前向传播 → 16t    ← 线性增长！

Block Diffusion draft 的时间：

  猜 4 个 token  → 1 次前向传播 → ~1.0t
  猜 16 个 token → 1 次前向传播 → ~1.2t   ← 几乎不变！
```

#### "？"之间怎么猜得准？

4 个 mask token 不是独立猜的——它们在 Transformer 的注意力层中**互相参考**：

```
Block 内的注意力：

  位置2（？）可以看到：
    ✅ 大模型的 Context Feature
    ✅ 锚点 token（"一般"）
    ✅ 位置3、4、5 的 mask token

  虽然一开始大家都是"？"，但：
    ① 位置编码不同 → 模型知道每个"？"的位置
    ② 通过多层 Transformer 的反复交互，信息在位置之间流动
    ③ 最终每个位置分化出不同的预测

  类比：交响乐团调音
    指挥给一个起始音 → 每个乐手根据自己的声部位置
    → 多轮排练（多层 Transformer）→ 最终协调出完美和弦
```

---

### 5.2 KV Injection 特征注入

#### 核心思想

> **让 draft 模型"偷看"大模型的内部思考过程，借用大模型的推理能力。**

```
没有 KV Injection：

  教授："题目给你，自己做。"
  助教：只能靠自己的 5 层浅薄理解 → 猜得很不准


有 KV Injection：

  教授："题目给你，另外这是我在 5 个层的思考笔记。"
  助教：拿着教授的笔记，每做一道题都参考 → 猜得又快又准
```

#### 具体实现

```
大模型（36 层）处理用户输入

    第 2 层 → hidden states₂  ← 抽取 ✅
    第 10 层 → hidden states₁₀ ← 抽取 ✅
    第 18 层 → hidden states₁₈ ← 抽取 ✅
    第 26 层 → hidden states₂₆ ← 抽取 ✅
    第 34 层 → hidden states₃₄ ← 抽取 ✅

    5 层的 hidden states → 通过投影矩阵（fc）融合 → Context Feature
    
    Context Feature 注入 draft 模型每一层的 KV Cache
```

#### 为什么注入每一层而不是只注入第一层？

```
只注入第 1 层：
  大模型信息 → 进入第1层 → 第2层稀释 → 第3层更淡 → 到第5层几乎没了
  
  就像在河上游倒一杯墨水，到下游看不到了

注入每一层：
  第1层：大模型信息 → 直接可用 ✅
  第2层：大模型信息 → 直接可用 ✅（重新注入的，不是传过来的）
  第3层：大模型信息 → 直接可用 ✅
  ...
  
  每一层都有"新鲜的墨水"，信息永远不会被稀释
```

---

### 5.3 Flex Attention 掩码

#### 掩码的作用：保证训练正确性

> **掩码不是"魔法"——它只是一套交通规则，保证训练时不作弊，推理时能正常工作。**

#### 三条规则

```
规则 1：每个 mask token ✅ 可以看到所有大模型 Context Feature
        → KV Injection，借用大模型的智慧

规则 2：每个 mask token ✅ 可以看到自己 block 内的锚点和其他 mask token
        → 同 block 内互相参考，保持连贯

规则 3：每个 mask token ❌ 不能看到后续 block 的内容
        → 防止"偷看未来"，训练-推理一致
```

#### 阶梯状掩码示意

```
                 大模型特征     Block 1            Block 2
                 (蓝色)       锚 空 空 空        锚 空 空 空
                 ──────────   ───────────        ───────────
Block 1 空位1：  ✅✅✅✅    ✅ ✅ ✅ ✅        ❌ ❌ ❌ ❌
Block 1 空位2：  ✅✅✅✅    ✅ ✅ ✅ ✅        ❌ ❌ ❌ ❌
Block 1 空位3：  ✅✅✅✅    ✅ ✅ ✅ ✅        ❌ ❌ ❌ ❌

Block 2 空位1：  ✅✅✅✅    ✅ ✅ ✅ ✅        ✅ ✅ ✅ ✅
Block 2 空位2：  ✅✅✅✅    ✅ ✅ ✅ ✅        ✅ ✅ ✅ ✅
Block 2 空位3：  ✅✅✅✅    ✅ ✅ ✅ ✅        ✅ ✅ ✅ ✅
```

#### 为什么没有掩码会崩溃？

```
训练时没有掩码（可以偷看后续 block）：
  MASK₄ 偷看到后续 block → 轻松猜对 → 损失很低 → 模型觉得自己"学会了"
  但实际学到的是"偷看答案"，不是真正的预测能力

推理时没有后续 block 可偷看 → 模型完全不会做 → 效果崩溃

有掩码：
  训练时看不到后续 block → 被迫学习真正的预测能力
  推理时同样看不到后续内容 → 和训练行为一致 → 效果正常
```

---

## 6. Hidden State vs KV Cache vs Context Feature

这是理解 DFlash 相对于 EAGLE-3 优势的关键。

### 6.1 三者对比

| | Hidden State（EAGLE-3） | KV Cache（原始） | Context Feature（DFlash） |
|---|---|---|---|
| **存的是什么** | 所有 token **加权混合**后的 1 个向量 | 每个 token **独立**的 K 和 V | 大模型多层 hidden states **每 token 独立**融合后的特征 |
| **类比** | 会议纪要（有取舍） | 完整会议录音（每人一条轨道） | 会议纪要（但每个人独立写一份） |
| **信息丢失？** | ✅ 注意力低的 token 信息被严重稀释 | ❌ 完整保留 | ❌ 每个 token 独立保留 |
| **大小** | 1 个向量（~4096 维） | 36层 × 2 × n_token × 4096（巨大） | n_token × d_draft（适中） |
| **适合传递？** | ✅ 很小 | ❌ 太大 | ✅ 大小合适 |

### 6.2 Hidden State 为什么丢信息？

```
原始序列：["勾", "股", "定", "理"]

Hidden State h₃（"理"位置的输出）：

  h₃ = 0.02 × V_勾 + 0.03 × V_股 + 0.15 × V_定 + 0.80 × V_理
  
  "勾"的信息在 h₃ 里只剩 2%！几乎找不到了
  
  → 如果后续预测需要"勾"的信息，从 h₃ 里提取不出来
  → 这就是 EAGLE-3 "长距衰减"问题的根源
```

### 6.3 Context Feature 为什么不丢？

```
DFlash 的 Context Feature：

  "勾" → f₁（独立向量）  ← 完整保留
  "股" → f₂（独立向量）  ← 完整保留
  "定" → f₃（独立向量）  ← 完整保留
  "理" → f₄（独立向量）  ← 完整保留
  
  想要"勾"的信息？直接用 Query 去 attend f₁ 就行
  → 信息完整，不存在被压缩丢失的问题
```

### 6.4 Context Feature 比 Hidden State 大多少？

```
EAGLE-3 传递的信息：  1 × 4096 维 ≈ 4,096 个数字
DFlash 传递的信息：   20 × 1024 维 ≈ 20,480 个数字（假设 20 个 token）

→ 约 5 倍，换来的是 2.5× 的额外加速，完全值得
```

---

## 7. 5 层 Transformer 中的信息流动

以预测"一般认为是毕达哥拉"为例，锚点为"一般"，block_size=4。

### 第 0 步：Embedding

```
"一般"  → [0.3, -0.1, 0.8, ...]    ← 真实词
MASK₂   → [0, 0, ...] + PE(2)       ← mask + 位置编码(2)
MASK₃   → [0, 0, ...] + PE(3)       ← mask + 位置编码(3)
MASK₄   → [0, 0, ...] + PE(4)       ← mask + 位置编码(4)
MASK₅   → [0, 0, ...] + PE(5)       ← mask + 位置编码(5)

此刻 4 个 MASK 的唯一区别：位置编码不同
```

### 第 1 层：初步感知

```
每个 MASK 用自己的 Q 去查询大模型的 Context Feature + 锚点 + 其他 MASK
位置编码告诉模型"这是第几个位置的空"

MASK₂（位置编码=2，紧跟锚点）：
  → 更关注大模型 Context Feature 中关于"下一个词"的信息
  → 初步倾向："认为"类动词

MASK₅（位置编码=5，离锚点远）：
  → 更关注大模型 Context Feature 中关于"后续内容"的信息
  → 初步倾向：某种名词延续

第 1 层结束：每个 MASK 有了模糊的方向（确信度 ~20-30%）
```

### 第 2 层：互相参考，开始分化

```
MASK₂ 看到 MASK₃ 倾向 "是"
  → "后面接'是'很合理，我更确信自己是'认为'了"

MASK₄ 看到 MASK₂="认为"类、MASK₃="是"类
  → 前面是 "一般认为是___" → 我应该是人名开头 → "毕达"概率上升

MASK₅ 看到 MASK₄ 越来越像人名前半部分
  → 我紧跟位置4 → 应该是人名后半 → "哥拉"概率上升

第 2 层结束：确信度提升到 ~40-60%
```

### 第 3 层：结构确定

```
"一般 + 认为 + 是 + 人名" 框架清晰
结合 Context Feature 中"毕达哥拉斯"的信息 → 锁定具体词

第 3 层结束：确信度 ~75-85%
```

### 第 4-5 层：精细化

```
概率分布变得更尖锐（更确定）
处理细节（比如"毕达"而不是"毕得"）

第 5 层输出：
  MASK₂ → "认为" （确信度 92%）  ✅
  MASK₃ → "是"   （确信度 88%）  ✅
  MASK₄ → "毕达" （确信度 85%）  ✅
  MASK₅ → "哥拉" （确信度 80%）  ✅
```

### 信息流动总结

```
信息流动的方向：

  大模型 Context Feature ──→ 每一层的每个 MASK（垂直注入，始终新鲜）
  锚点 "一般" ──→ 所有 MASK（提供起始上下文）
  MASK ←──→ MASK（水平流动，互相协调）

  三股信息流在每一层汇合，逐层细化
```

---

## 8. 训练过程与推理过程的一致性

### 8.1 训练时做的事

```
训练数据：一条完整的文本序列
  "一般认为是毕达哥拉斯提出了勾股定理"

步骤：
  ① 随机选一个锚点，比如 "一般"
  ② 锚点后面的 4 个 token 遮住（mask）
     ["一般", MASK, MASK, MASK, MASK]
  ③ 大模型提取多层 hidden states → Context Feature
  ④ Context Feature 注入 draft 模型 KV
  ⑤ draft 模型一次前向传播，预测 4 个 MASK
     预测：["认为", "是", "毕达", "哥拉"]
     正确：["认为", "是", "毕达", "哥拉"]
  ⑥ 计算损失 → 反向传播 → 更新 draft 模型权重
  ⑦ 重复几万次
```

### 8.2 推理时做的事

```
用户问："勾股定理是谁提出的？"

步骤：
  ① 大模型 Prefill，生成第一个 token "一般"，同时提取 Context Feature
  ② Context Feature 注入 draft 模型 KV
  ③ draft 模型收到 ["一般", MASK, MASK, MASK, MASK]
  ④ draft 模型一次前向传播，预测 4 个 MASK
     预测：["认为", "是", "毕达", "哥拉"]
  ⑤ 大模型验证 → 全对 → 接受
  ⑥ 从最后一个接受的 token 继续，重复 ②-⑤
```

### 8.3 训练-推理一致性对比

```
              训练                          推理
            ─────────                     ─────────
输入格式   ["一般", M, M, M, M]          ["一般", M, M, M, M]
              ↑ 完全一样 ↑

KV 注入    大模型的 Context Feature      大模型的 Context Feature
              ↑ 完全一样 ↑

掩码规则   不能看后续 block              不存在后续 block
              ↑ 效果一样 ↑

draft 输出  预测 4 个 token              预测 4 个 token
              ↑ 完全一样 ↑

之后做什么  和正确答案比较 → 更新权重    大模型验证 → 接受/拒绝
              ↑ 唯一区别 ↑
```

### 8.4 训练中的三个精妙设计

#### 随机 Anchor 采样

不均匀切块，而是随机选择锚点，起**数据增强**作用，且与推理行为一致。

#### 指数衰减损失

Block 中越靠前的 token 越重要（错了后面全废），给前面更高的损失权重：

```
Block: [t₁, t₂, t₃, t₄]

损失权重：
  t₁: ████████████  1.0    ← 最重要
  t₂: ██████████    0.82
  t₃: ████████      0.67
  t₄: ██████        0.55
```

#### Flex Attention

实现阶梯状注意力掩码，保证训练-推理一致性。

---

## 9. 完整推理流程图

```
═══════════════════════════════════════════════════════════════
                    DFlash 推理完整流程
═══════════════════════════════════════════════════════════════

┌────────────────── 第 0 轮：Prefill ──────────────────────┐
│                                                           │
│  用户输入："勾股定理是谁提出的？"                         │
│                                                           │
│  大模型（Target）做 Prefill：                             │
│  ① 处理整个输入序列                                      │
│  ② 生成第一个 token："一般"                              │
│  ③ 同时提取多层 hidden states → 融合为 Context Feature    │
│                                                           │
│  输出：token "一般" + Context Feature                     │
│                                                           │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────── 第 1 轮：Draft ─────────────────────────┐
│                                                           │
│  DFlash Draft 模型收到：                                  │
│    • Context Feature → 注入每层 KV                        │
│    • Clean Token："一般"（锚点）                          │
│    • Mask Tokens：[?, ?, ?, ?]                            │
│                                                           │
│  Draft 模型一次前向传播（并行）：                         │
│    位置2: ? → 预测 "认为"  ✅                             │
│    位置3: ? → 预测 "是"    ✅                             │
│    位置4: ? → 预测 "毕达"  ✅                             │
│    位置5: ? → 预测 "哥拉"  ✅                             │
│                                                           │
│  ⏱ 耗时：~t_parallel（和猜几个 token 几乎无关）          │
│                                                           │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────── 第 1 轮：Verify ────────────────────────┐
│                                                           │
│  大模型收到：["一般", "认为", "是", "毕达", "哥拉"]      │
│                                                           │
│  一次前向传播，并行验证所有位置：                         │
│    "一般" → 应该接 "认为"？ ✅ 接受                      │
│    "认为" → 应该接 "是"？   ✅ 接受                      │
│    "是"   → 应该接 "毕达"？ ✅ 接受                      │
│    "毕达" → 应该接 "哥拉"？ ✅ 接受                      │
│    "哥拉" → Bonus Token：  "斯"                          │
│                                                           │
│  本轮净收益：5 个 token                                   │
│  同时提取新的 Context Feature → 给下一轮用                │
│                                                           │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
                    第 2 轮：从 "斯" 继续...
                              │
                              ▼
                    循环直到生成 EOS
```

### 如果猜错了？

```
假设 Draft 猜了：["发现", "了", "无理数", "的"]

大模型验证：
  "发现"   → "了"？     ✅
  "了"     → "无理数"？ ✅
  "无理数" → "的"？     ❌ 大模型认为应该接 "和"
  
结果：接受 ["发现", "了", "无理数"] + bonus "和"
      从 "和" 开始下一轮
      
最终输出和大模型单独生成完全一致 → 无损
```

---

## 10. 实验结果与性能对比

### 加速比对比

| 方法 | Qwen3-8B 加速比 | Draft 架构 | Draft 方式 |
|------|-----------------|-----------|-----------|
| 自回归解码 | 1.0× (基准) | 无 | 无 |
| EAGLE-3 | ~2.4× | 1 层 Transformer | 自回归（串行） |
| **DFlash** | **~6.1×** | **5 层 Transformer** | **块扩散（并行）** |

### 为什么 DFlash 比 EAGLE-3 快 2.5 倍？

```
每轮投机解码，生成一个 token 的平均延迟：

          T_draft + T_verify
  L  =  ─────────────────────
                 τ

  T_draft  = draft 模型生成候选的时间
  T_verify = 大模型验证的时间
  τ        = 本轮实际接受的 token 数
```

| | EAGLE-3 | DFlash |
|---|---|---|
| T_draft | γ × t_step（线性增长） | ≈ t_parallel（几乎恒定） |
| Draft 架构深度 | 1 层（必须极浅才够快） | 5 层（更深更准） |
| 可用 block size | 小（否则太慢） | 大（γ=16 甚至更大） |
| 接受率 τ | 较低（模型太浅） | 较高（更深 + KV Injection） |

**DFlash 打破了 EAGLE-3 的"快 vs 准"矛盾**——并行生成让模型做深也不变慢，KV Injection 让小模型也能猜准。

---

## 11. 支持的模型列表

| 目标模型 | DFlash Draft 模型 |
|---------|-------------------|
| gemma-4-31B-it | z-lab/gemma-4-31B-it-DFlash |
| gemma-4-26B-A4B-it | z-lab/gemma-4-26B-A4B-it-DFlash |
| Qwen3.6-27B | z-lab/Qwen3.6-27B-DFlash |
| Qwen3.5-4B | z-lab/Qwen3.5-4B-DFlash |
| Qwen3.5-9B | z-lab/Qwen3.5-9B-DFlash |
| Qwen3.5-27B | z-lab/Qwen3.5-27B-DFlash |
| Qwen3.5-122B-A10B | z-lab/Qwen3.5-122B-A10B-DFlash |
| Qwen3-Coder-30B-A3B | z-lab/Qwen3-Coder-30B-A3B-DFlash |
| Kimi-K2.5 | z-lab/Kimi-K2.5-DFlash |
| MiniMax-M2.5 | z-lab/MiniMax-M2.5-DFlash |
| Llama-3.1-8B-Instruct | z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat |

---

## 12. 快速上手

### 安装

```bash
# Transformers 后端
pip install transformers==4.57.3 torch==2.9.1 accelerate

# 或 SGLang 后端（生产部署）
pip install "git+https://github.com/sgl-project/sglang.git#subdirectory=python"
```

### Transformers 用法

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 1. 加载 DFlash Draft 模型
model = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16",
    trust_remote_code=True, dtype="auto", device_map="cuda:0"
).eval()

# 2. 加载目标模型
target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype="auto", device_map="cuda:0"
).eval()

# 3. 准备输入
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
messages = [{"role": "user", "content": "勾股定理是谁提出的？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. 推测解码生成
output_ids = model.spec_generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=2048,
    temperature=0.0,
    target=target,
    stop_token_ids=[tokenizer.eos_token_id]
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### SGLang 用法（生产部署）

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \
    --dtype bfloat16 \
    --attention-backend fa3
```

---

## 13. 总结

### 一张图看全 DFlash

```
╔══════════════════════════════════════════════════════════════╗
║                    DFlash 加速原理全景                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ❶ 问题                                                     ║
║     LLM 逐 token 生成太慢（内存带宽瓶颈）                   ║
║                                                              ║
║  ❷ 投机解码思路                                              ║
║     小模型快速猜 → 大模型并行验 → 猜对的全收                ║
║                                                              ║
║  ❸ 现有瓶颈（EAGLE-3）                                      ║
║     小模型也是自回归的，猜 γ 个要 γ 步 → 加速上限 2-3×      ║
║                                                              ║
║  ❹ DFlash 创新 1：Block Diffusion 并行 Draft                ║
║     一次前向传播猜出整个 block → 时间 ≈ 常数                ║
║     可以用更深的模型（5层 vs 1层）→ 猜得更准               ║
║                                                              ║
║  ❺ DFlash 创新 2：KV Injection                              ║
║     大模型多层 hidden states → 融合为 Context Feature        ║
║     注入 draft 每一层 KV → 每个 token 信息完整保留           ║
║     小模型借用大模型的"推理能力" → 接受率大幅提升           ║
║                                                              ║
║  ❻ DFlash 创新 3：Flex Attention 掩码                       ║
║     同 block 内互相可见 + 后续 block 不可见                  ║
║     保证训练-推理行为一致 → 学到的能力能正常迁移             ║
║                                                              ║
║  ❼ 结果                                                     ║
║     6×+ 无损加速 → 比 EAGLE-3 快 2.5×                       ║
║     输出与大模型单独生成完全一致                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 三个核心机制的关系

```
┌─────────────────────┐
│  KV Injection       │  → 解决"猜得准"的问题
│  (大模型给小模型     │     小模型借用大模型的深度理解
│   递思考笔记)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Block Diffusion    │  → 解决"猜得快"的问题
│  (一口气填完所有空)  │     一次前向传播猜出整个 block
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Flex Attention 掩码 │  → 解决"训练正确性"的问题
│  (考试不能作弊)      │     保证训练和推理行为一致
└─────────────────────┘

三者缺一不可：
  没有 KV Injection → 猜不准，白忙活
  没有 Block Diffusion → 还是串行猜，和 EAGLE-3 没区别
  没有掩码 → 训练时作弊，推理时崩溃
```

---

> **文档整理自与用户的多轮深度对话，涵盖 DFlash 的原理、机制、训练、推理等所有核心内容。**
