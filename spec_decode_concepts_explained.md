# 投机解码核心概念详解

> 本文档解释 SGLang v0.5.12 中的两个核心优化概念：Adaptive Speculative Decoding 和 SWA

---

## 1. 动态调整步数 + EMA跟踪接受率

### 什么是"步数"？

在投机解码中，**步数(num_speculative_tokens)** 指一次猜测多少个token：

```
步数 = 3 时:
输入: "今天天气"
Draft模型猜测: ["很", "好", "，"]  ← 一次猜3个token
Target模型验证: 接受["很", "好"]，拒绝["，"]  ← 验证猜测
最终输出: "今天天气很好"  ← 获得2个token
```

### 为什么要动态调整？

不同场景下最优步数不同：

| 场景 | 最优步数 | 原因 |
|------|---------|------|
| 简单文本（天气预报） | 7+ | 模型很确定，猜啥中啥 |
| 复杂推理（数学证明）| 1-3 | 模型犹豫，猜了也白猜 |
| 代码生成 | 3-5 | 中等复杂度 |
| 混合对话 | 动态变化 | 闲聊→简单，推理→复杂 |

**固定步数的问题：**
- 设成7：简单文本很快，复杂场景浪费算力
- 设成3：复杂场景刚好，简单场景太慢

---

### EMA 是什么？

**EMA = Exponential Moving Average（指数移动平均）**

简单说：给最近的数据更高权重，老数据逐渐"遗忘"。

#### 公式

```
EMA_t = (1 - α) × EMA_{t-1} + α × 当前值

α = 平滑因子 (通常 0.1-0.3)
```

#### 生活例子

想象你追踪"每天学习时长"：
- **普通平均**：30天加起来÷30，今天学了多久影响很小
- **EMA**：最近3-5天权重高，能更快反映你最近是不是变懒了

#### 代码示例

```python
# 跟踪接受率
ema_accept_len = 4.0  # 初始猜测
alpha = 0.2           # 20%权重给新数据

# 第1批验证：实际接受了 3 个token
batch_avg = 3.0
ema_accept_len = 0.8 × 4.0 + 0.2 × 3.0 = 3.8

# 第2批验证：实际接受了 5 个token  
batch_avg = 5.0
ema_accept_len = 0.8 × 3.8 + 0.2 × 5.0 = 4.04

# 第3批验证：实际接受了 2 个token
batch_avg = 2.0
ema_accept_len = 0.8 × 4.04 + 0.2 × 2.0 = 3.63
```

**EMA 的好处：**
- 平滑噪声：单批次波动不会剧烈影响判断
- 快速响应：几批后就能反映真实趋势
- 无需存储历史数据：只需保存一个浮点数

---

### 完整的动态调整算法

```python
class AdaptiveController:
    def __init__(self):
        self.candidate_steps = [1, 3, 5, 7]  # 可选步数
        self.current_steps = 5                # 当前步数
        self.ema_accept_len = 4.0             # EMA跟踪值
        self.alpha = 0.2                      # 平滑因子

    def on_batch_complete(self, accepted_tokens_per_seq):
        # 1. 计算本批次平均接受长度
        batch_avg = sum(accepted_tokens_per_seq) / len(accepted_tokens_per_seq)
        # 例: [3, 4, 3, 5, 2] → 平均 3.4
        
        # 2. 更新EMA
        self.ema_accept_len = (1 - self.alpha) * self.ema_accept_len + self.alpha * batch_avg
        # 例: 0.8 × 4.0 + 0.2 × 3.4 = 3.88
        
        # 3. 决定是否调整步数（每N批一次）
        if self.should_adjust():
            self.adjust_steps()
    
    def adjust_steps(self):
        # 原理：如果EMA接近候选步数-0.5，就切换到那个步数
        
        # 当前EMA=3.88，当前步数=5
        # 检查是否降到3：
        # 阈值 = 3 - 0.5 + (-0.25) = 2.25
        # 3.88 > 2.25，不降
        
        # 检查是否升到7：
        # 阈值 = 5 - 0.5 + 0 = 4.5  
        # 3.88 < 4.5，不升
        
        # 所以保持5
        
        # 如果EMA降到2.0：
        # 2.0 <= 2.25，降到步数3
        pass
```

---

### 动态调整过程图示

```
时间 →

步数:    5 ────────────────────────────────
         │         ↓ 发现接受率下降
         │         3 ────────────
         │                      ↑ 接受率回升
         │                      5 ────────
         │                                  
EMA:     4.0 → 3.8 → 3.5 → 2.8 → 2.2 → 3.0 → 3.6 → 4.2
                 ↓ EMA<2.25        ↑ EMA>4.5
                 触发降步数        触发升步数

实际场景：
- 闲聊对话（接受率高）→ 步数升到7 → 加速明显
- 遇到数学题（接受率降）→ 步数降到3 → 避免浪费
- 回到简单话题 → 步数升回 → 再次加速
```

---

## 2. SWA (Sliding Window Attention)

### 核心问题：长文本的灾难

Transformer的Attention机制复杂度是 **O(n²)**：

```
序列长度 | 计算量 (相对)
--------|-------------
512     | 1x
1024    | 4x  
2048    | 16x
4096    | 64x
8192    | 256x  ← 爆显存！
```

**现象**：长文本时 KV Cache 占用线性增长，计算量平方增长。

---

### SWA 解决方案

**核心思想**：每个token只关注最近的 W 个token，而不是全部历史

```
普通 Attention (Full Attention):
"我今天很开心" 的 "很" 要关注: [我, 今, 天, 很] 全部4个

SWA (窗口大小 W=2):
"我今天很开心" 的 "很" 只关注: [天, 很] 最近2个
"心" 只关注: [开, 心] 最近2个
```

---

### 图示对比

**Full Attention（全连接）：**
```
我 ──┬──┬──┬──┐
今 ──┼──┼──┼──┤
天 ──┼──┼──┼──┤
很 ──┼──┼──┼──┤
开 ──┼──┼──┼──┤
心 ──┴──┴──┴──┘
每个token关注所有前面的token，呈下三角矩阵
```

**SWA (W=2，只看最近的2个)：**
```
我 ──┐
今 ──┼──┐
天 ──┴──┼──┐
很 ──┬──┴──┼──┐
开 ──┼──┬──┴──┼──┐  
心 ──┴──┼──┬──┴──┘
      (只看窗口内的，呈带状)
```

---

### 为什么这样可行？

**语言局部性原理**：
- 词语的依赖关系主要集中在附近
- "今天" 和 "天气" 相关（距离2）
- "今天" 和 "昨天" 也相关，但通常有更近的表达

**实际效果**：
- 长文本（32K）内存占用从 32GB → 4GB
- 推理速度提升 5-10x
- 准确率下降 < 2%（可接受）

---

### SWA + 投机解码的挑战

投机解码时，Draft 模型会预测未来多个 token：

```
当前位置: 100
SWA 窗口: 96-100 (最近5个)
Draft 预测: 位置 101, 102, 103 的 token

问题：验证位置103时，Attention应该看到什么？
- 正常应该看到 [96-100, 101, 102, 103]
- 但如果 101, 102 被拒绝了，就不能用了！
```

**SGLang 的解决方案**：
1. 识别 SWA 层和非 SWA 层
2. Draft token 位置计算时考虑滑动窗口
3. 验证阶段正确处理窗口边界
4. 被拒绝的 token 从 KV Cache 正确清除

---

### 代码层面的关键

```python
class SWASpecDecodingManager:
    def __init__(self, sliding_window: int = 4096):
        self.window = sliding_window
    
    def get_attention_range(self, current_pos: int) -> tuple[int, int]:
        """计算当前位置应该关注哪些历史token"""
        start = max(0, current_pos - self.window)
        end = current_pos
        return (start, end)  # 只返回窗口内的范围
    
    def should_prune_draft(self, seq_len: int, num_draft: int) -> bool:
        """如果draft超出窗口，需要裁剪"""
        if seq_len + num_draft > self.window:
            # 超出窗口部分无法有效attention
            return True
        return False
```

---

## 总结对比

| 特性 | 解决的问题 | 核心机制 |
|------|-----------|---------|
| **Adaptive Spec** | 固定步数不适应不同场景 | EMA跟踪接受率，动态调整猜测长度 |
| **SWA** | 长文本KV Cache爆炸 | 只关注最近W个token，降低复杂度 |

**两者结合的价值**：
- SWA 让长文本推理成为可能
- Adaptive Spec 让不同位置都能用最优步数
- 长文本 + 动态优化 = 最大化吞吐

---

## 参考资料

- SGLang Adaptive Spec: `python/sglang/srt/speculative/adaptive_spec_params.py`
- SGLang SWA Support: `python/sglang/srt/mem_cache/swa_radix_cache.py`
- SGLang Release Notes: https://github.com/sgl-project/sglang/releases/tag/v0.5.12
