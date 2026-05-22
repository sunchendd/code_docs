# SGLang MTP 对 Adaptive Spec 和 SWA 的支持情况（代码验证）

> 基于 SGLang v0.5.12 源代码分析

---

## 1. 结论速览

| 特性 | EAGLE/EAGLE3 | MTP (Frozen-KV) | 说明 |
|------|-------------|-----------------|------|
| **Adaptive Spec** | ✅ 完整支持 | ❌ **不支持** | 代码明确限制 |
| **SWA** | ✅ 完整支持 | ⚠️ **基础支持** | 支持evict，但非完整投机集成 |

**重要发现：**
- **MTP 不支持 Adaptive Spec**（代码硬性限制）
- **MTP 有基础 SWA 支持**（仅支持 `maybe_evict_swa()`）
- 这两个特性主要是为 **EAGLE/EAGLE3** 设计的

---

## 2. Adaptive Spec 支持验证

### 2.1 代码证据：硬性限制

文件：`python/sglang/srt/speculative/adaptive_spec_params.py`

```python
def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"  # ← 明确排除MTP
        )
```

**关键发现：**
- 第22-26行明确检查 `speculative_algorithm` 必须是 `"EAGLE"` 或 `"EAGLE3"`
- MTP 的算法名称通常是 `"MTP"` 或 `"FROZEN_KV_MTP"`
- **不满足条件，会直接返回错误**

---

### 2.2 EAGLE Worker 的完整实现

文件：`python/sglang/srt/speculative/eagle_worker.py` 和 `eagle_worker_v2.py`

```python
# eagle_worker.py 第37行
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)

# eagle_worker.py 第119-122行
self.adaptive_controller: Optional[AdaptiveController] = None
if server_args.speculative_adaptive:
    self.adaptive_controller = AdaptiveController(
        self, config_path=server_args.speculative_adaptive_config
    )

# eagle_worker.py 第541-542行（验证后回调）
if self.adaptive_controller is not None:
    self.adaptive_controller.on_verify_complete(num_correct_drafts_per_req)
```

**对比 MTP Worker：**

文件：`python/sglang/srt/speculative/frozen_kv_mtp_worker.py`

```python
# 搜索 adaptive_controller：无结果
# 搜索 AdaptiveController：无结果

# 第193行只有一个暴露的接口
@property
def get_attn_backend(self):  # pragma: no cover - exposed for adaptive
    return self.draft_attn_backend
```

**结论：**
- MTP Worker **没有**实例化 `AdaptiveController`
- 虽然有 `get_attn_backend()` 接口（标记为"exposed for adaptive"）
- 但**从未实际使用**Adaptive Spec功能

---

### 2.3 为什么不支持 MTP？

**技术原因：**

```
EAGLE/EAGLE3 特性：
- 独立的 draft model
- 可动态调整猜测步数（num_speculative_steps）
- 每步都重新生成 draft tokens
- 适合动态调整

MTP (Frozen-KV) 特性：
- 使用目标模型自身的MTP head
- 固定深度的多token预测（如固定预测1-2个）
- 无法动态改变"步数"（模型架构固定）
- 调整步数需要重新加载模型
```

**代码层面的不匹配：**

```python
# Adaptive Spec 的核心是调整 num_speculative_steps
class AdaptiveSpeculativeParams:
    def _recompute_steps(self) -> bool:
        # 动态调整步数...
        target = self.candidate_steps[current_idx]
        self.current_steps = target

# 但 MTP 的步数是模型定义时固定的
class FrozenKVMTPWorker:
    def __init__(self, ...):
        self.speculative_num_steps = server_args.speculative_num_steps
        # 固定值，无法动态调整
```

---

## 3. SWA 支持验证

### 3.1 MTP 的基础 SWA 支持

文件：`python/sglang/srt/speculative/frozen_kv_mtp_worker.py`

```python
# 第574行
def draft(self, batch: ScheduleBatch):
    # ...
    batch.maybe_evict_swa()  # ← 调用SWA evict
    # ...
```

**解读：**
- ✅ MTP Worker **支持**基础的 `maybe_evict_swa()` 操作
- ⚠️ 但这只是**通用的KV Cache管理**，不是投机解码特有的SWA集成

---

### 3.2 EAGLE 的完整 SWA 支持

对比 EAGLE Worker 的 SWA 集成：

```python
# eagle_worker_v2.py - 多处SWA相关处理
- SWA HiCache 集成 (#24664)
- SWA tree mask 支持
- SWA position 计算
- UnifiedRadixTree + SWA (#23391)

# frozen_kv_mtp_worker.py
- 仅调用 batch.maybe_evict_swa()
- 无专门的SWA投机解码逻辑
```

**PR 记录验证：**

```
#24664: SWA support for EAGLE-3 drafter
- 明确提到 "EAGLE-3 drafter"
- 不涉及MTP

#23391: SWA HiCache for unified radix cache
- 主要是HiCache框架
- MTP未使用HiCache
```

---

### 3.3 SWA + 投机解码的挑战

**MTP 的限制：**

```
场景：128K文档，使用MTP

问题1：MTP head通常只预测1-2个token
       - 无法像EAGLE3那样调整步数
       - SWA窗口内的draft token数量有限

问题2：MTP使用目标模型自身的KV Cache
       - Frozen-KV MTP复用目标模型的KV
       - SWA evict可能影响MTP的frozen KV假设

问题3：缺乏专门的SWA tree mask处理
       - EAGLE有专门的SWA tree mask构建
       - MTP没有类似逻辑
```

**代码证据：**

```python
# eagle_utils.py - EAGLE的SWA tree mask
class TreeMaskMode(Enum):
    FULL_MASK = auto()
    QLEN_ONLY = auto()
    QLEN_ONLY_BITPACKING = auto()

def build_tree_kernel_efficient(
    tree_mask_mode: TreeMaskMode,  # ← SWA相关
    ...
):
    # 专门的SWA tree mask处理

# frozen_kv_mtp_worker.py - 无类似处理
# 只有基础的batch.maybe_evict_swa()
```

---

## 4. 特性支持矩阵（修正版）

### 4.1 SGLang v0.5.12 各投机方式特性支持

| 投机方式 | Adaptive Spec | SWA完整支持 | SWA基础支持 | 说明 |
|----------|---------------|-------------|-------------|------|
| **EAGLE3** | ✅ | ✅ | ✅ | 完全支持两个特性 |
| **EAGLE** | ✅ | ✅ | ✅ | 完全支持两个特性 |
| **MTP (Frozen-KV)** | ❌ | ⚠️ | ✅ | 不支持Adaptive，SWA基础功能 |
| **Draft Model** | ✅ | ⚠️ | ✅ | 支持Adaptive，SWA需验证 |
| **N-gram** | ❌ | ❌ | ✅ | 都不支持 |
| **Standalone** | ✅ | ⚠️ | ✅ | 类似EAGLE |

### 4.2 详细对比

| 维度 | EAGLE3 | MTP (Frozen-KV) |
|------|--------|-----------------|
| **Adaptive Spec** | | |
| - 支持 | ✅ 是 | ❌ 否（硬性限制） |
| - 实现 | 完整 | 无 |
| - 原因 | 独立draft，可动态调步 | 模型架构固定，步数不可变 |
| **SWA** | | |
| - 基础evict | ✅ | ✅ |
| - 投机集成 | ✅ 完整 | ⚠️ 基础 |
| - Tree Mask | ✅ SWA感知 | ❌ 普通mask |
| - Position计算 | ✅ SWA感知 | ⚠️ 基础 |
| - HiCache集成 | ✅ 完整 | ❌ 无 |

---

## 5. 代码层面的不支持证据

### 5.1 Adaptive Spec 不支持的完整检查清单

```bash
# 1. 检查MTP worker是否import AdaptiveController
$ grep "AdaptiveController" frozen_kv_mtp_worker.py
# 结果：无

# 2. 检查MTP worker是否实例化adaptive_controller
$ grep "adaptive_controller" frozen_kv_mtp_worker.py
# 结果：无

# 3. 检查MTP worker是否调用on_verify_complete
$ grep "on_verify_complete" frozen_kv_mtp_worker.py
# 结果：无

# 4. 检查限制函数
$ grep -A5 "adaptive_unsupported_reason" adaptive_spec_params.py
# 结果：明确检查 algorithm in ("EAGLE", "EAGLE3")
```

### 5.2 SWA 支持程度的检查

```bash
# 1. MTP中的SWA
$ grep -n "swa\|SWA" frozen_kv_mtp_worker.py
574:        batch.maybe_evict_swa()  # 仅一处

# 2. EAGLE中的SWA
$ grep -n "swa\|SWA\|tree_mask_mode" eagle_worker_v2.py | wc -l
# 结果：多处（10+）

# 3. SWA相关PR
$ git log --grep="SWA" --oneline | head
# 结果：主要提到EAGLE-3
```

---

## 6. 为什么文档会有误导？

### 6.1 SGLang v0.5.12 Release Notes

```
Speculative Decoding V2 maturation: 
- Adaptive Spec V2, 
- EAGLE-3 SWA + newer drafters
- ...

注意：这里的描述可能被误解为"所有投机方式都支持"
      实际是针对EAGLE-3的改进
```

### 6.2 实际支持情况

```
Release Notes中提到的：
✅ Adaptive Spec V2 → 仅 EAGLE/EAGLE3
✅ EAGLE-3 SWA → 仅 EAGLE-3
✅ newer drafters → 指EAGLE-3变体

未提及MTP的Adaptive或SWA支持
```

---

## 7. 修正后的建议

### 7.1 使用建议（基于代码事实）

| 场景 | 之前建议 | 修正后建议 | 原因 |
|------|---------|-----------|------|
| 需要Adaptive Spec | MTP + Adaptive | **EAGLE3** + Adaptive | MTP不支持 |
| 需要SWA + Spec | MTP + SWA | **EAGLE3** + SWA | MTP仅基础支持 |
| 长文档MTP | MTP + SWA | **EAGLE3** 或纯MTP | MTP投机效果有限 |
| 动态负载 | 任意 + Adaptive | **EAGLE/EAGLE3** | 仅它们支持 |

### 7.2 配置修正

```python
# ❌ 错误配置（MTP不支持Adaptive）
{
    "method": "deepseek_mtp",  # 或 frozen_kv_mtp
    "adaptive_config": {"enabled": True}  # 会被拒绝
}

# ✅ 正确配置
{
    "method": "eagle3",  # 或 eagle
    "adaptive_config": {"enabled": True}  # 正常工作
}

# ✅ 正确配置（长文档）
{
    "method": "eagle3",
    "sliding_window": 4096,  # SWA支持
    "adaptive_config": {"enabled": True}
}
```

---

## 8. 总结

### 8.1 核心发现

| 问题 | 答案 |
|------|------|
| MTP支持Adaptive Spec？ | **不支持**（代码硬性限制） |
| MTP支持SWA？ | **基础支持**（仅evict，非完整集成） |
| 哪个投机方式支持这两个特性？ | **EAGLE/EAGLE3** |
| 为什么MTP不支持？ | 架构限制（固定步数，frozen KV） |

### 8.2 影响

**对移植vLLM的影响：**
- 如果只需要EAGLE3支持 → 已有参考实现（SGLang）
- 如果要扩展MTP支持 → 需要额外设计（非简单移植）
- SWA + MTP集成 → 优先级降低（效果有限）

**对用户的影响：**
- 使用MTP → 无需考虑Adaptive和SWA投机集成
- 需要动态调整 → 切换到EAGLE3
- 需要长文档投机 → 使用EAGLE3 + SWA

---

## 参考代码

- 限制代码：`python/sglang/srt/speculative/adaptive_spec_params.py:22`
- EAGLE实现：`python/sglang/srt/speculative/eagle_worker.py:119-122`
- MTP实现：`python/sglang/srt/speculative/frozen_kv_mtp_worker.py:574`
