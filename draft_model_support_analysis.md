# Draft Model (Standalone) 对 Adaptive Spec 和 SWA 的支持分析

> 基于 SGLang v0.5.12 源代码验证

---

## 1. 结论速览

| 特性 | Standalone (Draft Model) | 说明 |
|------|-------------------------|------|
| **Adaptive Spec** | ❌ **不支持**（代码限制） | 架构上可行，但被限制 |
| **SWA** | ⚠️ **部分支持** | 继承 EAGLEWorker 能力 |
| **Spec V2** | ✅ **支持** | 代码明确支持 |

**关键发现：**
- Standalone Worker 继承自 EAGLEWorker
- 虽然声明了 `adaptive_controller` 变量，但**未实际初始化**
- 限制函数明确排除非 EAGLE/EAGLE3 算法

---

## 2. Adaptive Spec 支持分析

### 2.1 代码证据：限制函数

文件：`python/sglang/srt/speculative/adaptive_spec_params.py`

```python
def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"  # ← 排除 Standalone
        )
```

**关键发现：**
- 第22行硬性限制：算法必须是 `"EAGLE"` 或 `"EAGLE3"`
- Standalone 的算法名称是 `"STANDALONE"`
- **不满足条件，返回错误**

---

### 2.2 Standalone Worker 实现

文件：`python/sglang/srt/speculative/standalone_worker.py`

```python
# 第12-13行：导入了 AdaptiveController
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)

# 第54-55行：声明了变量但未初始化
# TODO: Adaptive speculative
self.adaptive_controller: Optional[AdaptiveController] = None
```

**关键发现：**
- ✅ 导入了 AdaptiveController
- ✅ 声明了实例变量
- ❌ **未实例化**（始终为 None）
- ❌ **有 TODO 注释** "TODO: Adaptive speculative"

**对比 EAGLEWorker：**

```python
# eagle_worker.py 第119-122行
self.adaptive_controller: Optional[AdaptiveController] = None
if server_args.speculative_adaptive:
    self.adaptive_controller = AdaptiveController(
        self, config_path=server_args.speculative_adaptive_config
    )  # ← 实际实例化
```

**结论：**
- Standalone Worker 虽然**架构上继承了 EAGLEWorker**
- 但**跳过了 AdaptiveController 的初始化**
- 加上限制函数的硬性检查，**实际上不支持**

---

### 2.3 为什么架构支持但实际不支持

**类继承关系：**

```python
class StandaloneWorker(EAGLEWorker):
    """Standalone draft model worker."""
```

**理论上：**
- StandaloneWorker 继承了 EAGLEWorker 的所有能力
- **应该**支持 Adaptive Spec

**实际上：**
1. 限制函数只允许 EAGLE/EAGLE3
2. StandaloneWorker 未初始化 adaptive_controller
3. 代码中有 TODO 注释，说明是**已知未完成项**

---

## 3. SWA 支持分析

### 3.1 继承 EAGLEWorker 的能力

```python
class StandaloneWorker(EAGLEWorker):
```

**SWA 相关能力：**
- ✅ 继承 EAGLEWorker 的 `maybe_evict_swa()` 调用
- ✅ 继承 EAGLEWorker 的 KV Cache 管理
- ⚠️ 但未明确测试 SWA + Standalone 的组合

### 3.2 Spec V2 支持确认

文件：`python/sglang/srt/speculative/spec_info.py` 第127-128行

```python
def supports_spec_v2(self) -> bool:
    return (self.is_eagle() and not self.is_frozen_kv_mtp()) or self.is_standalone()
```

**关键发现：**
- ✅ `is_standalone()` 明确返回支持 Spec V2
- Spec V2 包含部分 SWA 能力

---

## 4. 特性支持矩阵（最终版）

### 4.1 SGLang v0.5.12 完整支持表

| 投机方式 | Adaptive Spec | SWA完整 | SWA基础 | Spec V2 |
|----------|---------------|---------|---------|---------|
| **EAGLE3** | ✅ | ✅ | ✅ | ✅ |
| **EAGLE** | ✅ | ✅ | ✅ | ✅ |
| **STANDALONE** | ❌ | ⚠️ | ✅ | ✅ |
| **MTP** | ❌ | ⚠️ | ✅ | ❌ |
| **N-gram** | ❌ | ❌ | ✅ | ❌ |
| **DFlash** | ❌ | ⚠️ | ✅ | ✅ |

---

## 5. 代码验证命令

```bash
# 1. 检查限制函数
grep -A5 "adaptive_unsupported_reason" adaptive_spec_params.py

# 2. 检查 Standalone Worker
grep "adaptive_controller" standalone_worker.py
# 输出: self.adaptive_controller: Optional[AdaptiveController] = None

# 3. 检查 TODO
grep "TODO" standalone_worker.py
# 输出: # TODO: Adaptive speculative

# 4. 检查继承
grep "class StandaloneWorker" standalone_worker.py
# 输出: class StandaloneWorker(EAGLEWorker):
```

---

## 6. 使用建议

| 需求 | 推荐方案 | 原因 |
|------|---------|------|
| 需要 Adaptive Spec | **EAGLE3** / **EAGLE** | Standalone 不支持 |
| 需要 Draft Model + 自适应 | ❌ 无法实现 | 代码限制 |
| 简单 Draft Model | ✅ Standalone | 基础功能可用 |

---

## 7. 总结

**Draft Model (Standalone) 当前状态：**
- ❌ **不支持 Adaptive Spec**（硬性代码限制）
- ⚠️ **部分支持 SWA**（继承 EAGLEWorker）
- ✅ **支持 Spec V2**（代码明确支持）

**只有 EAGLE/EAGLE3 完全支持 Adaptive Spec 和 SWA！**
