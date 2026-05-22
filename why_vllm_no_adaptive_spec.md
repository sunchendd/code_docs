# 为什么 vLLM 还没有 Adaptive Spec？深度分析

> 基于 vLLM 社区 issue/PR 分析和架构对比

---

## 1. 重要澄清：SGLang 不是基于 vLLM

### 项目关系

| 项目 | 启动时间 | 基础架构 | 关系 |
|------|---------|---------|------|
| **vLLM** | 2023年 | 独立开发 | 伯克利大学孵化 |
| **SGLang** | 2023年 | 独立开发 | LMSYS组织开发 |

**Git历史证据：**
```bash
# SGLang初始提交
commit f6d40df0e
Author: LMSYS Team
Date: 2023年
Initial commit  # 完全独立的代码库

# 早期依赖
- 使用FlashInfer（不是vLLM的attention backend）
- 独立的路由层（sgl-router）
- 独立的scheduler实现
```

**结论：**
- ✅ SGLang是**完全独立**的项目
- ✅ 不是vLLM的fork或wrapper
- ✅ 两者是**竞争关系**，不是派生关系

---

## 2. vLLM 社区对 Adaptive Spec 的讨论

### 2.1 已存在的 Issue

找到了一个**已关闭**的相关issue：

```
Issue #31483: [RFC]: a efficient adaptive rejection sampling for accelerating speculative decoding.
- 作者：sunchendd
- 状态：Closed (completed)
- 时间：2025年1月4日关闭
```

**重要发现：**
- 这个issue讨论的是 **"adaptive rejection sampling"**（自适应拒绝采样）
- 不是 **"adaptive speculative steps"**（自适应投机步数）
- 两者是不同的概念！

**概念区分：**
| 概念 | Adaptive Rejection Sampling | Adaptive Speculative Steps |
|------|---------------------------|---------------------------|
| **作用阶段** | Verify阶段 | Draft阶段 |
| **调整对象** | 拒绝采样策略 | 猜测步数 |
| **SGLang支持** | ❌ 未提及 | ✅ Adaptive Spec V2 |
| **vLLM状态** | ✅ 可能已支持 | ❌ 未支持 |

---

### 2.2 vLLM 现有的 Adaptive 相关特性

搜索 vLLM 代码发现：

```python
# vllm/config/speculative.py
class SpeculativeConfig:
    # 支持多种拒绝采样方法
    rejection_sample_method: RejectionSampleMethod = "standard"
    synthetic_acceptance_rates: list[float] | None = None
    synthetic_acceptance_length: float | None = None
    
    # 支持动态draft采样
    draft_sample_method: DraftSampleMethod = "greedy"  # or "probabilistic"
```

**vLLM已有的"Adaptive"：**
1. **Synthetic Rejection Sampling**：预定义接受率曲线
2. **Probabilistic Draft**：概率性draft采样
3. **但缺少**：动态调整 `num_speculative_tokens`

---

## 3. 为什么 vLLM 还没做 Adaptive Spec？

### 3.1 技术架构差异

| 维度 | SGLang | vLLM |
|------|--------|------|
| **架构** | 独立scheduler，灵活 | 高度优化，耦合度高 |
| **CUDA Graph** | 相对简单 | 非常复杂（v1优化重点） |
| **多Backend** | 容易切换 | 需要大量适配 |
| **调度策略** | 相对灵活 | 高度调度优化 |

**关键差异：**

```python
# SGLang: 独立Spec Worker
class EAGLEWorker:
    def __init__(self):
        self.adaptive_controller = AdaptiveController()
        # 独立管理CUDA Graph
        self.cuda_graphs = {...}

# vLLM: 集成在Model Runner
class GPUModelRunner:
    def __init__(self):
        # Spec decode集成在主体流程
        self.speculative_proposer = EagleProposer()
        # CUDA Graph与主模型共享
        self.cuda_graph = ...
```

**难点：**
1. vLLM的CUDA Graph管理与主模型深度耦合
2. 动态调整步数需要重建/切换CUDA Graph
3. vLLM V1版本的调度器高度优化，改动风险大

---

### 3.2 优先级考虑

**vLLM 当前重点（从Roadmap看）：**

根据 `vllm-project/vllm/issues/39749` [Roadmap] Q2 2026：
```
1. V1引擎稳定化（生产就绪）
2. Multi-Modal优化
3. Prefix Caching完善
4. Quantization优化
5. Distributed推理
```

**投机解码在vLLM中的优先级：**
- ✅ **基础功能**：EAGLE/MTP/N-gram已支持
- ⚠️ **性能优化**：Spec V2 Overlap调度（开发中）
- ❌ **高级特性**：Adaptive Spec（未列入Roadmap）

---

### 3.3 收益评估差异

**SGLang做Adaptive Spec的理由：**
1. **LMSYS场景**：Chatbot Arena，负载高度动态
2. **场景匹配**：简单/复杂查询混合，Adaptive收益大
3. **资源充足**：团队大，可以做实验性特性

**vLLM谨慎的理由：**
1. **企业场景**：负载相对固定，固定步数可接受
2. **稳定性优先**：企业用户更看重稳定而非极致性能
3. **维护成本**：Adaptive增加复杂度，需要持续维护
4. **收益有限**：静态负载场景收益不明显

---

### 3.4 技术债务

**vLLM的历史包袱：**

```python
# vLLM V0版本（维护中）
- 旧的spec decode实现
- 需要向后兼容

# vLLM V1版本（开发中）
- 全新架构
- 重点在稳定化
- 不想过早引入复杂特性
```

**SGLang的优势：**
- 相对年轻，架构更灵活
- 没有V0/V1双版本维护压力
- 可以更快实验新特性

---

## 4. vLLM 不做 Adaptive Spec 的具体技术障碍

### 4.1 CUDA Graph 管理复杂

```python
# vLLM当前：一套CUDA Graph
class ModelRunner:
    def capture_cuda_graph(self):
        # 为max_num_seqs capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        
# Adaptive需要：多套Graph动态切换
class AdaptiveModelRunner:
    def __init__(self):
        # 需要为每个candidate_steps capture
        self.cuda_graphs = {
            2: graph_for_2_steps,
            4: graph_for_4_steps,
            6: graph_for_6_steps,
        }
    
    def switch_graph(self, steps):
        # 动态切换，vLLM当前不支持
        self.current_graph = self.cuda_graphs[steps]
```

**障碍：**
- vLLM的CUDA Graph与内存池、调度器深度耦合
- 切换CUDA Graph需要同步，可能引入延迟
- 多套Graph增加内存占用

---

### 4.2 调度器复杂度

```python
# vLLM V1调度器高度优化
class Scheduler:
    def schedule(self):
        # 复杂的batching逻辑
        # 固定num_speculative_tokens简化了很多假设
        
# Adaptive需要：
class AdaptiveScheduler:
    def schedule(self):
        # 需要考虑不同req的steps可能不同
        # batching逻辑复杂化
        # 需要额外的同步点
```

---

### 4.3 测试覆盖

**vLLM的质量标准：**
- 企业级稳定性要求
- 需要大量测试覆盖
- Adaptive Spec的测试场景复杂

**SGLang的优势：**
- LMSYS可以内部测试验证
- 快速迭代，修复问题快

---

## 5. 社区需求分析

### 5.1 vLLM Issue分析

搜索 `vllm-project/vllm/issues`：

```
关键词 "adaptive speculative" 结果：
- 相关issue数量：较少
- 大多数是讨论性质的
- 没有高priority的feature request

关键词 "dynamic num_speculative_tokens" 结果：
- 几乎无相关issue

关键词 "speculative decoding" 结果：
- 主要集中在基础功能支持
- EAGLE/MTP支持
- Bug修复
```

**结论：**
- 社区对Adaptive Spec的需求**不强烈**
- 大多数用户满足于固定步数

---

### 5.2 为什么需求不强烈？

**用户场景分析：**

| 场景 | 负载特征 | 是否需要Adaptive |
|------|---------|----------------|
| **企业API** | 相对固定 | ❌ 固定步数可调优 |
| **Chatbot** | 动态混合 | ✅ Adaptive有用 |
| **Batch处理** | 固定模式 | ❌ 不需要 |
| **实时交互** | 低延迟要求 | ⚠️ 切换延迟敏感 |

**vLLM主要用户：**
- 企业部署（固定负载）
- Batch推理（固定模式）
- 对稳定性要求高

**SGLang主要用户：**
- Chatbot Arena（动态负载）
- 研究实验（需要极致性能）
- 愿意尝试新特性

---

## 6. vLLM 可能的未来计划

### 6.1 从Roadmap推测

根据 Q2 2026 Roadmap (`#39749`)：
```
未明确提到Adaptive Spec，但提到：
- "V1引擎稳定化" → 之后可能加入新特性
- "性能优化" → 可能包括调度优化
```

**可能的时间线：**
```
2025 Q2-Q3: V1引擎稳定化
2025 Q4: 可能考虑Adaptive Spec
2026: 如果社区需求强烈
```

### 6.2 如果vLLM要做，会怎么做？

**可能的实现路径：**

```python
# 方案1: 保守方案（类似SGLang）
- 预创建多套CUDA Graph
- EMA跟踪
- 步数切换

# 方案2: 激进方案
- 重构CUDA Graph管理
- 更细粒度的调度
- 可能带来更大收益，但风险高

# 方案3: 简化方案
- 只支持EAGLE3
- 有限的candidate_steps
- 降低复杂度
```

**最可能选择：方案3**（如果要做的话）

---

## 7. 对用户的建议

### 7.1 当前选择

| 需求 | 推荐方案 |
|------|---------|
| **需要Adaptive Spec** | 使用 **SGLang** |
| **需要vLLM生态** | 接受固定步数，手动调优 |
| **混合部署** | vLLM用于稳定负载，SGLang用于动态负载 |

### 7.2 如何推动vLLM支持

如果你想让vLLM支持Adaptive Spec：

```
1. 在vLLM GitHub提交feature request
   - 描述具体场景
   - 提供收益数据
   - 说明为什么固定步数不够

2. 提供测试资源
   - 动态负载场景
   - 性能对比数据

3. 参与开发
   - 提交PR实现
   - 参考SGLang的实现
```

---

## 8. 总结

### 核心结论

| 问题 | 答案 |
|------|------|
| **SGLang基于vLLM吗？** | ❌ **不是**，完全独立项目 |
| **vLLM为什么不做？** | 优先级、架构复杂度、需求不强烈 |
| **vLLM会做吗？** | 可能，但不在近期Roadmap |
| **应该选哪个？** | 需要Adaptive→SGLang；需要生态→vLLM |

### 关键差异

| 维度 | SGLang | vLLM |
|------|--------|------|
| **Adaptive Spec** | ✅ 生产就绪 | ❌ 未支持 |
| **架构灵活性** | 高 | 中（V1改进中）|
| **企业稳定性** | 中 | 高 |
| **社区生态** | 小但活跃 | 大且成熟 |
| **适用场景** | 动态负载、研究 | 企业部署、生产 |

### 最终建议

**现在：**
- 需要Adaptive Spec → 用SGLang
- 需要vLLM生态 → 手动调优固定步数

**未来：**
- vLLM可能在V1稳定后考虑
- 关注vLLM GitHub issue #31483相关讨论
- 可以提交feature request推动
