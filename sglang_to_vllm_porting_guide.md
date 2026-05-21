# SGLang 投机解码特性移植到 vLLM 实施指南

## 1. Adaptive Speculative Decoding 移植指南

### 1.1 核心组件映射

| SGLang 组件 | vLLM 对应位置 | 移植策略 |
|------------|--------------|---------|
| `AdaptiveSpeculativeParams` | `vllm/config/speculative.py` | 新增配置类 |
| `AdaptiveController` | `vllm/v1/spec_decode/adaptive_controller.py` | 新增控制器 |
| `SpecRuntimeState` | `vllm/v1/spec_decode/runtime_state.py` | 新增状态管理 |

### 1.2 配置层实现

```python
# vllm/config/speculative.py 新增

@config
class AdaptiveSpeculativeConfig:
    """Configuration for adaptive speculative decoding."""

    enabled: bool = False
    """Enable adaptive speculative decoding."""

    candidate_steps: list[int] = Field(default_factory=lambda: [1, 3, 5, 7])
    """Candidate speculative steps to switch between."""

    ema_alpha: float = 0.2
    """EMA smoothing factor for acceptance rate tracking."""

    update_interval: int = 5
    """Number of batches between step adjustments."""

    warmup_batches: int = 10
    """Number of warmup batches before adaptation starts."""

    down_hysteresis: float = -0.25
    """Hysteresis threshold for decreasing steps."""

    up_hysteresis: float = 0.0
    """Hysteresis threshold for increasing steps."""

    @field_validator("candidate_steps")
    @classmethod
    def validate_candidate_steps(cls, v: list[int]) -> list[int]:
        if len(v) < 2:
            raise ValueError("candidate_steps must have at least 2 values")
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("candidate_steps must be positive integers")
        return sorted(set(v))
```

### 1.3 核心算法实现

```python
# vllm/v1/spec_decode/adaptive_controller.py

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

logger = init_logger(__name__)


@dataclass
class AdaptiveMetrics:
    """Metrics for adaptive speculative decoding."""
    batch_count: int = 0
    ema_accept_len: float = 0.0
    current_steps: int = 0
    target_steps: int = 0


class AdaptiveSpeculativeController:
    """Controller for adaptive speculative decoding.

    Dynamically adjusts num_speculative_tokens based on observed acceptance rates.

    Algorithm:
        1. Track acceptance length via EMA: ema = (1-alpha)*ema + alpha*accept_len
        2. Recompute target steps every update_interval batches (after warmup)
        3. Apply hysteresis to avoid oscillation
        4. Clamp to candidate_steps range
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        proposer: "SpecDecodeBaseProposer",
    ):
        self.config = vllm_config.speculative_config.adaptive_config
        self.proposer = proposer

        # Initialize state
        self.candidate_steps = self.config.candidate_steps
        self.current_steps = proposer.num_speculative_tokens

        # Ensure current steps is in candidates
        if self.current_steps not in self.candidate_steps:
            self.candidate_steps = sorted(set(self.candidate_steps + [self.current_steps]))
            logger.info(
                f"Added current steps {self.current_steps} to candidate_steps: "
                f"{self.candidate_steps}"
            )

        # EMA state
        self.ema_accept_len = float(self.current_steps - 1)
        self.batch_count = 0
        self.last_update_time = time.time()

        # Metrics
        self.metrics = AdaptiveMetrics(
            current_steps=self.current_steps,
            ema_accept_len=self.ema_accept_len,
        )

        logger.info(
            f"AdaptiveSpeculativeController initialized: "
            f"steps={self.current_steps}, candidates={self.candidate_steps}, "
            f"ema_alpha={self.config.ema_alpha}"
        )

    def update(self, num_accepted_tokens_per_seq: list[int]) -> bool:
        """Update EMA with observed acceptance lengths.

        Args:
            num_accepted_tokens_per_seq: Per-sequence accepted draft token counts

        Returns:
            True if steps should be changed
        """
        if not num_accepted_tokens_per_seq:
            return False

        # Update EMA
        batch_avg = sum(num_accepted_tokens_per_seq) / len(num_accepted_tokens_per_seq)
        alpha = self.config.ema_alpha
        self.ema_accept_len = (1 - alpha) * self.ema_accept_len + alpha * batch_avg

        self.batch_count += 1
        self.metrics.batch_count = self.batch_count
        self.metrics.ema_accept_len = self.ema_accept_len

        # Skip during warmup
        if self.batch_count <= self.config.warmup_batches:
            return False

        # Check update interval
        if (self.batch_count - self.config.warmup_batches) % self.config.update_interval != 0:
            return False

        return self._recompute_steps()

    def _recompute_steps(self) -> bool:
        """Recompute target steps from EMA. Returns True if steps changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)

        # Try decreasing steps
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = prev_step - 0.5 + self.config.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        # Try increasing steps
        while current_idx < len(self.candidate_steps) - 1:
            current_step = self.candidate_steps[current_idx]
            rise_threshold = current_step - 0.5 + self.config.up_hysteresis
            if self.ema_accept_len > rise_threshold:
                current_idx += 1
            else:
                break

        target = self.candidate_steps[current_idx]

        if target != old_steps:
            self.current_steps = target
            self.metrics.current_steps = target
            self.metrics.target_steps = target

            logger.info(
                f"Adaptive spec steps changed: {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})"
            )
            return True

        return False

    def get_target_steps(self) -> int:
        """Get current target speculative steps."""
        return self.current_steps

    def get_metrics(self) -> AdaptiveMetrics:
        """Get current metrics."""
        return self.metrics
```

### 1.4 与 vLLM Proposer 集成

```python
# vllm/v1/spec_decode/llm_base_proposer.py 修改

class SpecDecodeBaseProposer:
    def __init__(...):
        # ... existing init code ...

        # Add adaptive controller if enabled
        self.adaptive_controller: Optional[AdaptiveSpeculativeController] = None
        if (
            self.speculative_config.adaptive_config is not None
            and self.speculative_config.adaptive_config.enabled
        ):
            from vllm.v1.spec_decode.adaptive_controller import (
                AdaptiveSpeculativeController,
            )
            self.adaptive_controller = AdaptiveSpeculativeController(
                vllm_config, self
            )

    def propose_tree(self, ...) -> SpecDecodeMetadata:
        """Propose speculative tokens with adaptive support."""

        # Use adaptive steps if controller is active
        num_spec_tokens = self.num_speculative_tokens
        if self.adaptive_controller is not None:
            num_spec_tokens = self.adaptive_controller.get_target_steps()

        # ... existing proposal logic using num_spec_tokens ...

    def on_verification_complete(
        self,
        accepted_token_counts: list[int],
    ) -> None:
        """Called after verification to update adaptive controller."""
        if self.adaptive_controller is not None:
            should_update = self.adaptive_controller.update(accepted_token_counts)
            if should_update:
                self._on_adaptive_steps_changed()

    def _on_adaptive_steps_changed(self) -> None:
        """Handle steps change - may need to reallocate buffers or update CUDA graphs."""
        new_steps = self.adaptive_controller.get_target_steps()

        # Update buffer sizes if needed
        # This is the challenging part - CUDA graphs and buffer sizes
        # may need to be adjusted dynamically

        logger.info(f"Adaptive steps changed to {new_steps}, updating proposer state")

        # For Phase 1: just update the num_speculative_tokens
        # For Phase 2: implement proper buffer/CUDA graph switching
        self.num_speculative_tokens = new_steps
```

---

## 2. TreeMaskMode 移植指南

### 2.1 设计思路

```python
# vllm/v1/attention/backends/tree_attn.py

from enum import Enum, auto
from typing import Optional

import torch


class TreeMaskMode(Enum):
    """Different modes for tree attention masking."""
    FULL_MASK = auto()              # Full attention matrix
    QLEN_ONLY = auto()              # Query length only (memory efficient)
    QLEN_ONLY_BITPACKING = auto()   # Bit-packed version (most efficient)


class TreeMaskBuilder:
    """Build tree attention masks with different optimization modes."""

    def __init__(
        self,
        mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
        max_tree_size: int = 256,
        device: str = "cuda",
    ):
        self.mode = mode
        self.max_tree_size = max_tree_size
        self.device = device

        # Pre-allocate buffers based on mode
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate mask buffers based on mode."""
        if self.mode == TreeMaskMode.FULL_MASK:
            # Full (tree_size, seq_len + tree_size) mask
            self.mask_buffer = torch.full(
                (self.max_tree_size, self.max_tree_size * 4),
                fill_value=True,
                dtype=torch.bool,
                device=self.device,
            )
        elif self.mode == TreeMaskMode.QLEN_ONLY:
            # Only (tree_size,) query lengths
            self.mask_buffer = torch.full(
                (self.max_tree_size,),
                fill_value=0,
                dtype=torch.int32,
                device=self.device,
            )
        elif self.mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            # Bit-packed mask
            packed_size = (self.max_tree_size + 63) // 64
            self.mask_buffer = torch.zeros(
                (self.max_tree_size, packed_size),
                dtype=torch.int64,
                device=self.device,
            )

    def build_mask(
        self,
        tree_structure: torch.Tensor,
        seq_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """Build tree mask for given structure."""
        if self.mode == TreeMaskMode.FULL_MASK:
            return self._build_full_mask(tree_structure, seq_lens, num_draft_tokens)
        elif self.mode == TreeMaskMode.QLEN_ONLY:
            return self._build_qlen_mask(tree_structure, num_draft_tokens)
        elif self.mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            return self._build_bitpacked_mask(tree_structure, num_draft_tokens)

    def _build_full_mask(
        self,
        tree_structure: torch.Tensor,
        seq_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """Build full attention mask."""
        batch_size = len(seq_lens)
        seq_len_sum = seq_lens.sum().item()

        # (num_draft_tokens, seq_len_sum + num_draft_tokens)
        mask = self.mask_buffer[:num_draft_tokens, :seq_len_sum + num_draft_tokens]
        mask.fill_(True)

        # Apply tree structure
        for i in range(num_draft_tokens):
            # Each token attends to its ancestors in the tree
            ancestors = tree_structure[i]
            mask[i, seq_len_sum:] = ancestors

        return mask

    def _build_qlen_mask(
        self,
        tree_structure: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """Build query-length-only mask (memory efficient)."""
        mask = self.mask_buffer[:num_draft_tokens]

        # Store query lengths for each position
        for i in range(num_draft_tokens):
            # Count number of ancestors (query length)
            mask[i] = tree_structure[i].sum().item()

        return mask

    def _build_bitpacked_mask(
        self,
        tree_structure: torch.Tensor,
        num_draft_tokens: int,
    ) -> torch.Tensor:
        """Build bit-packed mask (most memory efficient)."""
        packed_size = (num_draft_tokens + 63) // 64
        mask = self.mask_buffer[:num_draft_tokens, :packed_size]
        mask.fill_(0)

        # Pack tree structure into bits
        for i in range(num_draft_tokens):
            for j in range(num_draft_tokens):
                if tree_structure[i, j]:
                    word_idx = j // 64
                    bit_idx = j % 64
                    mask[i, word_idx] |= (1 << bit_idx)

        return mask
```

---

## 3. SWA + Speculative Decoding 移植指南

### 3.1 核心挑战

SGLang 的 SWA + 投机解码集成主要依赖 `HiCache` 和 `UnifiedRadixTree`。vLLM 需要：
1. 在投机解码路径中识别 SWA 层
2. 管理 SWA 特有的 KV Cache 布局
3. 处理 SWA 滑动窗口对 draft token 的影响

### 3.2 实现草案

```python
# vllm/v1/spec_decode/swa_utils.py

import torch

from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.logger import init_logger

logger = init_logger(__name__)


class SWASpecDecodingManager:
    """Manages SWA-specific handling in speculative decoding."""

    def __init__(
        self,
        sliding_window: int,
        num_speculative_tokens: int,
    ):
        self.sliding_window = sliding_window
        self.num_speculative_tokens = num_speculative_tokens

    def compute_swa_positions(
        self,
        seq_lens: torch.Tensor,
        draft_token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute positions for draft tokens considering SWA."""
        # For SWA, we need to ensure draft tokens don't exceed window
        max_positions = seq_lens + self.num_speculative_tokens

        # Compute relative positions within window
        swa_positions = torch.remainder(draft_token_positions, self.sliding_window)

        return swa_positions

    def should_prune_draft_tokens(
        self,
        seq_len: int,
        num_draft_tokens: int,
    ) -> tuple[bool, int]:
        """Determine if draft tokens should be pruned due to SWA boundary."""
        # If adding draft tokens would exceed window, prune
        if seq_len + num_draft_tokens > self.sliding_window:
            max_allowed = max(0, self.sliding_window - seq_len)
            return True, max_allowed

        return False, num_draft_tokens

    def update_swa_cache_metadata(
        self,
        kv_cache_spec: KVCacheSpec,
        accepted_token_counts: list[int],
    ) -> None:
        """Update SWA-specific cache metadata after verification."""
        # Track which tokens are within vs outside the sliding window
        # This affects cache eviction policy
        pass
```

---

## 4. 测试策略

### 4.1 Adaptive Spec 测试

```python
# tests/v1/spec_decode/test_adaptive.py

import pytest
import torch

from vllm.config import AdaptiveSpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.adaptive_controller import AdaptiveSpeculativeController


class TestAdaptiveSpeculativeController:
    """Tests for AdaptiveSpeculativeController."""

    def test_ema_tracking(self):
        """Test EMA acceptance rate tracking."""
        config = VllmConfig(
            speculative_config=SpeculativeConfig(
                num_speculative_tokens=5,
                adaptive_config=AdaptiveSpeculativeConfig(
                    enabled=True,
                    ema_alpha=0.2,
                ),
            ),
        )

        # Mock proposer
        class MockProposer:
            num_speculative_tokens = 5

        controller = AdaptiveSpeculativeController(config, MockProposer())

        # Simulate acceptance lengths
        for _ in range(10):
            controller.update([3, 4, 3, 4, 3])  # Average ~3.4

        # EMA should converge towards 3.4
        assert 3.0 < controller.ema_accept_len < 4.0

    def test_step_adjustment(self):
        """Test dynamic step adjustment."""
        config = AdaptiveSpeculativeConfig(
            enabled=True,
            candidate_steps=[1, 3, 5, 7],
            ema_alpha=0.5,
            update_interval=1,
            warmup_batches=0,
        )

        controller = AdaptiveSpeculativeController(config, MockProposer())

        # High acceptance should increase steps
        changed = False
        for _ in range(5):
            if controller.update([7, 7, 7, 7, 7]):  # Max acceptance
                changed = True

        assert changed
        assert controller.current_steps > 3

    def test_hysteresis(self):
        """Test hysteresis prevents oscillation."""
        config = AdaptiveSpeculativeConfig(
            enabled=True,
            candidate_steps=[3, 5],
            down_hysteresis=-0.25,
            up_hysteresis=0.0,
        )

        controller = AdaptiveSpeculativeController(config, MockProposer())
        controller.current_steps = 5
        controller.ema_accept_len = 4.0

        # With hysteresis, steps shouldn't change at boundary
        changed = controller._recompute_steps()

        # ema=4.0, threshold for dropping from 5 to 3 is:
        # prev_step - 0.5 + down_hysteresis = 3 - 0.5 - 0.25 = 2.25
        # 4.0 > 2.25, so should stay at 5
        assert not changed
```

---

## 5. 性能基准

### 5.1 Adaptive Spec 预期收益

基于 SGLang 的经验数据，Adaptive Spec 预计在以下场景有显著收益：

| 工作负载特征 | 固定 Steps | Adaptive Steps | 收益 |
|------------|-----------|----------------|------|
| 动态复杂度 (代码/推理) | 5 | 1-7 动态 | +15-25% |
| 简单生成 (摘要) | 7 | 稳定在 7 | 无损失 |
| 复杂生成 (数学) | 7 | 快速降至 1-3 | +20-30% |
| 混合负载 | 5 | 自适应调整 | +10-20% |

### 5.2 关键指标

- **接受率稳定性**：EMA 平滑后震荡 < 10%
- **步数切换开销**：< 1ms ( amortized )
- **内存开销**：每候选步数 +5-10% (可接受)

---

## 6. 附录：参考实现

### 6.1 SGLang 关键代码链接

- `adaptive_spec_params.py`: `python/sglang/srt/speculative/adaptive_spec_params.py`
- `adaptive_runtime_state.py`: `python/sglang/srt/speculative/adaptive_runtime_state.py`
- `eagle_worker_v2.py`: `python/sglang/srt/speculative/eagle_worker_v2.py`
- `eagle_utils.py`: `python/sglang/srt/speculative/eagle_utils.py`

### 6.2 vLLM 相关代码

- `llm_base_proposer.py`: `vllm/v1/spec_decode/llm_base_proposer.py`
- `speculative_config`: `vllm/config/speculative.py`
- `rejection_sampler`: `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`
