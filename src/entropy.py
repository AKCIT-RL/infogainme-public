"""Entropy utilities for information theory calculations.

Implements Shannon entropy computation and information gain calculations
for measuring uncertainty reduction in the candidate pool.
"""

from __future__ import annotations

import math


class Entropy:
    """Shannon entropy calculator for candidate pool.

    Uses uniform distribution assumption: each active candidate has equal probability
    of being the target, providing a simple yet effective uncertainty measure.
    """

    @staticmethod
    def compute(n: int) -> float:
        """Compute Shannon entropy for a number of active candidates.

        Args:
            n: Number of active candidates.

        Returns:
            Shannon entropy in bits. Returns 0.0 for 0 or 1 candidates.

        Note:
            Uses uniform distribution: H = log2(N) where N = n.
            This assumes each candidate has equal probability of being the target.
        """
        if n <= 1:
            return 0.0
        return math.log2(n)

    @staticmethod
    def info_gain(h_before: float, h_after: float) -> float:
        """Calculate information gain from entropy reduction.

        Args:
            h_before: Entropy before an operation (e.g., before pruning).
            h_after: Entropy after an operation (e.g., after pruning).

        Returns:
            Information gain in bits. Always non-negative due to max(0, ...).

        Note:
            Information gain = H(before) - H(after). The max() ensures we never
            report negative gains due to floating-point precision issues.
        """
        return max(0.0, h_before - h_after)


if __name__ == "__main__":
    # Self-tests

    def _test_entropy_edge_cases() -> None:
        assert Entropy.compute(0) == 0.0
        assert Entropy.compute(1) == 0.0

    def _test_entropy_computation() -> None:
        assert abs(Entropy.compute(2) - 1.0) < 1e-10
        assert abs(Entropy.compute(4) - 2.0) < 1e-10
        assert abs(Entropy.compute(8) - 3.0) < 1e-10

    def _test_info_gain() -> None:
        assert Entropy.info_gain(3.0, 0.0) == 3.0
        assert Entropy.info_gain(4.0, 2.0) == 2.0
        assert Entropy.info_gain(2.0, 2.0) == 0.0
        assert Entropy.info_gain(1.0, 1.1) == 0.0

    def _test_realistic_scenario() -> None:
        h_initial = Entropy.compute(16)
        assert abs(h_initial - 4.0) < 1e-10
        h_after = Entropy.compute(4)
        assert abs(h_after - 2.0) < 1e-10
        gain = Entropy.info_gain(h_initial, h_after)
        assert abs(gain - 2.0) < 1e-10

    _test_entropy_edge_cases()
    _test_entropy_computation()
    _test_info_gain()
    _test_realistic_scenario()
    print("Entropy self-tests: OK")
