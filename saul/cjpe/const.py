"""Shared constants for Phase 4. Gains in the plan are in points: 1 point = 0.01 macro-F1."""
from __future__ import annotations

PRIOR0 = 0.59
PRIOR1 = 0.41
POINT = 0.01
ADOPT_GAIN = 0.7 * POINT
THRESH_GAIN = 0.3 * POINT
STACK_GAIN = 0.5 * POINT
MIX_ADD = 0.5 * POINT
MEMBER_MARGIN = 0.06
A1_GATE = 0.74
G1_FAST = 30_000.0
G1_SLOW = 15_000.0

BOS, PAD, SEP, CLS = 0, 2, 4, 5
BUCKETS = (1024, 2048, 4096, 8192)
EXPECTED_OVL = 1517
EXPECTED_SEL = 1988
SEED = 781
