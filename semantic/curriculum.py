"""Curriculum schedules shared by RGB/semantic joint training.

The warm-up is deliberately a hard RGB-only phase.  Semantic supervision and
importance-aware capacity are then introduced with a smooth cosine ramp so a
noisy early SAM/CLIP target cannot immediately move scene geometry.
"""

import math


def cosine_ramp(step, start, length):
    """Return a stable 0..1 curriculum weight for ``step``.

    ``length=0`` is useful for ablations and preserves the old hard-switch
    behavior.  The function is dependency-free so it can also be used by
    configuration tools and unit tests.
    """
    step = int(step)
    start = int(start)
    length = int(length)
    if step < start:
        return 0.0
    if length <= 0 or step >= start + length:
        return 1.0
    progress = (step - start) / float(length)
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def curriculum_phase(step, start, length):
    if int(step) < int(start):
        return "rgb_warmup"
    if int(step) < int(start) + max(0, int(length)):
        return "semantic_ramp"
    return "joint_refine"

