"""Separate checkpoint selection from early-stopping tolerance."""

import math


def validation_decision(score, best, patience_reference, min_delta):
    if not math.isfinite(score):
        return False, False, patience_reference
    selected = score > best
    significant = score > patience_reference + min_delta
    return selected, significant, score if significant else patience_reference
