"""Pure-Python sparse-view selection helpers."""


def select_uniform(items, max_views=-1, stride=1):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    selected = list(items)[::stride]
    if max_views is None or max_views < 0 or len(selected) <= max_views:
        return selected
    if max_views < 2:
        raise ValueError("max_views must be -1 or >= 2")
    last = len(selected) - 1
    indices = [round(i * last / (max_views - 1)) for i in range(max_views)]
    return [selected[index] for index in indices]
