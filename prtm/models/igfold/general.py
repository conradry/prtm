def exists(x):
    return x is not None


def default(
    val,
    d,
):
    return val if exists(val) else d
