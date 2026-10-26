"""Self-supervised methods (SimCLR default, DINO optional) and GPU tensor augmentations.

Named `ssl_methods` (not `ssl`) so it can never shadow Python's stdlib `ssl` module.
"""
try:
    import ssl_methods.byol  # noqa: F401
except ImportError:
    pass
try:
    import ssl_methods.swav  # noqa: F401
except ImportError:
    pass
try:
    import ssl_methods.looc_swav  # noqa: F401
except ImportError:
    pass
try:
    import ssl_methods.looc_byol  # noqa: F401
except ImportError:
    pass

