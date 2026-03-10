DATASET_REGISTRY = {}


def register_dataset(name):
    """Decorator to register a dataset factory function."""

    def wrapper(fn):
        DATASET_REGISTRY[name] = fn
        return fn

    return wrapper


def create_dataset(cfg, device="cpu"):
    """Create a pykan-format dataset dict from Hydra config."""
    name = cfg.dataset.name
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return DATASET_REGISTRY[name](cfg.dataset, device=device)
