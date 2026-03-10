from .pykan_model import PyKANModel
from .mlp_model import MLPModel
from .efficient_kan_model import EfficientKANModel

MODEL_REGISTRY = {
    "pykan": PyKANModel,
    "mlp": MLPModel,
    "efficient_kan": EfficientKANModel,
}


def create_model(cfg):
    model_type = cfg.model.type
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    return MODEL_REGISTRY[model_type](cfg.model)
