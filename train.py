import matplotlib

matplotlib.use("Agg")

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    from src.training.trainer import train

    results = train(cfg)
    return float(results["test_loss"][-1])


if __name__ == "__main__":
    main()
