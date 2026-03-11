import matplotlib

matplotlib.use("Agg")

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    trainer = instantiate(cfg.trainer, cfg=cfg, _recursive_=False)
    results = trainer.train()
    return float(results["test_loss"][-1])


if __name__ == "__main__":
    main()
