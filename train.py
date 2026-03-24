import math

import matplotlib

matplotlib.use("Agg")

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "make_width",
    lambda in_d, out_d, n_h, h_w: [int(in_d)] + [int(h_w)] * int(n_h) + [int(out_d)],
    replace=True,
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    trainer = instantiate(cfg.trainer, cfg=cfg, _recursive_=False)
    results = trainer.train()
    final_loss = float(results["test_loss"][-1])
    return final_loss if math.isfinite(final_loss) else 1e10


if __name__ == "__main__":
    main()
