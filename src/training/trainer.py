import time

import mlflow
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.training.get("device", "cpu")

    def train(self):
        cfg = self.cfg

        # build model
        model = instantiate(cfg.model)
        model.build(device=self.device)

        # load dataset
        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create(device=self.device)

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment_name", "kan-lab"))

        with mlflow.start_run():
            # log config parameters
            flat_cfg = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)
            mlflow.log_param("parameter_count", model.parameter_count())

            # train
            t_start = time.time()
            results = model.fit(
                dataset=dataset,
                steps=cfg.training.steps,
                lr=cfg.training.lr,
                optimizer=cfg.training.optimizer,
                loss_fn=None,
                batch_size=cfg.training.get("batch_size", -1),
                lamb=cfg.training.get("lamb", 0.0),
            )
            train_time = time.time() - t_start

            # log metrics per step
            for step_i, (tl, vl) in enumerate(
                zip(results["train_loss"], results["test_loss"])
            ):
                mlflow.log_metrics(
                    {"train_rmse": float(tl), "test_rmse": float(vl)},
                    step=step_i,
                )

            # log final metrics
            mlflow.log_metric("final_train_rmse", float(results["train_loss"][-1]))
            mlflow.log_metric("final_test_rmse", float(results["test_loss"][-1]))
            mlflow.log_metric("training_time_sec", train_time)

            # save model
            torch.save(model.get_model().state_dict(), "model.pt")
            mlflow.log_artifact("model.pt")

        return results


def _flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)
