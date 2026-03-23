import random
import time
from datetime import datetime

import mlflow
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.training.get("device", "cpu")

    def train(self):
        cfg = self.cfg

        # load dataset
        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create(device=self.device)

        # build model (width resolved from config via make_width resolver)
        model = instantiate(cfg.model)
        model.build(device=self.device)

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment_name", "kan-lab"))

        with mlflow.start_run(run_name=_generate_run_name(cfg)):
            # log config parameters
            flat_cfg = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
            mlflow.log_params(flat_cfg)
            mlflow.log_param("parameter_count", model.parameter_count())
            mlflow.log_param("dataset", _get_dataset_name(cfg))
            mlflow.log_param("model", _get_model_name(cfg))
            mlflow.log_param("shape", _get_shape(cfg))

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

        return results


_ADJECTIVES = [
    "swift",
    "bright",
    "calm",
    "bold",
    "keen",
    "warm",
    "cool",
    "fair",
    "wild",
    "deep",
    "glad",
    "pure",
    "vast",
    "free",
    "wise",
    "rare",
]
_NOUNS = [
    "fox",
    "owl",
    "elk",
    "jay",
    "ram",
    "bee",
    "ant",
    "yak",
    "emu",
    "cod",
    "hen",
    "ape",
    "bat",
    "cat",
    "dog",
    "hawk",
]


_sysrand = random.SystemRandom()


def _generate_run_name(cfg):
    adjective = _sysrand.choice(_ADJECTIVES)
    noun = _sysrand.choice(_NOUNS)
    number = _sysrand.randint(100, 999)
    return f"{adjective}-{noun}-{number}"


def _get_model_name(cfg):
    model_class = cfg.model.get("_target_", "unknown")
    return model_class.rsplit(".", 1)[-1].replace("Model", "")


def _get_dataset_name(cfg):
    dataset_class = cfg.dataset.get("_target_", "unknown")
    return dataset_class.rsplit(".", 1)[-1].replace("Dataset", "")


def _get_shape(cfg):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    return str(model_cfg.get("width") or model_cfg.get("layers_hidden", "unknown"))


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
