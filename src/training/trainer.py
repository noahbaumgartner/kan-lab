import random
import time
from datetime import datetime

import torch
import mlflow
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.training.get("device", "cpu")

    def _create_loss_fn(self, task_type):
        if task_type == "classification":
            return torch.nn.CrossEntropyLoss()
        return lambda pred, target: torch.mean((pred - target) ** 2)

    def _run_training(
        self, model, dataset, loss_fn, task_type, steps, lr, optimizer, batch_size, lamb
    ):
        if optimizer == "Adam":
            opt = torch.optim.AdamW(model.get_model().parameters(), lr=lr)
        elif optimizer == "LBFGS":
            opt = torch.optim.LBFGS(model.get_model().parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []

        pbar = tqdm(range(steps), desc="Training")
        for _ in pbar:
            n_train = dataset["train_input"].shape[0]
            if batch_size == -1 or batch_size >= n_train:
                x = dataset["train_input"]
                y = dataset["train_label"]
            else:
                idx = torch.randperm(n_train, device=model.device)[:batch_size]
                x = dataset["train_input"][idx]
                y = dataset["train_label"][idx]

            if optimizer == "LBFGS":

                def closure():
                    opt.zero_grad()
                    pred = model.predict(x)
                    loss = loss_fn(pred, y)
                    reg = model.regularization_loss()
                    if reg > 0:
                        loss = loss + lamb * reg
                    loss.backward()
                    return loss

                opt.step(closure)
            else:
                opt.zero_grad()
                pred = model.predict(x)
                loss = loss_fn(pred, y)
                reg = model.regularization_loss()
                if reg > 0:
                    loss = loss + lamb * reg
                loss.backward()
                opt.step()

            with torch.no_grad():
                train_pred = model.predict(dataset["train_input"])
                train_loss_val = loss_fn(train_pred, dataset["train_label"]).item()
                test_pred = model.predict(dataset["test_input"])
                test_loss_val = loss_fn(test_pred, dataset["test_label"]).item()

            results["train_loss"].append(train_loss_val)
            results["test_loss"].append(test_loss_val)
            results["reg"].append(0.0)

            if task_type == "classification":
                with torch.no_grad():
                    train_acc = (
                        (train_pred.argmax(dim=1) == dataset["train_label"])
                        .float()
                        .mean()
                        .item()
                    )
                    test_acc = (
                        (test_pred.argmax(dim=1) == dataset["test_label"])
                        .float()
                        .mean()
                        .item()
                    )
                results["train_acc"].append(train_acc)
                results["test_acc"].append(test_acc)
                pbar.set_postfix(loss=f"{test_loss_val:.4f}", acc=f"{test_acc:.4f}")
            else:
                pbar.set_postfix(rmse=f"{test_loss_val ** 0.5:.4f}")

        return results

    def train(self):
        cfg = self.cfg
        task_type = cfg.dataset.get("task_type", "regression")

        # load dataset
        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create(device=self.device)

        # build model
        model = instantiate(cfg.model)
        model.build(device=self.device)

        # create loss function
        loss_fn = self._create_loss_fn(task_type)

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment_name", "experiment"))

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

            if hasattr(model, "fit"):
                # PyKAN has its own training loop
                results = model.fit(
                    dataset=dataset,
                    steps=cfg.training.steps,
                    lr=cfg.training.lr,
                    optimizer=cfg.training.optimizer,
                    loss_fn=loss_fn,
                    batch_size=cfg.training.get("batch_size", -1),
                    lamb=cfg.training.get("lamb", 0.0),
                    task_type=task_type,
                )
            else:
                results = self._run_training(
                    model=model,
                    dataset=dataset,
                    loss_fn=loss_fn,
                    task_type=task_type,
                    steps=cfg.training.steps,
                    lr=cfg.training.lr,
                    optimizer=cfg.training.optimizer,
                    batch_size=cfg.training.get("batch_size", -1),
                    lamb=cfg.training.get("lamb", 0.0),
                )

            train_time = time.time() - t_start

            # log metrics per step
            if task_type == "classification":
                for step_i, (tl, vl, ta, va) in enumerate(
                    zip(
                        results["train_loss"],
                        results["test_loss"],
                        results["train_acc"],
                        results["test_acc"],
                    )
                ):
                    mlflow.log_metrics(
                        {
                            "train_loss": float(tl),
                            "test_loss": float(vl),
                            "train_acc": float(ta),
                            "test_acc": float(va),
                        },
                        step=step_i,
                    )

                mlflow.log_metric("final_train_loss", float(results["train_loss"][-1]))
                mlflow.log_metric("final_test_loss", float(results["test_loss"][-1]))
                mlflow.log_metric("final_train_acc", float(results["train_acc"][-1]))
                mlflow.log_metric("final_test_acc", float(results["test_acc"][-1]))
            else:
                for step_i, (tl, vl) in enumerate(
                    zip(results["train_loss"], results["test_loss"])
                ):
                    train_rmse = (
                        float(tl) ** 0.5 if not hasattr(model, "fit") else float(tl)
                    )
                    test_rmse = (
                        float(vl) ** 0.5 if not hasattr(model, "fit") else float(vl)
                    )
                    mlflow.log_metrics(
                        {"train_rmse": train_rmse, "test_rmse": test_rmse},
                        step=step_i,
                    )

                final_train = float(results["train_loss"][-1])
                final_test = float(results["test_loss"][-1])
                if not hasattr(model, "fit"):
                    final_train = final_train**0.5
                    final_test = final_test**0.5
                mlflow.log_metric("final_train_rmse", final_train)
                mlflow.log_metric("final_test_rmse", final_test)

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
