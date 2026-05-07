import random
import time

import numpy as np
import torch
import mlflow
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src import get_device
from src.datasets.gaussian_blob import GaussianBlobDataset


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        device_cfg = cfg.training.get("device", "auto")
        self.device = get_device() if device_cfg == "auto" else torch.device(device_cfg)
        print(f"Using device: {self.device}")
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True

    def _create_loss_fn(self, task_type):
        if task_type == "classification":
            return torch.nn.CrossEntropyLoss()
        return lambda pred, target: torch.mean((pred - target) ** 2)

    @staticmethod
    def _seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self):
        cfg = self.cfg
        task_type = cfg.dataset.get("task_type", "regression")

        # global seed
        self._seed_everything(cfg.get("seed", 42))

        # load dataset
        dataset_obj = instantiate(cfg.dataset)
        dataset = dataset_obj.create()

        # build model
        model = instantiate(cfg.model)
        model.build(device=self.device)

        # optimizer factory: call with model.parameters() to build the torch optimizer
        optimizer_factory = instantiate(cfg.optimizer)

        # create loss function
        loss_fn = self._create_loss_fn(task_type)

        # setup MLflow
        mlflow.set_tracking_uri(cfg.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(cfg.get("experiment", "experiment"))
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)

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

            fit_kwargs = dict(
                dataset=dataset,
                epochs=cfg.training.epochs,
                optimizer_factory=optimizer_factory,
                loss_fn=loss_fn,
                batch_size=cfg.training.get("batch_size", -1),
                lamb=cfg.training.get("lamb", 0.0),
                task_type=task_type,
            )

            if isinstance(dataset_obj, GaussianBlobDataset):
                fit_kwargs["epoch_callback"] = _make_blob_visualization_callback(
                    dataset_obj, dataset, every=10
                )

            results = model.fit(**fit_kwargs)

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
                    mlflow.log_metrics(
                        {
                            "train_rmse": float(tl) ** 0.5,
                            "test_rmse": float(vl) ** 0.5,
                        },
                        step=step_i,
                    )

                final_train_mse = float(results["train_loss"][-1])
                final_test_mse = float(results["test_loss"][-1])
                final_train_rmse = final_train_mse**0.5
                final_test_rmse = final_test_mse**0.5
                mlflow.log_metric("final_train_rmse", final_train_rmse)
                mlflow.log_metric("final_test_rmse", final_test_rmse)

                train_var = float(torch.var(dataset["train_label"], unbiased=False))
                test_var = float(torch.var(dataset["test_label"], unbiased=False))
                final_train_r2 = 1.0 - final_train_mse / train_var if train_var > 0 else float("nan")
                final_test_r2 = 1.0 - final_test_mse / test_var if test_var > 0 else float("nan")
                mlflow.log_metric("final_train_r2", final_train_r2)
                mlflow.log_metric("final_test_r2", final_test_r2)

                print(f"\nFinal Train RMSE: {final_train_rmse:.6f}")
                print(f"Final Test RMSE:  {final_test_rmse:.6f}")
                print(f"Final Train R²:   {final_train_r2:.6f}")
                print(f"Final Test R²:    {final_test_r2:.6f}")

            mlflow.log_metric("training_time_sec", train_time)

        return results


def _make_blob_visualization_callback(dataset_obj, dataset, every=10):
    image_size = dataset_obj.image_size
    sample_input = dataset["test_input"][:1]
    sample_label = dataset["test_label"][:1]
    gt_image = sample_input.detach().cpu().reshape(image_size, image_size).numpy()

    def callback(epoch, model_wrapper):
        if (epoch + 1) % every != 0:
            return
        was_training = model_wrapper.get_model().training
        model_wrapper.get_model().eval()
        with torch.no_grad():
            pred_label = model_wrapper.predict(sample_input)
            pred_image = GaussianBlobDataset.render_from_labels(
                pred_label, image_size=image_size
            )
        if was_training:
            model_wrapper.get_model().train()

        pred_np = pred_image.detach().cpu().reshape(image_size, image_size).numpy()
        gt_params = sample_label[0].detach().cpu().tolist()
        pred_params = pred_label[0].detach().cpu().tolist()

        vmax = max(gt_image.max(), pred_np.max(), 1e-6)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(gt_image, vmin=0, vmax=vmax, cmap="viridis")
        axes[0].set_title(
            "ground truth\n"
            f"x={gt_params[0]:.2f} y={gt_params[1]:.2f} "
            f"w={gt_params[2]:.2f} a={gt_params[3]:.2f}"
        )
        axes[0].axis("off")
        axes[1].imshow(pred_np, vmin=0, vmax=vmax, cmap="viridis")
        axes[1].set_title(
            "predicted\n"
            f"x={pred_params[0]:.2f} y={pred_params[1]:.2f} "
            f"w={pred_params[2]:.2f} a={pred_params[3]:.2f}"
        )
        axes[1].axis("off")
        fig.suptitle(f"epoch {epoch + 1}")
        fig.tight_layout()
        mlflow.log_figure(fig, f"blob_compare/epoch_{epoch + 1:04d}.png")
        plt.close(fig)

    return callback


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
    base_name = dataset_class.rsplit(".", 1)[-1].replace("Dataset", "")
    if base_name == "Feynman":
        return f"Feynman_{cfg.dataset.get('name', 'unknown')}"
    return base_name


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
