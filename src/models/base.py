from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class BaseKANModel(ABC):
    model: torch.nn.Module | None = None
    device: str = "cpu"
    reports_rmse: bool = False

    @abstractmethod
    def build(self, device: str = "cpu") -> None:
        """Construct the underlying model from config."""

    def regularization_loss(self) -> float:
        return 0.0

    def predict(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        return self.model(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self) -> torch.nn.Module:
        return self.model

    def fit(
        self,
        dataset,
        epochs,
        optimizer_factory,
        loss_fn,
        batch_size,
        lamb,
        task_type="regression",
        **kwargs,
    ):
        # Move data to device once, then build DataLoaders from on-device tensors
        # to avoid repeated CPU→GPU transfers every iteration.
        train_input = dataset["train_input"].to(self.device)
        train_label = dataset["train_label"].to(self.device)
        test_input = dataset["test_input"].to(self.device)
        test_label = dataset["test_label"].to(self.device)

        train_ds = TensorDataset(train_input, train_label)
        val_ds = TensorDataset(test_input, test_label)

        n_train = len(train_ds)
        bs = n_train if (batch_size == -1 or batch_size >= n_train) else batch_size

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

        opt = optimizer_factory(self.get_model().parameters())
        is_lbfgs = isinstance(opt, torch.optim.LBFGS)

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []

        device = self.device
        for epoch in tqdm(range(epochs), desc="Training"):
            # --- Train (accumulate metrics during the step to skip a redundant eval pass) ---
            self.get_model().train()
            train_loss_sum = torch.zeros((), device=device)
            train_correct_sum = torch.zeros((), device=device)
            train_total = 0

            for x, y in train_loader:
                if is_lbfgs:
                    captured = {}

                    def closure():
                        opt.zero_grad(set_to_none=True)
                        pred = self.predict(x)
                        data_loss = loss_fn(pred, y)
                        captured["pred"] = pred.detach()
                        captured["data_loss"] = data_loss.detach()
                        reg = self.regularization_loss()
                        loss = data_loss + lamb * reg if reg > 0 else data_loss
                        loss.backward()
                        return loss

                    opt.step(closure)
                    pred_detached = captured["pred"]
                    data_loss = captured["data_loss"]
                else:
                    opt.zero_grad(set_to_none=True)
                    pred = self.predict(x)
                    data_loss = loss_fn(pred, y)
                    reg = self.regularization_loss()
                    loss = data_loss + lamb * reg if reg > 0 else data_loss
                    loss.backward()
                    opt.step()
                    pred_detached = pred.detach()
                    data_loss = data_loss.detach()

                bs_x = x.shape[0]
                train_loss_sum = train_loss_sum + data_loss * bs_x
                if task_type == "classification":
                    train_correct_sum = train_correct_sum + (pred_detached.argmax(dim=1) == y).sum()
                train_total += bs_x

            train_mse = (train_loss_sum / train_total).item()

            # --- Validate (on val set only) ---
            self.get_model().eval()
            val_loss_sum = torch.zeros((), device=device)
            val_correct_sum = torch.zeros((), device=device)
            val_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = self.predict(x)
                    bs_x = x.shape[0]
                    val_loss_sum = val_loss_sum + loss_fn(pred, y) * bs_x
                    if task_type == "classification":
                        val_correct_sum = val_correct_sum + (pred.argmax(dim=1) == y).sum()
                    val_total += bs_x
            val_mse = (val_loss_sum / val_total).item()

            results["train_loss"].append(train_mse)
            results["test_loss"].append(val_mse)
            results["reg"].append(0.0)

            if task_type == "classification":
                results["train_acc"].append((train_correct_sum / train_total).item())
                results["test_acc"].append((val_correct_sum / val_total).item())

        return results
