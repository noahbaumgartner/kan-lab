import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseKANModel
from src.modules.efficientkan import KAN


class EfficientKANModel(BaseKANModel):
    def __init__(self, layers_hidden, grid_size=5, k=3, grid_update_freq=10, **kwargs):
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.k = k
        self.grid_update_freq = grid_update_freq

    def build(self, device="cpu"):
        self.model = KAN(
            layers_hidden=list(self.layers_hidden),
            grid_size=self.grid_size,
            spline_order=self.k,
            grid_range=[-3, 3],
        ).to(device)
        self.device = device

    def predict(self, x, update_grid=False):
        if update_grid and self.device == "mps":
            self.model.cpu()
            self.model(x.cpu(), update_grid=True)
            self.model.to(self.device)
            return self.model(x)
        return self.model(x, update_grid=update_grid)

    def regularization_loss(self):
        return self.model.regularization_loss()

    def fit(
        self,
        dataset,
        epochs,
        lr,
        optimizer,
        loss_fn,
        batch_size,
        lamb,
        task_type="regression",
        **kwargs,
    ):
        # Move data to device once to avoid repeated CPU→GPU transfers
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

        if optimizer in ("Adam", "AdamW"):
            opt_cls = torch.optim.AdamW if optimizer == "AdamW" else torch.optim.Adam
            opt = opt_cls(self.get_model().parameters(), lr=lr)
        elif optimizer == "LBFGS":
            opt = torch.optim.LBFGS(self.get_model().parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        results = {"train_loss": [], "test_loss": [], "reg": []}
        if task_type == "classification":
            results["train_acc"] = []
            results["test_acc"] = []

        for epoch in tqdm(range(epochs), desc="Training"):
            # --- Train ---
            self.get_model().train()
            for i, (x, y) in enumerate(train_loader):

                # periodic grid update
                update_grid = epoch % self.grid_update_freq == 0 and i == 0

                if optimizer == "LBFGS":

                    def closure():
                        opt.zero_grad()
                        pred = self.predict(x, update_grid=update_grid)
                        loss = loss_fn(pred, y)
                        reg = self.regularization_loss()
                        if reg > 0:
                            loss = loss + lamb * reg
                        loss.backward()
                        return loss

                    opt.step(closure)
                else:
                    opt.zero_grad()
                    pred = self.predict(x, update_grid=update_grid)
                    loss = loss_fn(pred, y)
                    reg = self.regularization_loss()
                    if reg > 0:
                        loss = loss + lamb * reg
                    loss.backward()
                    opt.step()

            # --- Validate ---
            self.get_model().eval()
            train_mse = 0.0
            train_total = 0
            train_correct = 0
            with torch.no_grad():
                for x, y in train_loader:
                    pred = self.predict(x)
                    train_mse += loss_fn(pred, y).item() * x.shape[0]
                    if task_type == "classification":
                        train_correct += (pred.argmax(dim=1) == y).sum().item()
                    train_total += x.shape[0]
            train_mse /= train_total

            val_mse = 0.0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    pred = self.predict(x)
                    val_mse += loss_fn(pred, y).item() * x.shape[0]
                    if task_type == "classification":
                        val_correct += (pred.argmax(dim=1) == y).sum().item()
                    val_total += x.shape[0]
            val_mse /= val_total

            results["train_loss"].append(train_mse)
            results["test_loss"].append(val_mse)
            results["reg"].append(self.regularization_loss())

            if task_type == "classification":
                results["train_acc"].append(train_correct / train_total)
                results["test_acc"].append(val_correct / val_total)

        return results
