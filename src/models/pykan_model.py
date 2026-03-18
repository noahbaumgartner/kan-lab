from kan import KAN
from .base import BaseKANModel


class PyKANModel(BaseKANModel):
    def __init__(self, width, grid, k, base_fun="silu", seed=1):
        self.width = width
        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        self.seed = seed
        self.model = None

    def build(self, device="cpu"):
        self.model = KAN(
            width=list(self.width),
            grid=self.grid,
            k=self.k,
            base_fun=self.base_fun,
            seed=self.seed,
            device=device,
            auto_save=False,
        )

    def fit(self, dataset, steps, lr, optimizer, loss_fn, batch_size, lamb, **kwargs):
        results = self.model.fit(
            dataset,
            opt=optimizer,
            steps=steps,
            lr=lr,
            lamb=lamb,
            loss_fn=loss_fn,
            batch=batch_size,
        )
        return results

    def predict(self, x):
        return self.model.forward(x)

    def parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())

    def get_model(self):
        return self.model
