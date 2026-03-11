from kan.MLP import MLP
from .base import BaseKANModel


class MLPModel(BaseKANModel):
    def __init__(self, width, seed=1):
        self.width = width
        self.seed = seed
        self.model = None

    def build(self, device="cpu"):
        self.model = MLP(
            width=list(self.width),
            seed=self.seed,
            device=device,
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
