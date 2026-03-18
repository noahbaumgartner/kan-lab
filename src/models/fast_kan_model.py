from .base import BaseKANModel
from ...modules.fastkan.fastkan import FastKAN


class FastKANModel(BaseKANModel):
    def __init__(self):
        return

    def build(self, device="cpu"):
        self.model = Eff

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
