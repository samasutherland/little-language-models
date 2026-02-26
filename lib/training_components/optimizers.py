
import torch

from lib import Context, Factory

class AdamWFactory(Factory[torch.optim.Optimizer]):
    type: Literal["adamw"] = "adamw"
    weight_decay: float = 0.02
    betas: tuple[float, float] = (0.9, 0.95)
    lr_scale: float = 1.0  # scale applied to base_lr when building (e.g. 1/3)

    def build(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        effective_lr = lr * self.lr_scale
        return AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=effective_lr,
            betas=list(self.betas),
        )
#
#
# class AdamFactory(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     type: Literal["adam"] = "adam"
#     weight_decay: float = 0.0
#     betas: tuple[float, float] = (0.9, 0.999)
#
#     def build(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
#         return Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay, betas=list(self.betas))
#
#
# class SGDFactory(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     type: Literal["sgd"] = "sgd"
#     momentum: float = 0.0
#     weight_decay: float = 0.0
#
#     def build(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
#         return SGD(model.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
#
#
# OptimizerFactory = Annotated[
#     Union[AdamWFactory, AdamFactory, SGDFactory],
#     Field(discriminator="type"),
# ]