import torch
LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
OPTIMIZER_REGISTRY = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "AdamW": torch.optim.AdamW}
SCHEDULER_REGISTRY = {"OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, "Cosine": torch.optim.lr_scheduler.CosineAnnealingLR}
