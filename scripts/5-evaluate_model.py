import torch
import tomllib
from lib.utils import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model
import os
import time

LOSS_REGISTRY = {"CrossEntropyLoss": torch.nn.CrossEntropyLoss}
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
save_dir = "configs/"

model_cfg, data_cfg, train_cfg = load_configs()
try:
    model_data = torch.load(os.path.join(save_dir, "ckpt_final.pt"), map_location=torch.device('cpu'))
except Exception:
    model_data = torch.load(os.path.join(save_dir, "ckpt_best_val.pt"), map_location=torch.device('cpu'))
dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg)
model = create_model(model_cfg, vocab_size, device, dataset)
model.load_state_dict(model_data["model"])

dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg, split="test")

criterion = LOSS_REGISTRY[train_cfg["loss"]](ignore_index=dataset.pad_id)

start = time.time()
losses = []
model.eval()
with torch.no_grad():
    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(device, non_blocking=True)
        logits = model(x[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        losses.append(loss.item())

end = time.time()
print(f"evaluation took {end-start} seconds")
with open(os.path.join(save_dir, "test_acc.txt"), "w") as f:
    f.write(f"{torch.tensor(losses).mean().item()}")