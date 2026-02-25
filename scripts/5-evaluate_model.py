import os
import time
import torch
from lib.run_config import load_run_config
from lib.data_config import build_dataset_and_loader
from lib.model_builder import build_model_from_config
from lib.training_config import build_criterion

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
save_dir = "configs/checkpoints"
out_dir = "configs"

run_config = load_run_config()
data_config = run_config.load_data()
train_config = run_config.load_training()

dataset, loader, vocab_size, pad_id = build_dataset_and_loader(
    data_config,
    batch_size=train_config.batch_size,
    split="test",
    shuffle=False,
    loop=False,
)

ctx = run_config.get_build_context(vocab_size, pad_id)
model = build_model_from_config(run_config.model_config_path, ctx=ctx)
model = model.to(device=device)

try:
    model_data = torch.load(os.path.join(save_dir, "ckpt_final.pt"), map_location=torch.device("cpu"))
except Exception:
    model_data = torch.load(os.path.join(save_dir, "ckpt_best_val.pt"), map_location=torch.device("cpu"))
model.load_state_dict(model_data["model"])

criterion = build_criterion(train_config, ignore_index=pad_id)

start = time.time()
losses = []
model.eval()
with torch.no_grad():
    for batch in loader:
        x = batch["input_ids"].to(device, non_blocking=True)
        logits = model(x[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        losses.append(loss.item())
end = time.time()
print(f"evaluation took {end - start} seconds")
with open(os.path.join(out_dir, "test_acc.txt"), "w") as f:
    f.write(f"{torch.tensor(losses).mean().item()}")
