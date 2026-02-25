import os
import torch
from lib.run_config import load_run_config
from lib.data_config import build_dataset_and_loader
from lib.model_builder import build_model_from_config
from lib.utils import generate_sample

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
save_dir = "configs/checkpoints"

run_config = load_run_config()
data_config = run_config.load_data()

dataset, _, vocab_size, pad_id = build_dataset_and_loader(
    data_config,
    batch_size=1,
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

examples = []
for opener in [
    "the dog ",
    "sam went ",
    "caitlin pooped on the ",
    "jayden had a jolly good time ",
    "a spaceship landed in the forest ",
    "trees swayed ",
]:
    examples.append(
        generate_sample(model, dataset, device, opener, n_words=100, max_new_tokens=100, temperature=0.0)
    )

with open("configs/examples.txt", "w") as f:
    for ex in examples:
        f.write(f"{ex}\n\n")
