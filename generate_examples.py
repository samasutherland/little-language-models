import torch
import tomllib
from lib.helpers import generate_sample, pad_collate_fn, load_configs, get_data_loader, create_model
import os


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
save_dir = "experiment/checkpoints"

model_cfg, data_cfg, train_cfg = load_configs()
try:
    model_data = torch.load(os.path.join(save_dir, "ckpt_final.pt"), map_location=torch.device('cpu'))
except Exception:
    model_data = torch.load(os.path.join(save_dir, "ckpt_best_val.pt"), map_location=torch.device('cpu'))
dataset, loader, vocab_size = get_data_loader(data_cfg, train_cfg)
model = create_model(model_cfg, vocab_size, device, dataset)
model.load_state_dict(model_data["model"])

examples = []
for opener in ["the dog ", "sam went ", "caitlin pooped on the ", "jayden had a jolly good time ", "a spaceship landed in the forest ", "trees swayed "]:
    examples.append(generate_sample(model, dataset, device, opener, n_words=100, max_new_tokens=100, temperature=0.0))

with open("experiment/examples.txt", "w") as f:
    for example in examples:
        f.write(f"{example}\n\n")

