from lib.utils import init_train_device, init_datasets_and_models, init_runtime_contexts
from lib.component_builder import build_component_from_config

from lib import Context
from lib.training_components.loops import BenchmarkingLoopFactory

import torch
import os
from aim import Run

def generate_sample(model, dataset, device, prompt, n_words=15, max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    with torch.no_grad():
        ids0 = dataset.tok.Encode(prompt)
        ids = torch.tensor(ids0, dtype=torch.long, device=device).unsqueeze(0)
        for i in range(max_new_tokens):
            logits = model(ids)[:, -1, :].float()
            if dataset.pad_id < logits.size(-1):
                logits[:, dataset.pad_id] = float("-inf")
            logits = logits / max(temperature, 1e-8)
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            if dataset.eos_id is not None and dataset.eos_id < logits.size(-1) and int(
                    next_id.item()) == dataset.eos_id:
                break
            ids = torch.cat([ids, next_id], dim=1)
            if len(dataset.tok.Decode(ids[0].tolist()).split()) >= n_words:
                break
    model.train()
    return " ".join(dataset.tok.Decode(ids[0].tolist()).split()[:n_words])


def main():
    # Initialise models
    device, autocast_context = init_train_device()
    context = Context(autocast_ctx=autocast_context, device=device)

    runtime_context, runtime_configs = init_runtime_contexts()
    context.merge(runtime_context)
    torch.manual_seed(context.require("seed"))

    context, data_and_model_configs = init_datasets_and_models(context, shuffle=False)

    configs = runtime_configs | data_and_model_configs
    context.merge({"config_dicts": configs})
    save_dir = os.path.join("data/", "checkpoints")
    
    try:
        model_data = torch.load(os.path.join(save_dir, "ckpt_final.pt"), map_location=torch.device('cpu'))
    except Exception:
        try:
            model_data = torch.load(os.path.join(save_dir, "ckpt_best_val.pt"), map_location=torch.device('cpu'))
        except Exception:
            model_data = torch.load(os.path.join(save_dir, "ckpt_best_train.pt"), map_location=torch.device('cpu'))

    context.model.load_state_dict(model_data["model"])
    
    # Generate example outputs
    
    examples = []
    for opener in ["the dog ", 
                   "sam went ", 
                   "caitlin stood on the ", 
                   "jayden had a jolly good time ",
                   "a spaceship landed in the forest ", 
                   "trees swayed ", 
                   "What do you think about the existence of God? I think", 
                   "Sol de Janeiro is an American skincare and fragrance brand founded in 2015 by", 
                   "For nineteen years she performed domestic duties; cooking meals, ",
                   "In Japanese culture, women are often",
                   "Dragons smell like cheese because "]:
        examples.append(
            generate_sample(context.model, context.train_dataloader.dataset, device, opener, n_words=100, max_new_tokens=100, temperature=0.0))
            
    # Validate Model (BenchmarkingLoopFactory: same steps as training, no Aim run)
    print("Starting validation")
    benchmarking_loop, training_config = build_component_from_config(
        BenchmarkingLoopFactory,
        "configs/training.yaml",
        context.fork(accumulation_steps=max(context.accumulated_batch_size // context.batch_size, 1)),
    )
    validation_step = benchmarking_loop.validation_step
    validation_step.num_batches = None
    validation_error = validation_step.step()
    print(f"Validation completed. Error: {float(validation_error)}")

    with open(os.path.join(save_dir, "aim_run_hash.txt"), "r") as f:
        run_hash = f.read().strip()
    run = Run(run_hash=run_hash)
    run["posttraining_examples"] = examples
    run["posttraining_validation_loss"] = float(validation_error)
    run.close()


if __name__ == "__main__":
    main()
