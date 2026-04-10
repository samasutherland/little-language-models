from lib.utils import init_train_device, init_datasets_and_models, init_runtime_contexts
from lib.component_builder import build_component_from_config

from lib import Context
from lib.training_components.loops import TrainingLoopFactory

import torch
import os
torch.manual_seed(42)


def main():
    device, autocast_context = init_train_device()
    context = Context(autocast_ctx=autocast_context, device=device)

    runtime_context, runtime_configs = init_runtime_contexts()
    context.merge(runtime_context)

    context, data_and_model_configs = init_datasets_and_models(context, shuffle=True)
    
    configs = runtime_configs | data_and_model_configs
    context.merge({"config_dicts": configs})

    training_loop, training_config = build_component_from_config(TrainingLoopFactory,
                                "../configs/training.yaml", context.fork(accumulation_steps=max(context.accumulated_batch_size//context.batch_size, 1)))
    
    token_count, loss, val_loss, best_train_loss, best_val_loss, total_descent_steps = training_loop.run()
    run = training_loop.aim_logger
    
    print("Saving final weights...")
    torch.save({"model": context.model.state_dict(), "optimizer": training_loop.gradient_step.optimizer.state_dict(),
                "step": total_descent_steps}, os.path.join(training_loop.train_checkpointer.save_dir, "ckpt_final.pt"))
    with open(os.path.join(training_loop.train_checkpointer.save_dir, "aim_run_hash.txt"), "w") as f:
        f.write(run.hash)
    print("storing token_count and tokens per parameter...")
    run["git_sha"] = os.environ.get("GIT_SHA", "")
    run["image_tag"] = os.environ.get("IMAGE_TAG", "")
    run["pod_name"] = os.environ.get("POD_NAME", "")
    run["runpod_pod_id"] = os.environ.get("RUNPOD_POD_ID", "")
    run["token_count"] = token_count
    total_params = sum(p.numel() for p in context.model.parameters() if p.requires_grad)
    run["tokens_per_parameter"] = token_count / total_params
    run["best_train_loss"] = best_train_loss
    run["best_val_loss"] = best_val_loss

    run.close()


if __name__ == "__main__":
    main()
