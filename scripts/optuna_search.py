import copy

import optuna
import yaml
from pathlib import Path

from lib import Context
from lib.component_builder import build_component_from_dict
from lib.training_components.loops import TrainingLoopFactory
from lib.training_components.pretraining import LayerSweepFactory
from lib.utils import init_datasets_and_models, init_train_device

device, autocast_context = init_train_device()


def load_config(path: str | Path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def test_config(trial: optuna.Trial):

    acc_batch_size_exponent = trial.suggest_int("acc_batch_size_exponent", low=4, high=8)
    acc_batch_size = 2 ** acc_batch_size_exponent
    lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
    vocab_size = trial.suggest_int("vocab_size", 2, 12, step=2)
    num_layers = trial.suggest_int("num_layers", 4, 32)

    num_heads = trial.suggest_int("num_heads", 2, 8, step=1)
    qk_dim = trial.suggest_int("qk_dim", 16, 128, step=16)

    ctx = Context(autocast_ctx=autocast_context, device=device)

    ctx_cfg = load_config("configs/context.yaml")
    server_context = load_config("configs/server.yaml")
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    training_cfg = load_config("configs/training.yaml")
    pretraining_cfg = load_config("configs/pretraining.yaml")

    ctx_cfg["accumulated_batch_size"] = acc_batch_size
    ctx_cfg["learning_rate"] = lr
    ctx_cfg["num_layers"] = num_layers
    ctx_cfg["training_time"] = pretraining_cfg["training_time"]
    ctx_cfg["warmup_steps"] = pretraining_cfg["warmup_steps"]

    ctx.merge(ctx_cfg)
    ctx.merge(server_context)
    
    data_cfg["dataset_factory"]["tokenizer_factory"]["tokenizer_path"] = f"data/tokenizers/baby_unigram_{vocab_size}K.model"

    embedding_dim = num_heads * qk_dim
    model_cfg["embedding_dim"] = embedding_dim
    model_cfg["transformer_layer_factory"]["attention_factory"]["n_heads"] = num_heads
    model_cfg["transformer_layer_factory"]["attention_factory"]["qk_dim"] = qk_dim
    model_cfg["transformer_layer_factory"]["attention_factory"]["v_dim"] = qk_dim
    model_cfg["transformer_layer_factory"]["feedforward_dim"] = embedding_dim * 4

    layer_sweep = LayerSweepFactory.model_validate(pretraining_cfg["layer_sweep"]).build(ctx)
    batch_size, _, total_descent_steps = layer_sweep.find_batch_size(
        ctx,
        data_config_dict=data_cfg,
        model_config_dict=model_cfg,
    )
    if batch_size <= 0 or total_descent_steps <= 0:
        raise optuna.TrialPruned("No valid memory-fitting batch size for this trial.")

    ctx = ctx.fork(
        batch_size=batch_size,
        accumulation_steps=max(acc_batch_size // max(batch_size, 1), 1),
        descent_steps=total_descent_steps,
    )
    short_descent_steps = int(
        max(
            pretraining_cfg["learning_rate_sweep"]["sweep_time"]
            * total_descent_steps
            / pretraining_cfg["training_time"],
            1,
        )
    )
    # Skip validation during short run and do one full validation at end.
    ctx = ctx.fork(val_frequency=short_descent_steps + 1)

    # Build datasets/model with tokenizer override.
    ctx, data_and_model_configs = init_datasets_and_models(
        ctx,
        shuffle=False,
        data_config_dict=data_cfg,
        model_config_dict=model_cfg,
    )
    # Run full validation set at end.
    training_cfg = copy.deepcopy(training_cfg)
    # training_cfg["validation_step_factory"]["validation_batches"] = None
    ctx.merge(
        {
            "config_dicts": {
                "run_context": ctx_cfg,
                "server": server_context,
                "data": data_cfg,
                "model": model_cfg,
                "pretraining": pretraining_cfg,
                "training": training_cfg,
            }
        }
    )

    training_loop, _ = build_component_from_dict(
        TrainingLoopFactory,
        training_cfg,
        ctx.fork(accumulation_steps=max(ctx.accumulated_batch_size // ctx.batch_size, 1)),
    )
    training_loop.descent_steps = short_descent_steps
    token_count, train_loss, _, best_train_loss, best_val_loss, _ = training_loop.run()
    final_val_loss = training_loop.validation_step.step()
    training_loop.aim_logger.track_val_metrics({"loss": final_val_loss}, short_descent_steps)
    training_loop.aim_logger["token_count"] = token_count
    training_loop.aim_logger["best_train_loss"] = best_train_loss
    training_loop.aim_logger["final_val_loss"] = final_val_loss
    training_loop.aim_logger.close()

    trial.set_user_attr("batch_size", int(batch_size))
    trial.set_user_attr("total_descent_steps", int(total_descent_steps))
    trial.set_user_attr("final_train_loss", float(train_loss))
    return float(final_val_loss)

def main():
    study = optuna.create_study(
        study_name="llm_optuna_search",
        direction="minimize",                 # minimizing final_val_loss
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(test_config, n_trials=100)

    print(f"Best final_val_loss: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main()
