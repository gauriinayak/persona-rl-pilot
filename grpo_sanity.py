"""Step 0: does GRPO actually change a small model on verifiable math? (No persona yet.)

Trains a LoRA adapter with TRL's GRPOTrainer on GSM8K, evaluates the base model and every
saved checkpoint on a fixed held-out set, and writes report.json for sanity_summary.py.

    python grpo_sanity.py --seed 42 --output runs/grpo-sanity-s42
    python sanity_summary.py runs/grpo-sanity-s{42,43,44}/report.json

Pipeline check only (a few minutes):  python grpo_sanity.py --smoke --output runs/grpo-smoke
"""

import argparse
import gc
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from rl_task import correct_reward, extract_final, format_reward, load_gsm8k, same_number

LOG_KEYS = (
    "step", "reward", "reward_std", "frac_reward_zero_std", "kl", "entropy",
    "completions/mean_length", "completions/clipped_ratio",
)


def load_model(name, revision, device, adapter=None):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype).to(device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


@torch.inference_mode()
def greedy_eval(model, tokenizer, rows, max_new_tokens, batch_size):
    """Greedy decode held-out items; report accuracy, format rate, length, truncation."""
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}
    correct, formatted, lengths, truncated = [], [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
        for row, ids in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
            stops = [i for i, token in enumerate(ids) if token in eos_ids]
            lengths.append(stops[0] + 1 if stops else len(ids))
            truncated.append(0 if stops else 1)
            final = extract_final(tokenizer.decode(ids, skip_special_tokens=True))
            formatted.append(int(final is not None))
            correct.append(int(same_number(final, row["answer"])))
    n_formatted = sum(formatted)
    return {
        "accuracy": mean(correct),
        "format_rate": mean(formatted),
        "accuracy_given_format": sum(correct) / n_formatted if n_formatted else 0.0,
        "mean_tokens": mean(lengths),
        "truncation_rate": mean(truncated),
        "correct": correct,
    }


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5, help="LoRA scale; sweep 1e-5 to 1e-4.")
    parser.add_argument("--beta", type=float, default=0.001, help="KL coefficient; KL is only logged if > 0.")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-batch", type=int, default=16, help="Completions per device per micro-step.")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-completion", type=int, default=384)
    parser.add_argument("--eval-n", type=int, default=300)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--vllm", action="store_true", help="Generate with vLLM (colocated). Needs trl[vllm].")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to check the pipeline end to end.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        args.max_steps, args.save_steps, args.eval_n, args.eval_batch = 4, 2, 16, 8
    if args.output.exists():
        parser.error("Choose a fresh output directory.")
    if args.device == "cpu":
        print("WARNING: no CUDA device found; this will be very slow.", flush=True)
    args.output.mkdir(parents=True)
    set_seed(args.seed)

    train = load_gsm8k("train")
    eval_rows = list(load_gsm8k("test", n=args.eval_n, seed=2026))  # fixed across training seeds

    print("Evaluating base model", flush=True)
    base = load_model(args.model, args.revision, args.device)
    revision = getattr(base.config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    baseline = greedy_eval(base, tokenizer, eval_rows, args.max_completion, args.eval_batch)
    del base
    free_memory()

    config = GRPOConfig(
        output_dir=str(args.output), seed=args.seed, data_seed=args.seed,
        learning_rate=args.lr, beta=args.beta, num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=args.grad_accum,
        max_completion_length=args.max_completion, max_steps=args.max_steps,
        save_steps=args.save_steps, save_only_model=True, logging_steps=1,
        bf16=args.device.startswith("cuda"), use_vllm=args.vllm, report_to=args.report_to,
        reward_weights=[1.0, 0.0],  # train on correctness; log format without optimizing it
        model_init_kwargs={"dtype": torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                           "revision": revision},
    )
    print("Running GRPO", flush=True)
    trainer = GRPOTrainer(
        model=args.model, reward_funcs=[correct_reward, format_reward], args=config,
        train_dataset=train,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    final_dir = args.output / "final"
    trainer.save_model(str(final_dir))
    rl_log = [
        {k: v for k, v in row.items() if k in LOG_KEYS or k.startswith("rewards/")}
        for row in trainer.state.log_history if "reward" in row
    ]
    del trainer
    free_memory()

    checkpoints = {}
    saved = sorted(
        (p for p in args.output.glob("checkpoint-*") if int(p.name.split("-")[1]) != args.max_steps),
        key=lambda p: int(p.name.split("-")[1]),
    )  # the last checkpoint duplicates "final"
    for path in [*saved, final_dir]:
        name = "final" if path == final_dir else path.name.split("-")[1]
        print(f"Evaluating checkpoint {name}", flush=True)
        model = load_model(args.model, revision, args.device, adapter=path)
        checkpoints[name] = greedy_eval(model, tokenizer, eval_rows, args.max_completion, args.eval_batch)
        del model
        free_memory()

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    here = Path(__file__).parent
    report = {
        "seed": args.seed,
        "protocol": {
            "model": args.model, "revision": revision, "args": {k: str(v) for k, v in vars(args).items()},
            "grpo_config": json.loads(json.dumps(config.to_dict(), default=str)),
            "code_sha256": {
                name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                for name in ("grpo_sanity.py", "rl_task.py")
            },
            "versions": versions,
        },
        "baseline": baseline, "checkpoints": checkpoints, "rl_log": rl_log,
    }
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
