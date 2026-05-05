import json
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scifact_midway import build_experiment_config, run_all_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciFact midway experiments.")
    parser.add_argument("--preset", default="cpu_quick", choices=["cpu_quick", "gpu_midway", "gpu_strong"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--retriever-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--retriever-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--negatives-per-claim", type=int, default=None)
    parser.add_argument("--prefer-device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    overrides = {
        "retriever_model_name": args.retriever_model,
        "verifier_model_name": args.verifier_model,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "retriever_batch_size": args.retriever_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "negatives_per_claim": args.negatives_per_claim,
        "prefer_device": args.prefer_device,
        "seed": args.seed,
    }
    config = build_experiment_config(preset=args.preset, overrides=overrides)
    results = run_all_experiments(
        project_root=PROJECT_ROOT,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        config=config,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
