"""Run with python -m facs; no training or network downloads occur on import."""

import argparse
import json

from .diagnostics import benchmark, overfit, pretrain_edge
from .engine import evaluate, read_config, train
from .checkpoints import migrate_legacy


def main(argv=None):
    parser = argparse.ArgumentParser(description="FACS-Net paper reimplementation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "overfit", "benchmark", "pretrain-edge"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    commands.choices["train"].add_argument("--resume")
    commands.choices["train"].add_argument("--stop-after-epoch", type=int)
    commands.choices["overfit"].add_argument("--steps", type=int, default=128)
    commands.choices["overfit"].add_argument("--sample-ids", type=int, nargs="+")
    commands.choices["benchmark"].add_argument("--batch-sizes", type=int, nargs="+", default=[2, 12])
    commands.choices["benchmark"].add_argument("--steps", type=int, default=5)
    commands.choices["pretrain-edge"].add_argument("--epochs", type=int, default=5)
    commands.choices["pretrain-edge"].add_argument("--batch-size", type=int, default=64)
    commands.choices["pretrain-edge"].add_argument("--noise-sigma", type=float, default=0.1)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--subset")
    evaluation.add_argument("--batch-size", type=int, default=12)
    evaluation.add_argument("--workers", type=int, default=4)
    evaluation.add_argument("--no-save-predictions", action="store_true")
    migration = commands.add_parser("migrate-legacy")
    migration.add_argument("--checkpoint", required=True)
    migration.add_argument("--config", required=True)
    migration.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "migrate-legacy":
        result = migrate_legacy(args.checkpoint, read_config(args.config), args.output)
    elif args.command == "evaluate":
        result = evaluate(args.checkpoint, args.output_dir, split=args.split, device=args.device,
                          subset=args.subset, batch_size=args.batch_size, workers=args.workers,
                          save_predictions=not args.no_save_predictions)
    else:
        config = read_config(args.config)
        if args.command == "train":
            result = train(config, args.run_dir, device=args.device, resume=args.resume, stop_after_epoch=args.stop_after_epoch)
        elif args.command == "overfit":
            result = overfit(config, args.run_dir, device=args.device, steps=args.steps, sample_ids=args.sample_ids)
        elif args.command == "benchmark":
            result = benchmark(config, args.run_dir, device=args.device, batch_sizes=args.batch_sizes, steps=args.steps)
        else:
            result = pretrain_edge(config, args.run_dir, device=args.device, epochs=args.epochs,
                                   batch_size=args.batch_size, noise_sigma=args.noise_sigma)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
