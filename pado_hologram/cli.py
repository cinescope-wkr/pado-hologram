from __future__ import annotations

import argparse
import importlib.util
import sys
from importlib.resources import files
from pathlib import Path
from typing import Sequence

from . import DESCRIPTION, PROJECT_NAME, __version__
from .experiments import (
    available_experiments,
    compose_experiment_config,
    render_config_yaml,
    run_experiment,
)

ASCII_ART = r"""
~  .  coherent wavefronts / phase / interference  .  ~

PPPP    AAA    DDDD    OOO
P   P  A   A   D   D  O   O
PPPP   AAAAA   D   D  O   O
P      A   A   D   D  O   O
P      A   A   DDDD    OOO

              PADO Hologram
""".strip("\n")

PACKAGE_TREE = """
pado_hologram/
  core/             source, targets, losses, pipelines
  devices/          SLM-facing and camera/observation abstractions
  phase_only/       GS and DPAC
  primitive_based/  Gaussian renderer baseline and future splatting paths
  experiments/      registry, config composition, Hydra runners
  backends/         optional kernel backends such as Warp
  representations/  primitive-based CGH data models
  neural/           capture, calibration, and learning-facing scaffolds
""".strip("\n")


def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _status(module_name: str, *, optional: bool = False) -> tuple[str, str]:
    available = _is_available(module_name)
    if available:
        return "available", "ok"
    if optional:
        return "optional / not installed", "warn"
    return "missing", "error"


def _print_banner() -> None:
    print(ASCII_ART)
    print(f"{PROJECT_NAME} {__version__}")
    print(DESCRIPTION)
    print()


def _cmd_banner(_: argparse.Namespace) -> int:
    _print_banner()
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    _print_banner()
    print("Package:")
    print(f"- version: {__version__}")
    print(f"- root: {Path(__file__).resolve().parent}")
    print(f"- config: {files('pado_hologram.conf')}")
    print(f"- experiments: {', '.join(available_experiments())}")
    print()
    print("Quick start:")
    print("- pado-hologram doctor --run-smoke")
    print("- pado-hologram run experiment=gs")
    print("- pado-hologram run experiment=dpac target=gaussian")
    print("- pado-hologram run experiment=primitive_gaussian")
    print("- pado-hologram run experiment=primitive_gaussian_splat")
    print("- pado-hologram run experiment=primitive_gaussian_wave primitives=gaussian_depth_ring")
    print("- pado-hologram run experiment=primitive_gaussian_awb primitives=gaussian_depth_ring")
    print("- pado-hologram run experiment=primitive_gaussian_rpws primitives=gaussian3d_depth_ring  # exact RPWS baseline")
    print("- pado-hologram run experiment=primitive_gaussian_gws_exact primitives=gaussian3d_depth_ring")
    print("- pado-hologram run experiment=primitive_gaussian_splat backend=warp")
    print("- pado-hologram run experiment=primitive_gaussian_splat primitives=gaussian_ring backend=torch")
    print("- pado-hologram run experiment=primitive_gaussian_splat primitives=gaussian_ring camera=binned2 backend=torch")
    print("- pado-hologram tree")
    print("- Visit docs: https://cinescope-wkr.github.io/pado-hologram/")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    _print_banner()
    checks = {
        "pado": _status("pado"),
        "pado_hologram": _status("pado_hologram"),
        "hydra-core": _status("hydra"),
        "NVIDIA Warp": _status("warp", optional=True),
    }

    print("Environment checks:")
    failures = 0
    for name, (label, level) in checks.items():
        print(f"- {name}: {label}")
        if level == "error":
            failures += 1

    try:
        cfg = compose_experiment_config()
        print("- config compose: ok")
        print(f"- default experiment: {cfg.experiment.method}")
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"- config compose: failed ({exc})")
        failures += 1
        cfg = None

    if args.run_smoke and cfg is not None:
        try:
            summary = run_experiment(cfg)
            print(f"- smoke run: ok ({summary.method})")
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"- smoke run: failed ({exc})")
            failures += 1

    print()
    print("Available experiments:")
    for name in available_experiments():
        print(f"- {name}")

    return 1 if failures else 0


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = compose_experiment_config(args.overrides, config_name=args.config_name)
    if args.show_config:
        print(render_config_yaml(cfg).rstrip())
        print()
    summary = run_experiment(cfg)
    print(f"method: {summary.method}")
    print("metrics:")
    for key, value in summary.metrics.items():
        print(f"  {key}: {value}")
    print("extras:")
    for key, value in summary.extras.items():
        print(f"  {key}: {value}")
    return 0


def _cmd_tree(_: argparse.Namespace) -> int:
    _print_banner()
    print(PACKAGE_TREE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pado-hologram",
        description="CLI entry point for the PADO Hologram package.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("banner", help="Print the package banner.")
    subparsers.add_parser("info", help="Show package information and quick-start commands.")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check optional dependencies and optionally run a smoke experiment.",
    )
    doctor_parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="Run the default experiment after composing the default config.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Compose an experiment config from overrides and run it without Hydra changing directories.",
    )
    run_parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name to compose from pado_hologram.conf.",
    )
    run_parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved config before running the experiment.",
    )
    run_parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides such as experiment=dpac target=gaussian backend=warp.",
    )

    subparsers.add_parser("tree", help="Show the intended internal package layout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["info"]
    args = parser.parse_args(args_list)

    command = args.command or "info"
    if command == "banner":
        return _cmd_banner(args)
    if command == "doctor":
        return _cmd_doctor(args)
    if command == "run":
        return _cmd_run(args)
    if command == "tree":
        return _cmd_tree(args)
    return _cmd_info(args)


if __name__ == "__main__":
    raise SystemExit(main())
