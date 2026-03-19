"""Backward-compatible Hydra CLI entry point."""

from .experiments.hydra import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
