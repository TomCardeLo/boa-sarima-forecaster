"""Click-based CLI for boa-forecaster.

Entry point registered as ``boa-forecaster`` in ``pyproject.toml``.
Subcommands live in sibling modules (``run``, ``validate``, ``compare``) so
each can evolve independently.
"""

from __future__ import annotations

import click

from boa_forecaster import __version__
from boa_forecaster.cli import compare as compare_mod
from boa_forecaster.cli import run as run_mod
from boa_forecaster.cli import validate as validate_mod


@click.group()
@click.version_option(version=__version__, prog_name="boa-forecaster")
def cli() -> None:
    """boa-forecaster: Bayesian TPE time-series forecasting."""


cli.add_command(run_mod.run)
cli.add_command(validate_mod.validate)
cli.add_command(compare_mod.compare)


__all__ = ["cli"]
