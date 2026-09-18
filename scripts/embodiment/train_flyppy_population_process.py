#!/usr/bin/env python3
"""Compatibility CLI for the process-isolated population trainer."""
from virtual_fly.training.population_process import *  # noqa: F401,F403
from virtual_fly.training.population_process import main

if __name__ == "__main__":
    raise SystemExit(main())
