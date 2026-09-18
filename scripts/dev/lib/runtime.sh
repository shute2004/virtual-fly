#!/usr/bin/env bash

# Shared runtime discovery for current virtual-fly developer entry points.
# The repository can be used from a normal shell with uv/cargo on PATH or from
# DevSpace-like environments where only the checked-out .venv and ~/.cargo are
# available.

vf_resolve_python() {
  local root="$1"
  if command -v uv >/dev/null 2>&1; then
    VF_PYTHON=(uv run python)
  elif [ -x "$root/.venv/bin/python" ]; then
    VF_PYTHON=("$root/.venv/bin/python")
  else
    echo 'uv or an existing .venv/bin/python is required' >&2
    return 127
  fi
}

vf_resolve_cargo() {
  # Keep Rust build artifacts out of the repository.  Cargo's default `target/`
  # directory can add hundreds of MB to virtual-fly after one bridge build.
  if [ -z "${CARGO_TARGET_DIR:-}" ]; then
    export CARGO_TARGET_DIR="${VF_CARGO_TARGET_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/virtual-fly/cargo-target}"
  fi
  if command -v cargo >/dev/null 2>&1; then
    VF_CARGO=(cargo)
  elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    VF_CARGO=("$HOME/.cargo/bin/cargo")
  else
    echo 'cargo is required' >&2
    return 127
  fi
}
