# Installation

RAN installs a single command-line program, `ran`. Pick the path that matches
what you want to do:

- **Use RAN** — install the CLI with whichever Python installer you already
  have. [`uv`](https://docs.astral.sh/uv/) is recommended because it resolves
  and installs considerably faster, but nothing here requires it.
- **Work on RAN** — see [Development install](#development-install), which does
  expect `uv`.

!!! warning "RAN is not on PyPI yet"

    Do **not** run `pip install deconvolve`. The name `ran` on PyPI belongs to an
    unrelated project, and you would install someone else's package.

    Until the first release is published, install from the Git repository
    using the commands below — they all work today. The published
    distribution will be named `ranv4`, so these will eventually shorten to
    `uv tool install ranv4`, `pipx install ranv4` or `pip install deconvolvev4`.

---

## Prerequisites

- **Python** `>= 3.12`. Development happens on 3.14; the test suite runs on
  3.12, 3.13 and 3.14.
- A **Python installer** — one of `uv`, `pipx` or `pip`. If you use `uv` it
  will fetch a suitable Python for you, so you do not need one installed
  first.

To install `uv` itself:

=== "macOS / Linux"

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "pip"

    ```shell
    pip install uv
    ```

---

## Install the CLI

=== "uv (recommended)"

    ```shell
    uv tool install git+https://github.com/krishdesai7/deconvolve.git
    ran --help
    ```

    `uv tool install` places `ran` on your `PATH` in its own isolated
    environment, so RAN's dependencies cannot collide with anything else you
    have installed.

=== "uvx (no install)"

    ```shell
    uvx --from git+https://github.com/krishdesai7/deconvolve.git ran --help
    ```

    Runs RAN in a throwaway environment without installing anything
    permanently. Useful for trying it once, but it re-resolves on each run.

=== "pipx"

    ```shell
    pipx install git+https://github.com/krishdesai7/deconvolve.git
    ran --help
    ```

    The same isolated-environment model as `uv tool install`.

=== "pip"

    ```shell
    python -m venv .venv
    source .venv/bin/activate
    pip install git+https://github.com/krishdesai7/deconvolve.git
    ran --help
    ```

    Install into a virtual environment rather than the system interpreter.
    RAN pulls in JAX, Keras, NumPy and Matplotlib, and pinning those
    system-wide will eventually conflict with something else.

---

## Hardware and accelerator support

The JAX dependency resolves by platform, so the correct build is selected
automatically by every installer above.

=== "Linux (x86_64, NVIDIA GPU)"

    RAN installs `jax[cuda13]`, built against CUDA 13.

    The CUDA runtime libraries ship as PyPI wheels and are pulled in
    automatically, so the only host requirement is an NVIDIA driver new
    enough for CUDA 13. No system CUDA toolkit is needed.

=== "macOS (Apple Silicon, arm64)"

    RAN installs plain `jax` and runs on the CPU. The official macOS arm64
    JAX wheels provide no GPU acceleration.

    Experimental backends such as `jax-metal` are neither tested nor
    supported here.

=== "Other platforms"

    Any other platform resolves to CPU-only JAX if a wheel exists for it.
    Only the two configurations above are tested.

---

## Shell completion

The `ran` CLI is built with [Typer](https://typer.tiangolo.com/) and can
install its own completion script:

=== "zsh"

    ```shell
    ran --install-completion zsh
    ```

=== "bash"

    ```shell
    ran --install-completion bash
    ```

=== "fish"

    ```shell
    ran --install-completion fish
    ```

This needs the installed `ran` program: completion is registered against the
command name, so it does not work through `python -m ran`.

---

## Development install

Working on RAN itself expects `uv`, which is what the lockfile, the test
matrix and the `just` recipes are written against.

1.  **Clone the repository:**

    ```shell
    git clone https://github.com/krishdesai7/deconvolve.git
    cd Deconvolve
    ```

2.  **Synchronize dependencies:**

    ```shell
    uv sync
    ```

    This creates a `.venv` with every runtime and development dependency at
    the exact versions in `uv.lock`.

Inside a checkout, run the CLI through `uv run` so it uses the project
environment without your having to activate it:

```shell
uv run deconvolve --help
```

To add the documentation toolchain (MkDocs, the Material theme, mkdocstrings):

```shell
uv sync --group docs
```

---

## Verifying the installation

```shell
# The CLI is on your PATH and reports its commands
ran --help
```

In a development checkout, run the fast half of the test suite — everything
except the tests marked `slow`:

```shell
uv run just test-fast
```
