<!-- markdownlint-disable-file no-inline-html -->

# Installation

<span style="font-variant: small-caps;">Deconvolve</span> installs a single command-line program, `deconvolve`. The CLI can be installed through a Python installer such as `uv`, `pipx` or `pip`.

## Prerequisites

- **Python** `>= 3.12`
- A **Python installer** such as `uv`, `pipx` or `pip`. `uv` can fetch a suitable python interpreter, and therefore does not require one to be available on the system.

To install `uv` see the instructions from [UV-Astral](https://docs.astral.sh/uv/getting-started/installation/).

=== "uv (recommended)"

    ```shell
    uv tool install deconvolve@latest
    deconvolve --help
    ```

    `uv tool install` places `deconvolve` on the system `PATH` in its own isolated environment, so that its dependencies cannot collide with any other installed tools.

=== "uvx (no install)"

    ```shell
    uvx deconvolve --help
    ```

    Runs <span style="font-variant: small-caps;">Deconvolve</span> in an ephemeral environment without permanent installation.

=== "pipx"

    ```shell
    pipx install deconvolve
    deconvolve --help
    ```

    The same isolated-environment model as `uv tool install`.

=== "pip"

    ```shell
    python -m venv .venv
    source .venv/bin/activate
    pip install deconvolve
    deconvolve --help
    ```

    It is recommended that <span style="font-variant: small-caps;">Deconvolve</span> be installed into a virtual environment rather than the system interpreter. <span style="font-variant: small-caps;">Deconvolve</span> requires JAX and Keras, amongst other dependencies, and pinning them system-wide may conflict with other installed tools.

---

## Hardware and accelerator support

The JAX dependency resolves by platform, so the correct build is selected automatically by the installer above.

=== "Linux (x86_64, NVIDIA GPU)"

    <span style="font-variant: small-caps;">Deconvolve</span> installs `jax[cuda13]`, built against CUDA 13.

    The CUDA runtime libraries are provided as PyPI wheels and are resolved automatically, so the only host requirement is an NVIDIA driver new enough for CUDA 13. No system provided CUDA toolkit is required.

=== "macOS (Apple Silicon, arm64)"

    <span style="font-variant: small-caps;">Deconvolve</span> installs plain `jax` and runs on the CPU. The official macOS arm64 JAX wheels provide no GPU acceleration.

    Experimental alternatives, such as `jax-mps` or `IREE`-based workflows, may enable Metal acceleration, but these configurations are not officially tested or supported by <span style="font-variant: small-caps;">Deconvolve</span>. Users should independently validate their correctness and performance.

=== "Other platforms"

    Any other platform resolves to CPU-only JAX if a wheel exists for it. Only the two configurations above are officially supported.

---

## Shell completion

The `deconvolve` CLI is built with [Typer](https://typer.tiangolo.com/) and can install its own completion script:

=== "zsh"

    ```shell
    deconvolve --install-completion zsh
    ```

=== "bash"

    ```shell
    deconvolve --install-completion bash
    ```

=== "fish"

    ```shell
    deconvolve --install-completion fish
    ```

Shell completion requires the `deconvolve` program to be on the `PATH`: completion is registered against the command name, and therefore cannot be activated through `uv run -m deconvolve` or `python -m deconvolve`.

---

## Contributing

Working on <span style="font-variant: small-caps;">Deconvolve</span> itself requires `uv` to be installed in addition to a checkout of the repository.

1. **Clone the repository:**

    ```shell
    git clone https://github.com/krishdesai7/deconvolve.git
    cd deconvolve
    ```

2. **Synchronize dependencies:**

    ```shell
    uv sync
    ```

    This creates a `.venv` with every runtime and development dependency at
    the exact versions in `uv.lock`.

Inside a checkout, CLI help is available through

```shell
uv run deconvolve --help
```
