# Local Python environments

These environments were installed and checked on Apple Silicon with Python 3.11.6.

## Main course environment

From this directory:

```bash
source .venv/bin/activate
python -m pip check
```

In VS Code, choose `ARENA_materials/.venv/bin/python` as the interpreter or notebook kernel.
To recreate the installed package versions:

```bash
python3.11 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-macos-py311.lock.txt
```

The lock file records the complete installed dependency set, including Git commit
references. Use it for this Mac instead of resolving the unpinned requirements
again. CUDA is unavailable on this Mac; use CPU or PyTorch MPS where supported.

## Legacy TorchText

TorchText 0.18 requires the PyTorch 2.3 binary interface. The main interpretation
stack requires newer PyTorch, so TorchText has a separate environment:

```bash
source .venv-torchtext/bin/activate
python -c 'import torch, torchtext; print(torch.__version__, torchtext.__version__)'
```

Select `.venv-torchtext/bin/python` only for legacy TorchText work. Recreate it with:

```bash
python3.11 -m venv .venv-torchtext
uv pip install --python .venv-torchtext/bin/python -r requirements-torchtext-py311.lock.txt
```

## Compatibility changes

- Brax 0.12.1 works with the existing JAX 0.4.38 pin. Brax 0.14.2 requires
  an Orbax version that needs JAX >=0.5, so the previous combination was unsatisfiable.
- The main environment uses the official `mujoco` bindings. The obsolete
  `mujoco-py` bindings are excluded on macOS; their missing MuJoCo 2.1 runtime
  prevented even modern Gymnasium MuJoCo environments from loading.
- Other dependencies were resolved together using uv's standard resolver.
  No legacy resolver or ignored dependency constraints are needed for this setup.

Verified: dependency consistency in both environments; TorchText, TorchVision,
and TorchAudio imports in the legacy environment; core ML and evaluation imports;
a tiny TransformerLens forward pass; a JIT-compiled JAX calculation; CartPole and
HalfCheetah-v4 reset/step; and a Brax positional HalfCheetah reset/step.
EnvPool also passed CartPole-v1 and Pong-v5 reset/step with automatic native
runtime loading in a fresh Python process.
This does not validate every course notebook, model download, API credential,
GPU workload, or graphical rendering backend.

## EnvPool native runtime on this Mac

EnvPool 1.2.5 loads its Procgen extension during import, which needs Qt 5.
Qt 5.15.19 was installed under `/opt/homebrew/opt/qt@5` from its Sonoma bottle.
Homebrew's standard dependency installation stopped because the Apple Command
Line Tools were too old. The Qt bottle was installed without dependency upgrades;
the required existing libraries were checked by actually importing EnvPool.
The missing `md4c` library was compiled locally from the official 0.6.0 source.
Homebrew also installed `bison` while attempting the initial installation.

The local library is `.native/lib/libmd4c.0.dylib`. A small Python startup file
(`arena_native_runtime.pth`, copied into `.venv/lib/python3.11/site-packages/`)
loads it before EnvPool. This applies when launching Python directly or in a
notebook, without requiring additional shell environment variables.

To recreate that local library on the same Mac (Qt 5 is still required):

```bash
mkdir -p .native/src .native/lib
curl -fL https://github.com/mity/md4c/archive/refs/tags/v0.6.0.tar.gz -o .native/md4c-0.6.0.tar.gz
shasum -a 256 .native/md4c-0.6.0.tar.gz
# Must equal: 4d151298125a81da3b2efa2e0eed8bdb7a9318569804e4fa4d7a2375ab83ef70
tar -xzf .native/md4c-0.6.0.tar.gz -C .native/src
clang -O2 -dynamiclib .native/src/md4c-0.6.0/src/md4c.c -install_name /opt/homebrew/opt/md4c/lib/libmd4c.0.dylib -o .native/lib/libmd4c.0.dylib
cp arena_native_runtime.pth .venv/lib/python3.11/site-packages/
```

Upstream references:
- https://github.com/pytorch/text#installation
- https://github.com/openai/mujoco-py#readme
