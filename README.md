# Mixture of Geometric Experts (MGM)
A research-grade Large Language Model (LLM) framework that **mixes multiple geometric experts** (Euclidean, Hyperbolic, Spherical, *etc.*) and a powerful routing gate to achieve manifold specialisation, efficient capacity scaling and improved reasoning.

This repository collects a **self-contained training & evaluation pipeline** together with production tooling (resume, monitoring, dataset validation, memory guard) and several ready-to-run configurations that scale from a laptop smoke test to a 2 B-parameter flagship system.

---
## Table of Contents
1. [Quick start](#quick-start)
2. [Directory layout](#directory-layout)
3. [Core scripts](#core-scripts)
4. [Utility & tooling](#utility--tooling)
5. [Configuration files](#configuration-files)
6. [Common workflows](#common-workflows)
7. [FAQ](#faq)
8. [License](#license)

---
## Quick start
```bash
# Create & activate a fresh environment (conda / venv)
python -m pip install --upgrade pip
pip install torch transformers datasets wandb geoopt

# Clone & enter the repo
git clone https://github.com/<you>/Mixture-of-geometric-experts.git
cd Mixture-of-geometric-experts

# Run an end-to-end **smoke test** (takes <60 s on CPU)
cd mgm_project/scripts/6-10-25
python3 integration_test_runner.py --stage-steps 1 \
    --hidden-dim 64 --seq-len 32 --experts-num 8 --k-experts 2 \
    --num-heads 4 --learning-rate 3e-4 --dataset-conversational \
    --amp-off --flash-attention-off --skip-setup --sample-size 10 \
    --validation-steps 1 --nuanced-routing
```
The run prints curvature analysis and should end with **"EXCELLENT expert diversity"**.

---
## Directory layout
```text
Mixture-of-geometric-experts/
├─ mgm_project/scripts/6-10-25/   # All runnable scripts (see below)
│  ├─ *.py                        # Training, validation, monitoring, …
│  ├─ *.json                      # Ready-made configs
│  └─ validator_cache/            # Auto-generated artefacts
├─ tests/                         # Unit tests
└─ README.md                      # You are here
```

---
## Core scripts
The **`mgm_project/scripts/6-10-25`** folder is intentionally self-sufficient. Every script can be executed directly with `python3 <script>.py …`.

| Script | Purpose | Typical usage |
|--------|---------|---------------|
| `concept_aware_mgm.py` | Minimal demo showing concept-aware routing & generation. | `python3 concept_aware_mgm.py` |
| `train_geometric_model_v2.py` | Full training loop + model definition (~2 k LOC). Automatically invoked by other wrappers. | _Do not call directly_ (unless you want to hack the internals). |
| `integration_test_runner.py` | Patched training launcher used in CI / quick benchmarks. Allows **hundreds** of CLI flags to monkey-patch JSON configs on the fly. | See examples in [Quick start](#quick-start). |
| `run_training.py` | Generates an **optimised config** (`optim_run_cfg.json`) and a checkpoint-aware wrapper, then launches `train_geometric_model_v2.py`. Ideal for medium-scale experiments. | `python run_training.py` |
| `run_flagship_production.py` | Orchestrates a **32 K-context / 64-expert / 2 B param** production run. Handles environment tuning, dependency installation, monitoring and packaging. | `python run_flagship_production.py --config production_flagship_config.json` |
| `resume_orchestrator.py` & `resume_helper.py` | Robust resume logic – download or discover the latest checkpoint, patch optimiser / scheduler / AMP scaler state and continue training from the correct stage. | `python resume_orchestrator.py <ckpt_or_dir> --config optim_run_cfg.json` |

### Gating & experts
Internally the model lives in `train_geometric_model_v2.py` and is composed of:
* `GeometricExpert` – manifold-specific feed-forward block (+ exp/log maps)
* `NuancedGeometricGate` – temperature-annealed top-*k* router with concept analysis
* `ConceptGroupCombiner` / `SpectralCombiner` – merge expert outputs
* `ThoughtGenerator` – lightweight reasoning head used during PPO alignment

---
## Utility & tooling
| Script | What it does | Why you need it |
|--------|--------------|-----------------|
| `production_dataset_validator.py` | Deep inspection of NPZ/NPY or streaming datasets – detects out-of-vocab tokens, dtype inconsistencies, negative indices, sequence overflows, *etc.* Generates JSON & pretty console summary. | Run **before any long training** to avoid cryptic CUDA errors hours later.<br>`python production_dataset_validator.py optimized_minimal_memory_config.json` |
| `memory_guard.py` | Real-time GPU/CPU memory watchdog, gradient NaN filter and adaptive batch-size tuner. Import and wrap your training loop to make OOMs virtually impossible. | See `DynamicMemoryGuard` class for example usage. |
| `streaming_dataset_loader.py` | June-2025 grade HF streaming wrapper with retry logic, webdataset fixes and rolling cache. Powers all streaming runs. | Imported automatically by the training scripts. |
| `training_monitor.py` | Out-of-band watchdog that tails logs, checks GPU utilisation, disk space and kills/restarts stale runs. Ideal for remote clusters. | `python -m mgm_project.scripts.6-10-25.training_monitor` |

---
## Configuration files
All configs follow the same schema and can be hot-patched via CLI flags (see **`integration_test_runner.py`**):

* `smoke_cfg.json` – 8-token context & 16 experts, pure CPU verification.
* `optimized_minimal_memory_config.json` – 3 experts, single GPU friendly.
* `mgm_config.json` – Research default (16 experts, 1 K context).
* `production_flagship_config.json` – 64 experts, 32 K context, multimodal.
* `test_config_patched.json` – Auto-generated by the integration test runner.

> **Tip:** copy one of these and tweak `model.num_experts`, `training.batch_size`, … or drive everything from CLI flags (e.g. `--experts-num 8 --k-experts 2`).

---
## Common workflows
### 1. Smoke / CI test (CPU or single GPU)
```bash
cd mgm_project/scripts/6-10-25
python3 integration_test_runner.py --amp-off --flash-attention-off \
       --hidden-dim 64 --seq-len 32 --experts-num 8 --k-experts 2 \
       --stage-steps 1 --sample-size 10 --validation-steps 1
```

### 2. Dataset validation only
```bash
python3 production_dataset_validator.py optimized_minimal_memory_config.json
```

### 3. Full streaming training (research scale)
```bash
python run_training.py              # creates optim_run_cfg.json
# (optional) resume later:
python resume_orchestrator.py checkpoints/ --config optim_run_cfg.json
```

### 4. Flagship multimodal run (multi-GPU or TPU-v5e)
```bash
python run_flagship_production.py --config production_flagship_config.json
```

---
## FAQ
**Q:** *Why do I get "Out of memory" even at small batch sizes?*

> Enable the guard: wrap your training loop with `DynamicMemoryGuard.safe_forward_pass()` / `.safe_backward_pass()`.

**Q:** *How do I add my own dataset?*

> 1. Edit / copy a JSON config.<br>2. Add a new entry under `streaming.modalities`. The high-level loader takes care of the rest.

**Q:** *Can I train without internet access?*

> Yes – place NPZ/NPY shards and point `data.npz_files` / `data.npy_files` to them. The same trainer works offline.

---
## License
This project is licensed under the [Apache 2.0](LICENSE) license.
