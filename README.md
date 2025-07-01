# Mixture of Geometric Experts (MGM)

**Architecture:** MGM is a mixture-of-experts Transformer that incorporates multiple geometric manifolds and a working memory. It maintains *N* expert subnetworks (“manifolds”) of different types (e.g. Euclidean, hyperbolic, spherical, etc.) and uses an 8-way Mixture-of-Experts routing: at each layer the top-*K* experts are selected by the gating network for each token.  A **working memory** of 256 slots (each 2048-dimensional) is fused into the model’s computation.  Each Transformer block uses **RMSNorm** (root-mean-square normalization, omitting the mean subtraction) instead of LayerNorm (as in recent LLaMA-style models) to stabilize training.  Key architecture flags include `--hidden-dim` (hidden/embedding size), `--experts-num` (number of experts), `--k-experts` (top-*K* selected each step), and `--num-heads` (self-attention heads).  For example, the integration tests run with `--hidden-dim 128 --experts-num 16 --k-experts 4 --num-heads 12`. These determine the model’s depth and width (e.g. hidden size 1600 in the final model) and thus its total parameter count.  In practice, MGM’s parameter count is on the order of **hundreds of millions** (exact count depends on the chosen hidden size, number of experts, etc.).

* **Mixture-of-Experts Routing:** Uses top-*K* expert selection with curvature-aware gating (“nuanced-routing” flag). The gate learns to route tokens to the manifold experts most suited to their geometry.
* **Geometric Manifolds:** Supports 64 experts over 8 geometric types (euclid, hyperbolic, spherical, Poincaré, simplex, complex, Lorentzian, product). Each expert is a Transformer branch with its own parameters.
* **Memory Module:** A separate working memory (256 slots × 2048 width) is incorporated and fused via attention. This allows long-term context retention and “memory attention fusion.”
* **Normalization:** RMSNorm layers are used throughout (no mean-centering) to improve stability.
* **Gradient Scaling:** A **ComplexFilteredGradScaler** automatically filters complex-valued parameters during mixed-precision (AMP) training, preventing CUDA scatter/gather errors while keeping \~99% parameters in full AMP mode.

## Training Configuration

Training is controlled by a CLI script (`integration_test_runner.py`) with many flags that directly affect the model and data pipeline. Key options include:

* `--hidden-dim`, `--experts-num`, `--k-experts`, `--num-heads`, `--learning-rate` – Set the core model architecture and optimizer.  For example, `--hidden-dim 128 --experts-num 16 --k-experts 4 --num-heads 12` in the test run.
* `--seq-len` – Maximum sequence length for training.
* `--dataset-<subset>` – Choose training subsets. Available subsets include **chain-of-thought**, **reasoning**, and **conversational** data. (In integration tests, `--dataset-conversational` was used; other runs use `--dataset-reasoning` or `--dataset-cot` accordingly.) These flags switch which portion of the streaming dataset is loaded.
* `--amp-off` / `--flash-attention-off` – Disable automatic mixed precision or FlashAttention optimizations, respectively. By default, AMP is on (filtered by the custom scaler) and FlashAttention is used if available.
* `--skip-setup` – Skip initial calibration steps (e.g. curvature calibration) and resume an existing run.
* `--nuanced-routing` – Enable the curvature-aware (nuanced) routing algorithm in the gate. This influences how the gate balances experts, especially across different geometries.
* `--sample-size`, `--validation-steps`, `--stage-steps` – Control the length of training and validation runs (used mainly for debugging/integration tests).

These flags are parsed by `integration_test_runner.py` and passed to `train_geometric_model_v1.py` and the dataset loader. For instance, setting `--experts-num M --k-experts K` makes the model instantiate *M* experts and use top-*K* selection, while `--dataset-cot` would trigger the streaming loader to fetch chain-of-thought examples.  The dataset is loaded via a streaming loader (in `streaming_dataset_loader.py`), which assembles the selected subsets on-the-fly.

## Training Data and Procedure

The final checkpoint **model\_5** was trained on a *curated set* of about **80,000 text samples** drawn from diverse reasoning-focused data: chain-of-thought problems, logical reasoning puzzles, and conversational prompts.  This mixes formal reasoning text with dialogue-like inputs to improve both inference and chit-chat skills.  Training proceeded in stages:

1. **Curvature Calibration:** Initialize and calibrate the geometric manifolds (e.g. adjust curvature parameters for hyperbolic/other spaces).
2. **Reasoning Warmup:** A brief warmup on reasoning tasks to establish initial patterns.
3. **Branch Pretrain:** Train each expert branch (manifold network) on its data subset to specialize its representations.
4. **Gate Training:** Train the gating network (with branches fixed) to optimize routing decisions.
5. **Joint Finetune:** End-to-end fine-tuning of the full model (all experts + gate) on the entire dataset for final integration.

During training, **flash attention** is enabled by default for efficiency, and AMP is on (with the custom scaler handling complex tensors). Checkpoints are saved at each stage (see the files in `model_5`), with the final full-model weights in `mgm_geometric_model_final.pth`.

## Final Model Details

The `model_5` checkpoint corresponds to a modestly-sized MGM instance (hidden dimension 1600). With these settings, the total parameter count is on the order of a few hundred million.  The architecture includes multiple attention layers (num-heads as set above), *M* experts of size 1600 each, plus memory and gating parameters.  The **architecture properties** (hidden size, experts, heads) can be inferred from the training flags: e.g. using 16 experts with K=8 (top-8 routing), 12 heads, etc., would yield a specific param count.

Because of the MoE structure, the effective “parallel” capacity is large even with a moderate hidden size.  In particular, the 256×2048 memory adds \~500K parameters, each Transformer layer has \~O(4\*hidden²) params, and each expert branch replicates these. For instance, 16 experts of hidden 1600, with typical feed-forward factor 4, yields roughly **hundreds of millions** of weights. (Exact count depends on number of layers and experts.)

## Novel Components and Experimental Features

* **Geometric Mixture-of-Experts:** MGM’s core novelty is routing inputs across manifold experts. This *mixture-of-geometries* lets the model reason in different latent spaces. The gate uses a **curvature-aware routing** (“nuanced-routing”) strategy to balance experts from different manifolds.
* **Working Memory Fusion:** The 256-slot memory provides a persistent context vector. At each layer, attention is computed jointly over the token embeddings and the memory slots (a “memory-attention fusion”), allowing long-range information flow beyond the fixed context window.
* **ComplexGradient Scaling:** Training uses a custom **ComplexFilteredGradScaler** that automatically skips complex-valued parameters when scaling gradients. This eliminates rare CUDA scatter/gather errors while still benefiting from AMP for the rest of the model.
* **Streaming Data Pipeline:** The model was trained using a streaming loader to handle mixed datasets without large local storage. This enabled seamless mixing of multiple text sources (chain-of-thought, reasoning, dialogue).

## Usage

Once uploaded to Hugging Face, the model can be used like any Transformers model. For example:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("doodle-med/MGM")
model = AutoModelForCausalLM.from_pretrained("doodle-med/MGM", subfolder="model_5")

inputs = tokenizer("Q: Why is the sky blue? A:", return_tensors="pt")
output_ids = model.generate(**inputs)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

This will invoke the `model_5/mgm_geometric_model_final.pth` weights. Adjust generation parameters as needed for your application.

**Sources:** The architecture and training details are described in the MGM project repository and PR documentation. These explain the geometric expert design, memory module, gradient-scaler, and the CLI flags used for training.
