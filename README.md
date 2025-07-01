# MGM-model_1
Mixture of geometric experts - novel LLM architecture

## Integration Test Runner

To run the automated integration test use `integration_test_runner.py`. The validation phase is controlled with the `--validation-steps` flag. The example below runs a short test with 50 validation batches.

```bash
export WANDB_API_KEY="<your-wandb-api-key>"
export HF_TOKEN="<your-huggingface-token>"
export GIT_SECRET="<your-github-token>"
cd mgm_project/scripts/6-10-25
python3 integration_test_runner.py --stage-steps 10 \
    --hidden-dim 128 --seq-len 256 --experts-num 16 --k-experts 4 \
    --num-heads 12 --learning-rate 3e-4 --dataset-conversational \
    --amp-off --flash-attention-off --skip-setup \
    --sample-size 50 --validation-steps 50 --nuanced-routing
```

### Quick diversity check
Run a minimal calibration with 8 experts to verify manifold specialization:

```bash
python3 integration_test_runner.py --stage-steps 1 \
    --hidden-dim 64 --seq-len 32 --experts-num 8 --k-experts 2 \
    --num-heads 4 --learning-rate 3e-4 --dataset-conversational \
    --amp-off --flash-attention-off --skip-setup \
    --sample-size 10 --validation-steps 1 --nuanced-routing
```
The final curvature analysis should report **EXCELLENT expert diversity** and multi-manifold specialization.
