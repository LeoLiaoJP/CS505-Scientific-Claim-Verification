# Local GPU Run Guide

## Local Environment Used

- OS: Windows
- Python environment: local virtual environment (`.venv-gpu`)
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- CUDA available through the local PyTorch installation

## Running the Midway Experiments

From the project root:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_midway_experiments.py --preset gpu_midway --output-dir results/midway_local_gpu_10ep
```

## Recommended Midway Configuration

The `gpu_midway` preset uses:

- Dense retriever: `sentence-transformers/all-mpnet-base-v2`
- Verifier: `allenai/scibert_scivocab_uncased`
- Batch size: `16`
- Eval batch size: `32`
- Retriever batch size: `64`
- Epochs: `10`
- Mixed precision: enabled automatically on CUDA

The code keeps the checkpoint with the best validation macro-F1, so even though training runs for 10 epochs, the final saved model is whichever epoch performs best on dev.

This is a better midway-stage setup than the earlier CPU quick run because it:

- uses a stronger sentence embedding model for retrieval
- uses a domain-relevant scientific encoder for verification
- actually trains on GPU, which is much more appropriate for a course NLP experiment

## Running the Final Experiments

The final experiments add error analysis, sentence-level evidence selection, and PubMedBERT:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_final_experiments.py --num-workers 0
```

The `--num-workers 0` option avoids Windows DataLoader multiprocessing issues.

## Main Output Files

- `results/midway_local_gpu_10ep/summary.json`
- `results/midway_local_gpu_10ep/midway_summary.md`
- `results/midway_local_gpu_10ep/retrieval_metrics.csv`
- `results/midway_local_gpu_10ep/classification_metrics.csv`
- `results/midway_local_gpu_10ep/pipeline_metrics.csv`
- `results/final_experiments/final_summary.md`
- `results/final_experiments/final_model_comparison.csv`
- `results/final_experiments/final_pipeline_comparison.csv`
