# BU SCC V100 Run Guide

## Recommended OnDemand Settings

- Interactive App: `Jupyter Notebook`
- Modules: `miniconda academic-ml/fall-2025`
- Pre-launch command: `conda activate fall-2025-pyt`
- Interface: `lab`
- Number of hours: `6` to `12`
- Number of cores: `4`
- Number of GPUs: `1`
- GPU compute capability: `7.0 (V100 or ...)`
- Project: your assigned class project account, e.g. `cs505am`

## After the Jupyter Session Starts

Open a terminal in JupyterLab and run:

```bash
cd /path/to/CS505_Project
python -m pip install -r requirements_gpu.txt
nvidia-smi
python scripts/run_midway_experiments.py --preset gpu_midway --output-dir results/midway_v100
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

## Optional Stronger Run

If you want a stronger but slower setting, run:

```bash
python scripts/run_midway_experiments.py --preset gpu_strong --output-dir results/midway_v100_strong
```

This switches the verifier to `PubMedBERT` and increases training strength.

## Main Output Files

- `results/midway_v100/summary.json`
- `results/midway_v100/midway_summary.md`
- `results/midway_v100/retrieval_metrics.csv`
- `results/midway_v100/classification_metrics.csv`
- `results/midway_v100/pipeline_metrics.csv`
- `results/midway_v100/plots/retrieval_recall_curve.png`
- `results/midway_v100/plots/transformer_confusion_matrix.png`
