# CS505 Scientific Claim Verification Project

This repository contains the code and report source for an NLP project on retrieval-based scientific claim verification with SciFact.

## Project Overview

The system uses a two-stage pipeline:

1. Retrieve candidate evidence abstracts from the SciFact corpus.
2. Predict whether each claim--abstract pair is `SUPPORTS`, `REFUTES`, or `NOINFO`.

The experiments compare TF-IDF retrieval and dense MPNet retrieval, logistic regression and transformer verifiers, sentence-level evidence selection, and PubMedBERT as a stronger biomedical verifier.

## Main Results

- Dense MPNet retrieval improves Recall@5 from 0.803 to 0.872 over TF-IDF.
- SciBERT improves oracle verification Macro-F1 from 0.359 to 0.676 over logistic regression.
- Dense + PubMedBERT is the strongest end-to-end system, with JointHit@5 = 0.564 on the SciFact development set.
- Error analysis shows that stance classification is the main remaining bottleneck: at top-5, 12.8% of evidence-bearing claims are retrieval misses and 35.1% are stance errors.

## Important Files

- `src/scifact_midway/pipeline.py`: core retrieval, training, evaluation, and plotting pipeline.
- `scripts/run_midway_experiments.py`: midway experiment runner.
- `scripts/run_final_experiments.py`: final experiment runner for error analysis, sentence selection, and PubMedBERT comparison.
- `NLP_final_report.tex`: final report LaTeX source.
- `results/final_experiments/final_summary.md`: final experiment summary.
- `results/final_experiments/final_model_comparison.csv`: oracle verifier metrics.
- `results/final_experiments/final_pipeline_comparison.csv`: end-to-end pipeline metrics.

## Reproducing Experiments

Install dependencies:

```bash
pip install -r requirements_gpu.txt
```

Run the final experiments:

```bash
python scripts/run_final_experiments.py --num-workers 0
```

The SciFact data is downloaded automatically if it is not already present under `data/data`.

## Notes

Large local artifacts are intentionally not tracked, including the virtual environment, downloaded dataset, pretrained checkpoints, model weights, and embedding caches.

