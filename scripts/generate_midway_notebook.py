from pathlib import Path
import sys

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "scifact_midway.ipynb"


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# SciFact Midway Experiments\n\n"
            "This notebook reproduces the midway-stage experiments for the project "
            "**Retrieval-Based Scientific Claim Verification with SciFact**. "
            "It compares a lexical baseline against a stronger retrieval + encoder pipeline, "
            "and it is designed to run either on CPU for smoke tests or on GPU for the real midway experiments."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Experiment Setup\n\n"
            "- Dataset: SciFact (`corpus.jsonl`, `claims_train.jsonl`, `claims_dev.jsonl`)\n"
            "- Baseline retriever: TF-IDF cosine retrieval\n"
            "- Stronger retriever presets: `all-MiniLM-L6-v2` for CPU quick runs, `all-mpnet-base-v2` for local CUDA/GPU runs\n"
            "- Baseline verifier: TF-IDF + multinomial logistic regression\n"
            "- Stronger verifier presets: lightweight BERT for CPU quick runs, `SciBERT` or `PubMedBERT` for GPU runs\n"
            "- Main outputs: retrieval metrics, oracle document classification metrics, and end-to-end joint evidence hit"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Recommended Presets\n\n"
            "- `cpu_quick`: sanity check on a local CPU\n"
            "- `gpu_midway`: recommended for the actual midway report on a local CUDA/GPU machine\n"
            "- `gpu_strong`: optional stronger run if you have extra GPU time"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "PROJECT_ROOT = Path.cwd().resolve()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()\n"
            "SRC_ROOT = PROJECT_ROOT / 'src'\n"
            "if str(SRC_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC_ROOT))\n\n"
            "from scifact_midway import build_experiment_config, run_all_experiments\n\n"
            "RESULTS_DIR = PROJECT_ROOT / 'results' / 'midway'\n"
            "SUMMARY_PATH = RESULTS_DIR / 'summary.json'\n"
            "PROJECT_ROOT"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "PRESET = 'gpu_midway'\n"
            "RUN_FULL_EXPERIMENT = False\n"
            "RESULTS_DIR = PROJECT_ROOT / 'results' / PRESET\n"
            "SUMMARY_PATH = RESULTS_DIR / 'summary.json'\n\n"
            "if RUN_FULL_EXPERIMENT or not SUMMARY_PATH.exists():\n"
            "    config = build_experiment_config(PRESET)\n"
            "    results = run_all_experiments(project_root=PROJECT_ROOT, output_dir=RESULTS_DIR, config=config)\n"
            "else:\n"
            "    results = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))\n\n"
            "results.keys()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Dataset Summary")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "dataset_summary = results['dataset_summary']\n"
            "pd.Series(dataset_summary)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Retrieval Results")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "retrieval_df = pd.read_csv(RESULTS_DIR / 'retrieval_metrics.csv', index_col=0)\n"
            "retrieval_df.round(4)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'plots' / 'retrieval_recall_curve.png')))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Oracle Document Classification")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "classification_df = pd.read_csv(RESULTS_DIR / 'classification_metrics.csv', index_col=0)\n"
            "classification_df.round(4)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "display(Image(filename=str(RESULTS_DIR / 'plots' / 'logreg_confusion_matrix.png')))\n"
            "display(Image(filename=str(RESULTS_DIR / 'plots' / 'bert_tiny_confusion_matrix.png')))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## End-to-End Pipeline Results")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "pipeline_df = pd.read_csv(RESULTS_DIR / 'pipeline_metrics.csv', index_col=0)\n"
            "pipeline_df.round(4)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Quick Takeaways")
    )

    cells.append(
        nbf.v4.new_code_cell(
            "print('Dense retrieval should improve retrieval quality clearly over TF-IDF.')\n"
            "print('A scientific encoder such as SciBERT should improve macro-F1 and especially the REFUTES class compared with logistic regression.')\n"
            "print('Any remaining end-to-end gap is useful midway material because it leaves clear room for the final report.')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Optional: Inspect Predictions\n\n"
            "These files can be useful when writing the Results section or doing error analysis."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "bert_pipeline_preds = pd.read_csv(RESULTS_DIR / 'dense_plus_bert_pipeline_predictions.csv')\n"
            "bert_pipeline_preds.head()"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    }
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)


if __name__ == "__main__":
    main()
