import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from scifact_midway.pipeline import (
    DEFAULT_SEED,
    ID_TO_LABEL,
    LABEL_TO_ID,
    DenseRetriever,
    ExperimentConfig,
    LexicalVerifier,
    PairExample,
    TfidfRetriever,
    TransformerVerifier,
    build_pair_examples,
    corpus_texts_and_ids,
    doc_text,
    ensure_dir,
    evaluate_classifier,
    evaluate_end_to_end,
    evaluate_retrieval,
    get_device,
    gold_doc_labels,
    load_scifact,
    set_seed,
)


PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"


class TfidfSentenceSelector:
    """Unsupervised sentence selector used as a targeted evidence stage."""

    def __init__(
        self,
        corpus_by_id: Dict[int, Dict[str, Any]],
        train_claims: Sequence[Dict[str, Any]],
        max_features: int = 100000,
    ) -> None:
        self.corpus_by_id = corpus_by_id
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        fit_texts: List[str] = []
        fit_texts.extend(record["claim"] for record in train_claims)
        for record in corpus_by_id.values():
            title = str(record.get("title", "")).strip()
            if title:
                fit_texts.append(title)
            fit_texts.extend(sentence.strip() for sentence in record.get("abstract", []) if sentence.strip())
        self.vectorizer.fit(fit_texts)
        self._doc_cache: Dict[int, Tuple[List[int], List[str], Any]] = {}

    def _doc_sentences(self, record: Dict[str, Any]) -> Tuple[List[int], List[str], Any]:
        doc_id = int(record["doc_id"])
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        indexed_sentences = [
            (index, sentence.strip())
            for index, sentence in enumerate(record.get("abstract", []))
            if sentence.strip()
        ]
        if not indexed_sentences:
            result = ([], [], self.vectorizer.transform([""]))
            self._doc_cache[doc_id] = result
            return result
        indices = [index for index, _ in indexed_sentences]
        sentences = [sentence for _, sentence in indexed_sentences]
        matrix = self.vectorizer.transform(sentences)
        result = (indices, sentences, matrix)
        self._doc_cache[doc_id] = result
        return result

    def rank_sentences(self, claim: str, record: Dict[str, Any]) -> List[Tuple[int, str, float]]:
        indices, sentences, sentence_matrix = self._doc_sentences(record)
        if not indices:
            return []
        claim_vector = self.vectorizer.transform([claim])
        scores = (sentence_matrix @ claim_vector.T).toarray().ravel()
        order = np.argsort(scores)[::-1]
        return [(indices[position], sentences[position], float(scores[position])) for position in order]

    def selected_indices(self, claim: str, record: Dict[str, Any], top_n: int = 2) -> List[int]:
        return [index for index, _, _ in self.rank_sentences(claim, record)[:top_n]]

    def selected_text(self, claim: str, record: Dict[str, Any], top_n: int = 2) -> str:
        ranked = self.rank_sentences(claim, record)[:top_n]
        selected = sorted(ranked, key=lambda item: item[0])
        pieces: List[str] = []
        title = str(record.get("title", "")).strip()
        if title:
            pieces.append(title)
        pieces.extend(sentence for _, sentence, _ in selected)
        return " ".join(pieces).strip() or doc_text(record)


def get_gold_rationale_indices(claim_record: Dict[str, Any], doc_id: int) -> List[int]:
    evidence_sets = claim_record.get("evidence", {}).get(str(doc_id), [])
    indices = {
        int(sentence_index)
        for evidence_set in evidence_sets
        for sentence_index in evidence_set.get("sentences", [])
    }
    return sorted(indices)


def transform_examples_with_selector(
    examples: Sequence[PairExample],
    corpus_by_id: Dict[int, Dict[str, Any]],
    selector: TfidfSentenceSelector,
    top_n: int,
) -> List[PairExample]:
    transformed: List[PairExample] = []
    for example in examples:
        record = corpus_by_id[int(example.doc_id)]
        transformed.append(
            PairExample(
                claim_id=example.claim_id,
                doc_id=example.doc_id,
                claim=example.claim,
                abstract=selector.selected_text(example.claim, record, top_n=top_n),
                label=example.label,
            )
        )
    return transformed


def load_or_train_transformer(
    *,
    model_name: str,
    train_examples: Sequence[PairExample],
    output_dir: Path,
    device: Any,
    use_amp: bool,
    config: ExperimentConfig,
    force_train: bool,
) -> Tuple[TransformerVerifier, Dict[str, Any]]:
    config_path = output_dir / "config.json"
    tokenizer_path = output_dir / "tokenizer_config.json"
    history_path = output_dir / "train_history.json"
    if config_path.exists() and tokenizer_path.exists() and not force_train:
        train_info: Dict[str, Any] = {"loaded_existing_checkpoint": True}
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            train_info["history"] = history
            if history:
                train_info["best_valid_macro_f1"] = max(item["valid_macro_f1"] for item in history)
        verifier = TransformerVerifier(
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            use_amp=use_amp,
            max_length=config.max_length,
        )
        return verifier, train_info

    verifier, train_info = TransformerVerifier.train(
        train_examples=train_examples,
        output_dir=output_dir,
        device=device,
        model_name=model_name,
        epochs=config.epochs,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_length=config.max_length,
        use_amp=use_amp,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_info["loaded_existing_checkpoint"] = False
    return verifier, train_info


def evaluate_end_to_end_with_selector(
    claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    retrieved_doc_ids: Sequence[Sequence[int]],
    verifier: Any,
    selector: TfidfSentenceSelector,
    name: str,
    output_dir: Path,
    eval_batch_size: int,
    top_n: int,
    ks: Sequence[int] = (3, 5),
) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {f"joint_hit@{k}": [] for k in ks}
    rows: List[Dict[str, Any]] = []
    max_k = max(ks)
    for claim_record, retrieved in zip(claims, retrieved_doc_ids):
        gold = gold_doc_labels(claim_record)
        if not gold:
            continue
        pair_examples = []
        selected_sentence_indices: List[List[int]] = []
        for doc_id in retrieved[:max_k]:
            doc_id = int(doc_id)
            if doc_id not in corpus_by_id:
                continue
            record = corpus_by_id[doc_id]
            selected_sentence_indices.append(selector.selected_indices(claim_record["claim"], record, top_n=top_n))
            pair_examples.append(
                PairExample(
                    claim_id=int(claim_record["id"]),
                    doc_id=doc_id,
                    claim=claim_record["claim"],
                    abstract=selector.selected_text(claim_record["claim"], record, top_n=top_n),
                    label=gold.get(doc_id, "NOINFO"),
                )
            )
        probabilities = verifier.predict_proba(pair_examples, batch_size=eval_batch_size)
        predicted_labels = [ID_TO_LABEL[int(label_id)] for label_id in probabilities.argmax(axis=1)]
        for k in ks:
            success = any(
                example.doc_id in gold and predicted_labels[index] == gold[int(example.doc_id)]
                for index, example in enumerate(pair_examples[:k])
            )
            metrics[f"joint_hit@{k}"].append(float(success))
        rows.append(
            {
                "claim_id": int(claim_record["id"]),
                "claim": claim_record["claim"],
                "gold_doc_labels": json.dumps(gold, sort_keys=True),
                "retrieved_doc_ids": json.dumps([int(doc_id) for doc_id in retrieved[:max_k]]),
                "selected_sentence_indices": json.dumps(selected_sentence_indices),
                "predicted_labels": json.dumps(predicted_labels[:max_k]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{name}_pipeline_predictions.csv", index=False)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def evaluate_rationale_selection(
    claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    selector: TfidfSentenceSelector,
    output_dir: Path,
    top_ns: Sequence[int] = (1, 2, 3),
) -> Tuple[Dict[str, float], pd.DataFrame]:
    ensure_dir(output_dir)
    rows: List[Dict[str, Any]] = []
    for claim_record in claims:
        claim = claim_record["claim"]
        for doc_id in gold_doc_labels(claim_record).keys():
            if doc_id not in corpus_by_id:
                continue
            gold_sentences = get_gold_rationale_indices(claim_record, doc_id)
            if not gold_sentences:
                continue
            ranked = selector.rank_sentences(claim, corpus_by_id[doc_id])
            ranked_indices = [index for index, _, _ in ranked]
            row: Dict[str, Any] = {
                "claim_id": int(claim_record["id"]),
                "doc_id": int(doc_id),
                "claim": claim,
                "gold_sentence_indices": json.dumps(gold_sentences),
                "ranked_sentence_indices": json.dumps(ranked_indices[: max(top_ns)]),
                "gold_sentence_count": len(gold_sentences),
            }
            first_gold_rank = next(
                (rank for rank, sentence_index in enumerate(ranked_indices, start=1) if sentence_index in gold_sentences),
                0,
            )
            row["first_gold_rank"] = first_gold_rank
            row["reciprocal_rank"] = 0.0 if first_gold_rank == 0 else 1.0 / first_gold_rank
            gold_set = set(gold_sentences)
            for k in top_ns:
                selected = set(ranked_indices[:k])
                row[f"hit@{k}"] = float(bool(selected & gold_set))
                row[f"coverage@{k}"] = float(len(selected & gold_set) / len(gold_set))
            rows.append(row)

    details = pd.DataFrame(rows)
    details.to_csv(output_dir / "rationale_selection_doc_pairs.csv", index=False)
    metrics: Dict[str, float] = {"n_gold_doc_pairs": float(len(details))}
    if not details.empty:
        metrics["mrr"] = float(details["reciprocal_rank"].mean())
        for k in top_ns:
            metrics[f"rationale_hit@{k}"] = float(details[f"hit@{k}"].mean())
            metrics[f"rationale_coverage@{k}"] = float(details[f"coverage@{k}"].mean())
    pd.DataFrame([metrics]).round(4).to_csv(output_dir / "rationale_selection_metrics.csv", index=False)
    return metrics, details


def analyze_pipeline_errors(
    claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    retrieved_doc_ids: Sequence[Sequence[int]],
    verifier: Any,
    output_dir: Path,
    eval_batch_size: int,
    ks: Sequence[int] = (3, 5, 10),
    evidence_text_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    max_k = max(ks)
    claim_rows: List[Dict[str, Any]] = []
    doc_rows: List[Dict[str, Any]] = []
    for claim_record, retrieved in zip(claims, retrieved_doc_ids):
        gold = gold_doc_labels(claim_record)
        if not gold:
            continue
        pair_examples: List[PairExample] = []
        for doc_id in retrieved[:max_k]:
            doc_id = int(doc_id)
            if doc_id not in corpus_by_id:
                continue
            record = corpus_by_id[doc_id]
            evidence_text = (
                evidence_text_fn(claim_record["claim"], record)
                if evidence_text_fn is not None
                else doc_text(record)
            )
            pair_examples.append(
                PairExample(
                    claim_id=int(claim_record["id"]),
                    doc_id=doc_id,
                    claim=claim_record["claim"],
                    abstract=evidence_text,
                    label=gold.get(doc_id, "NOINFO"),
                )
            )
        probabilities = verifier.predict_proba(pair_examples, batch_size=eval_batch_size)
        predicted_labels = [ID_TO_LABEL[int(label_id)] for label_id in probabilities.argmax(axis=1)]
        pred_by_doc = {int(example.doc_id): predicted_labels[index] for index, example in enumerate(pair_examples)}
        best_gold_rank = next(
            (rank for rank, doc_id in enumerate(retrieved[:max_k], start=1) if int(doc_id) in gold),
            0,
        )
        gold_label_group = "+".join(sorted(set(gold.values())))
        claim_row: Dict[str, Any] = {
            "claim_id": int(claim_record["id"]),
            "claim": claim_record["claim"],
            "gold_doc_labels": json.dumps(gold, sort_keys=True),
            "gold_label_group": gold_label_group,
            "best_gold_rank_within_top10": best_gold_rank,
            "retrieved_top10": json.dumps([int(doc_id) for doc_id in retrieved[:max_k]]),
            "predicted_top10_labels": json.dumps(predicted_labels[:max_k]),
        }
        for k in ks:
            retrieved_gold = [int(doc_id) for doc_id in retrieved[:k] if int(doc_id) in gold]
            correct_gold = [doc_id for doc_id in retrieved_gold if pred_by_doc.get(doc_id) == gold[doc_id]]
            if correct_gold:
                status = "success"
            elif not retrieved_gold:
                status = "retrieval_miss"
            else:
                status = "stance_error"
            claim_row[f"status@{k}"] = status
            claim_row[f"retrieved_gold_docs@{k}"] = json.dumps(retrieved_gold)
            claim_row[f"correct_gold_docs@{k}"] = json.dumps(correct_gold)
        claim_rows.append(claim_row)

        for rank, (example, predicted_label, proba) in enumerate(
            zip(pair_examples, predicted_labels, probabilities),
            start=1,
        ):
            gold_label = gold.get(int(example.doc_id), "NOINFO")
            doc_rows.append(
                {
                    "claim_id": int(claim_record["id"]),
                    "rank": rank,
                    "doc_id": int(example.doc_id),
                    "is_gold_doc": int(example.doc_id) in gold,
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "is_correct_gold_label": int(example.doc_id) in gold and predicted_label == gold_label,
                    "prob_noinfo": float(proba[LABEL_TO_ID["NOINFO"]]),
                    "prob_supports": float(proba[LABEL_TO_ID["SUPPORTS"]]),
                    "prob_refutes": float(proba[LABEL_TO_ID["REFUTES"]]),
                    "claim": claim_record["claim"],
                }
            )

    claims_df = pd.DataFrame(claim_rows)
    docs_df = pd.DataFrame(doc_rows)
    claims_df.to_csv(output_dir / "error_analysis_claims.csv", index=False)
    docs_df.to_csv(output_dir / "pipeline_document_predictions.csv", index=False)

    breakdown_rows: List[Dict[str, Any]] = []
    for k in ks:
        counts = claims_df[f"status@{k}"].value_counts()
        total = float(len(claims_df))
        for status in ["success", "retrieval_miss", "stance_error"]:
            count = int(counts.get(status, 0))
            breakdown_rows.append(
                {
                    "k": int(k),
                    "status": status,
                    "count": count,
                    "proportion": 0.0 if total == 0 else count / total,
                }
            )
    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df.to_csv(output_dir / "error_breakdown.csv", index=False)

    label_breakdown = (
        claims_df.groupby(["gold_label_group", "status@5"]).size().reset_index(name="count")
        if not claims_df.empty
        else pd.DataFrame(columns=["gold_label_group", "status@5", "count"])
    )
    label_breakdown.to_csv(output_dir / "error_breakdown_by_label_top5.csv", index=False)
    stance_errors = docs_df[(docs_df["is_gold_doc"]) & ~(docs_df["is_correct_gold_label"])]
    stance_errors.to_csv(output_dir / "stance_error_gold_docs.csv", index=False)
    return claims_df, docs_df, breakdown_df


def plot_bar_metric(
    df: pd.DataFrame,
    label_column: str,
    metric_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plot_df = df.dropna(subset=[metric_column]).copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(plot_df[label_column], plot_df[metric_column])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, max(1.0, float(plot_df[metric_column].max()) * 1.15 if not plot_df.empty else 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_final_summary(
    output_path: Path,
    retrieval_results: Dict[str, Dict[str, float]],
    rationale_metrics: Dict[str, float],
    model_comparison: pd.DataFrame,
    pipeline_comparison: pd.DataFrame,
    error_breakdown: pd.DataFrame,
    train_info: Dict[str, Any],
    runtime_stats: Dict[str, float],
) -> None:
    def metric(df: pd.DataFrame, row_name: str, column: str) -> Optional[float]:
        row = df[df["system"] == row_name]
        if row.empty or column not in row:
            return None
        value = row.iloc[0][column]
        return None if pd.isna(value) else float(value)

    lines = [
        "# Final Experiment Summary",
        "",
        "## Retrieval",
        f"- TF-IDF: R@5={retrieval_results['tfidf']['recall@5']:.4f}, R@10={retrieval_results['tfidf']['recall@10']:.4f}, MRR={retrieval_results['tfidf']['mrr']:.4f}",
        f"- Dense MPNet: R@5={retrieval_results['dense']['recall@5']:.4f}, R@10={retrieval_results['dense']['recall@10']:.4f}, MRR={retrieval_results['dense']['mrr']:.4f}",
        "",
        "## Error Analysis",
    ]
    for _, row in error_breakdown[error_breakdown["k"] == 5].iterrows():
        lines.append(f"- Top-5 {row['status']}: {int(row['count'])} claims ({row['proportion']:.1%})")
    lines.extend(
        [
            "",
            "## Sentence Evidence Selection",
            f"- Rationale Hit@1={rationale_metrics.get('rationale_hit@1', 0.0):.4f}",
            f"- Rationale Hit@2={rationale_metrics.get('rationale_hit@2', 0.0):.4f}",
            f"- Rationale Coverage@2={rationale_metrics.get('rationale_coverage@2', 0.0):.4f}",
            "",
            "## Oracle Verification",
        ]
    )
    for _, row in model_comparison.iterrows():
        lines.append(
            f"- {row['system']}: Accuracy={row['accuracy']:.4f}, Macro-F1={row['macro_f1']:.4f}, "
            f"F1(Supports)={row['f1_supports']:.4f}, F1(Refutes)={row['f1_refutes']:.4f}"
        )
    lines.append("")
    lines.append("## Dense End-to-End Pipeline")
    for _, row in pipeline_comparison.iterrows():
        if row["retriever"] != "dense":
            continue
        lines.append(
            f"- {row['system']}: JointHit@3={row['joint_hit@3']:.4f}, JointHit@5={row['joint_hit@5']:.4f}"
        )
    lines.append("")
    lines.append("## Training Notes")
    for name, info in train_info.items():
        loaded = info.get("loaded_existing_checkpoint", False)
        best_f1 = info.get("best_valid_macro_f1")
        if best_f1 is None:
            lines.append(f"- {name}: loaded_existing={loaded}")
        else:
            lines.append(f"- {name}: loaded_existing={loaded}, best internal dev Macro-F1={best_f1:.4f}")
    lines.append("")
    lines.append("## Runtime")
    for name, value in runtime_stats.items():
        lines.append(f"- {name}: {value:.2f} seconds")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final SciFact experiments.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "final_experiments"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "data"))
    parser.add_argument("--base-results-dir", default=str(PROJECT_ROOT / "results" / "midway_local_gpu_10ep"))
    parser.add_argument("--sentence-top-n", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--retriever-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--negatives-per-claim", type=int, default=4)
    parser.add_argument("--retrieval-pool-k", type=int, default=30)
    parser.add_argument("--prefer-device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-pubmedbert", action="store_true")
    parser.add_argument("--skip-sentence-verifier", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    base_results_dir = Path(args.base_results_dir).resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "plots")
    ensure_dir(output_dir / "models")
    ensure_dir(output_dir / "error_analysis")
    ensure_dir(output_dir / "rationale_selection")

    config = ExperimentConfig(
        preset="final",
        retriever_model_name="sentence-transformers/all-mpnet-base-v2",
        verifier_model_name=SCIBERT_MODEL,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        retriever_batch_size=args.retriever_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_length=args.max_length,
        negatives_per_claim=args.negatives_per_claim,
        retrieval_pool_k=args.retrieval_pool_k,
        retrieval_top_k=10,
        retrieval_ks=(1, 3, 5, 10),
        pipeline_ks=(3, 5),
        prefer_device=args.prefer_device,
        use_amp=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    runtime_stats: Dict[str, float] = {}
    train_info: Dict[str, Any] = {}
    device = get_device(config.prefer_device)
    use_amp = config.use_amp and device.type == "cuda"

    start = time.perf_counter()
    corpus_by_id, train_claims, dev_claims = load_scifact(data_dir)
    doc_ids, corpus_texts = corpus_texts_and_ids(corpus_by_id)
    runtime_stats["load_data"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_retriever = TfidfRetriever.fit(corpus_texts, doc_ids)
    train_examples = build_pair_examples(
        train_claims,
        corpus_by_id,
        tfidf_retriever,
        negatives_per_claim=config.negatives_per_claim,
        retrieval_pool_k=config.retrieval_pool_k,
        seed=config.seed,
    )
    dev_examples = build_pair_examples(
        dev_claims,
        corpus_by_id,
        tfidf_retriever,
        negatives_per_claim=config.negatives_per_claim,
        retrieval_pool_k=config.retrieval_pool_k,
        seed=config.seed,
    )
    runtime_stats["build_tfidf_and_pairs"] = time.perf_counter() - start

    start = time.perf_counter()
    lexical_verifier = LexicalVerifier()
    lexical_verifier.fit(train_examples)
    runtime_stats["train_logreg"] = time.perf_counter() - start

    start = time.perf_counter()
    dense_retriever = DenseRetriever.fit(
        corpus_texts=corpus_texts,
        doc_ids=doc_ids,
        model_name=config.retriever_model_name,
        cache_dir=base_results_dir / "cache" / "retriever_embeddings",
        device=device,
        use_amp=use_amp,
        batch_size=config.retriever_batch_size,
        max_length=config.max_length,
    )
    claim_texts = [claim_record["claim"] for claim_record in dev_claims]
    tfidf_dev_retrieved = tfidf_retriever.batch_search(claim_texts, k=config.retrieval_top_k)
    dense_dev_retrieved = dense_retriever.batch_search(
        claim_texts,
        k=config.retrieval_top_k,
        batch_size=config.retriever_batch_size,
    )
    retrieval_results = {
        "tfidf": evaluate_retrieval(dev_claims, tfidf_dev_retrieved, ks=config.retrieval_ks),
        "dense": evaluate_retrieval(dev_claims, dense_dev_retrieved, ks=config.retrieval_ks),
    }
    pd.DataFrame.from_dict(retrieval_results, orient="index").round(4).to_csv(output_dir / "retrieval_metrics.csv")
    runtime_stats["retrieval"] = time.perf_counter() - start

    start = time.perf_counter()
    scibert_model_dir = base_results_dir / "models" / "transformer_verifier"
    scibert_verifier = TransformerVerifier(
        model_name=SCIBERT_MODEL,
        output_dir=scibert_model_dir,
        device=device,
        use_amp=use_amp,
        max_length=config.max_length,
    )
    scibert_history_path = scibert_model_dir / "train_history.json"
    train_info["scibert_abstract"] = {"loaded_existing_checkpoint": True}
    if scibert_history_path.exists():
        scibert_history = json.loads(scibert_history_path.read_text(encoding="utf-8"))
        train_info["scibert_abstract"]["history"] = scibert_history
        train_info["scibert_abstract"]["best_valid_macro_f1"] = max(
            item["valid_macro_f1"] for item in scibert_history
        )
    runtime_stats["load_scibert"] = time.perf_counter() - start

    start = time.perf_counter()
    model_metrics: Dict[str, Dict[str, Any]] = {}
    pipeline_metrics: Dict[str, Dict[str, Any]] = {}
    model_metrics["logreg_abstract"] = evaluate_classifier(
        "logreg_abstract",
        lexical_verifier,
        dev_examples,
        output_dir / "plots",
        eval_batch_size=config.eval_batch_size,
    )
    model_metrics["scibert_abstract"] = evaluate_classifier(
        "scibert_abstract",
        scibert_verifier,
        dev_examples,
        output_dir / "plots",
        eval_batch_size=config.eval_batch_size,
    )
    pipeline_metrics["tfidf_plus_logreg"] = evaluate_end_to_end(
        dev_claims,
        corpus_by_id,
        tfidf_dev_retrieved,
        lexical_verifier,
        "tfidf_plus_logreg",
        output_dir,
        eval_batch_size=config.eval_batch_size,
        ks=config.pipeline_ks,
    )
    pipeline_metrics["dense_plus_logreg"] = evaluate_end_to_end(
        dev_claims,
        corpus_by_id,
        dense_dev_retrieved,
        lexical_verifier,
        "dense_plus_logreg",
        output_dir,
        eval_batch_size=config.eval_batch_size,
        ks=config.pipeline_ks,
    )
    pipeline_metrics["dense_plus_scibert_abstract"] = evaluate_end_to_end(
        dev_claims,
        corpus_by_id,
        dense_dev_retrieved,
        scibert_verifier,
        "dense_plus_scibert_abstract",
        output_dir,
        eval_batch_size=config.eval_batch_size,
        ks=config.pipeline_ks,
    )
    runtime_stats["baseline_evaluation"] = time.perf_counter() - start

    start = time.perf_counter()
    _, _, error_breakdown = analyze_pipeline_errors(
        claims=dev_claims,
        corpus_by_id=corpus_by_id,
        retrieved_doc_ids=dense_dev_retrieved,
        verifier=scibert_verifier,
        output_dir=output_dir / "error_analysis",
        eval_batch_size=config.eval_batch_size,
        ks=(3, 5, 10),
    )
    runtime_stats["error_analysis"] = time.perf_counter() - start

    start = time.perf_counter()
    selector = TfidfSentenceSelector(corpus_by_id=corpus_by_id, train_claims=train_claims)
    rationale_metrics, _ = evaluate_rationale_selection(
        claims=dev_claims,
        corpus_by_id=corpus_by_id,
        selector=selector,
        output_dir=output_dir / "rationale_selection",
        top_ns=(1, 2, 3),
    )
    selected_train_examples = transform_examples_with_selector(
        train_examples,
        corpus_by_id,
        selector,
        top_n=args.sentence_top_n,
    )
    selected_dev_examples = transform_examples_with_selector(
        dev_examples,
        corpus_by_id,
        selector,
        top_n=args.sentence_top_n,
    )
    runtime_stats["sentence_selection"] = time.perf_counter() - start

    if not args.skip_sentence_verifier:
        start = time.perf_counter()
        selected_config = config
        selected_verifier, selected_train_info = load_or_train_transformer(
            model_name=SCIBERT_MODEL,
            train_examples=selected_train_examples,
            output_dir=output_dir / "models" / f"scibert_sentence_top{args.sentence_top_n}",
            device=device,
            use_amp=use_amp,
            config=selected_config,
            force_train=args.force_train,
        )
        train_info[f"scibert_sentence_top{args.sentence_top_n}"] = selected_train_info
        runtime_stats["train_or_load_sentence_scibert"] = time.perf_counter() - start

        start = time.perf_counter()
        model_metrics[f"scibert_sentence_top{args.sentence_top_n}"] = evaluate_classifier(
            f"scibert_sentence_top{args.sentence_top_n}",
            selected_verifier,
            selected_dev_examples,
            output_dir / "plots",
            eval_batch_size=config.eval_batch_size,
        )
        pipeline_metrics[f"dense_plus_scibert_sentence_top{args.sentence_top_n}"] = evaluate_end_to_end_with_selector(
            dev_claims,
            corpus_by_id,
            dense_dev_retrieved,
            selected_verifier,
            selector,
            f"dense_plus_scibert_sentence_top{args.sentence_top_n}",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            top_n=args.sentence_top_n,
            ks=config.pipeline_ks,
        )
        runtime_stats["sentence_scibert_evaluation"] = time.perf_counter() - start

    if not args.skip_pubmedbert:
        start = time.perf_counter()
        pubmedbert_verifier, pubmedbert_train_info = load_or_train_transformer(
            model_name=PUBMEDBERT_MODEL,
            train_examples=train_examples,
            output_dir=output_dir / "models" / "pubmedbert_abstract",
            device=device,
            use_amp=use_amp,
            config=config,
            force_train=args.force_train,
        )
        train_info["pubmedbert_abstract"] = pubmedbert_train_info
        runtime_stats["train_or_load_pubmedbert"] = time.perf_counter() - start

        start = time.perf_counter()
        model_metrics["pubmedbert_abstract"] = evaluate_classifier(
            "pubmedbert_abstract",
            pubmedbert_verifier,
            dev_examples,
            output_dir / "plots",
            eval_batch_size=config.eval_batch_size,
        )
        pipeline_metrics["dense_plus_pubmedbert_abstract"] = evaluate_end_to_end(
            dev_claims,
            corpus_by_id,
            dense_dev_retrieved,
            pubmedbert_verifier,
            "dense_plus_pubmedbert_abstract",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            ks=config.pipeline_ks,
        )
        runtime_stats["pubmedbert_evaluation"] = time.perf_counter() - start

    model_rows: List[Dict[str, Any]] = []
    for system, metrics in model_metrics.items():
        row = {"system": system}
        row.update(metrics)
        model_rows.append(row)
    model_comparison = pd.DataFrame(model_rows).round(4)
    model_comparison.to_csv(output_dir / "final_model_comparison.csv", index=False)

    pipeline_system_metadata = {
        "tfidf_plus_logreg": ("tfidf", "logreg", "abstract"),
        "dense_plus_logreg": ("dense", "logreg", "abstract"),
        "dense_plus_scibert_abstract": ("dense", "scibert", "abstract"),
        "dense_plus_pubmedbert_abstract": ("dense", "pubmedbert", "abstract"),
        f"dense_plus_scibert_sentence_top{args.sentence_top_n}": (
            "dense",
            "scibert",
            f"sentence_top{args.sentence_top_n}",
        ),
    }
    pipeline_rows = []
    for system, metrics in pipeline_metrics.items():
        retriever, verifier, evidence_input = pipeline_system_metadata.get(system, ("unknown", "unknown", "unknown"))
        row = {
            "system": system,
            "retriever": retriever,
            "verifier": verifier,
            "evidence_input": evidence_input,
        }
        row.update(metrics)
        pipeline_rows.append(row)
    pipeline_comparison = pd.DataFrame(pipeline_rows).round(4)
    pipeline_comparison.to_csv(output_dir / "final_pipeline_comparison.csv", index=False)

    if "macro_f1" in model_comparison:
        plot_bar_metric(
            model_comparison,
            label_column="system",
            metric_column="macro_f1",
            output_path=output_dir / "plots" / "oracle_macro_f1_comparison.png",
            title="Oracle Verification Macro-F1",
            ylabel="Macro-F1",
        )
    if "joint_hit@5" in pipeline_comparison:
        plot_bar_metric(
            pipeline_comparison,
            label_column="system",
            metric_column="joint_hit@5",
            output_path=output_dir / "plots" / "dense_pipeline_joint_hit5_comparison.png",
            title="End-to-End JointHit@5",
            ylabel="JointHit@5",
        )

    final_summary = {
        "config": asdict(config),
        "device": {
            "resolved_device": str(device),
            "amp_enabled": use_amp,
        },
        "retrieval_results": retrieval_results,
        "rationale_metrics": rationale_metrics,
        "model_metrics": model_metrics,
        "pipeline_metrics": pipeline_metrics,
        "train_info": train_info,
        "runtime_stats": runtime_stats,
    }
    (output_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    write_final_summary(
        output_path=output_dir / "final_summary.md",
        retrieval_results=retrieval_results,
        rationale_metrics=rationale_metrics,
        model_comparison=model_comparison,
        pipeline_comparison=pipeline_comparison,
        error_breakdown=error_breakdown,
        train_info=train_info,
        runtime_stats=runtime_stats,
    )
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
