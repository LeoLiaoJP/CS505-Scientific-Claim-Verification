import json
import math
import os
import random
import tarfile
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

SCIFACT_DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABEL_TO_ID = {"NOINFO": 0, "SUPPORTS": 1, "REFUTES": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SCIFACT_TO_INTERNAL = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES"}
DEFAULT_SEED = 42


@dataclass
class PairExample:
    claim_id: int
    doc_id: int
    claim: str
    abstract: str
    label: str


@dataclass
class ExperimentConfig:
    preset: str = "cpu_quick"
    retriever_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    verifier_model_name: str = "google/bert_uncased_L-4_H-256_A-4"
    train_batch_size: int = 8
    eval_batch_size: int = 16
    retriever_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256
    negatives_per_claim: int = 3
    retrieval_pool_k: int = 20
    retrieval_top_k: int = 10
    retrieval_ks: Tuple[int, ...] = (1, 3, 5, 10)
    pipeline_ks: Tuple[int, ...] = (3, 5)
    prefer_device: str = "auto"
    use_amp: bool = True
    num_workers: int = 0
    seed: int = DEFAULT_SEED


def build_experiment_config(
    preset: str = "cpu_quick",
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    config = ExperimentConfig(preset=preset)
    if preset == "gpu_midway":
        config = ExperimentConfig(
            preset=preset,
            retriever_model_name="sentence-transformers/all-mpnet-base-v2",
            verifier_model_name="allenai/scibert_scivocab_uncased",
            train_batch_size=16,
            eval_batch_size=32,
            retriever_batch_size=64,
            gradient_accumulation_steps=1,
            epochs=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            max_length=256,
            negatives_per_claim=4,
            retrieval_pool_k=30,
            retrieval_top_k=10,
            retrieval_ks=(1, 3, 5, 10),
            pipeline_ks=(3, 5),
            prefer_device="cuda",
            use_amp=True,
            num_workers=2,
            seed=DEFAULT_SEED,
        )
    elif preset == "gpu_strong":
        config = ExperimentConfig(
            preset=preset,
            retriever_model_name="sentence-transformers/all-mpnet-base-v2",
            verifier_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            train_batch_size=12,
            eval_batch_size=24,
            retriever_batch_size=64,
            gradient_accumulation_steps=2,
            epochs=5,
            learning_rate=2e-5,
            weight_decay=0.01,
            max_length=320,
            negatives_per_claim=4,
            retrieval_pool_k=30,
            retrieval_top_k=10,
            retrieval_ks=(1, 3, 5, 10),
            pipeline_ks=(3, 5),
            prefer_device="cuda",
            use_amp=True,
            num_workers=2,
            seed=DEFAULT_SEED,
        )
    if overrides:
        valid_fields = set(ExperimentConfig.__dataclass_fields__.keys())
        for key, value in overrides.items():
            if key in valid_fields and value is not None:
                setattr(config, key, value)
    return config


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def expected_scifact_files(data_dir: Path) -> List[Path]:
    return [
        data_dir / "corpus.jsonl",
        data_dir / "claims_train.jsonl",
        data_dir / "claims_dev.jsonl",
        data_dir / "claims_test.jsonl",
    ]


def ensure_scifact_data(data_dir: Path) -> None:
    if all(path.exists() for path in expected_scifact_files(data_dir)):
        return
    ensure_dir(data_dir.parent)
    archive_path = data_dir.parent / "scifact_data.tar.gz"
    urllib.request.urlretrieve(SCIFACT_DATA_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir.parent)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def doc_text(record: Dict[str, Any]) -> str:
    abstract_text = " ".join(record.get("abstract", []))
    return f"{record.get('title', '').strip()} {abstract_text.strip()}".strip()


def load_scifact(data_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    ensure_scifact_data(data_dir)
    corpus = read_jsonl(data_dir / "corpus.jsonl")
    train_claims = read_jsonl(data_dir / "claims_train.jsonl")
    dev_claims = read_jsonl(data_dir / "claims_dev.jsonl")
    corpus_by_id = {int(record["doc_id"]): record for record in corpus}
    return corpus_by_id, train_claims, dev_claims


def gold_doc_labels(claim_record: Dict[str, Any]) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    for doc_id, evidence_sets in claim_record.get("evidence", {}).items():
        if not evidence_sets:
            continue
        labels[int(doc_id)] = SCIFACT_TO_INTERNAL[evidence_sets[0]["label"]]
    return labels


def corpus_texts_and_ids(corpus_by_id: Dict[int, Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    doc_ids = sorted(corpus_by_id.keys())
    texts = [doc_text(corpus_by_id[doc_id]) for doc_id in doc_ids]
    return doc_ids, texts


def get_device(prefer_device: str = "auto") -> torch.device:
    if prefer_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer_device)


def get_amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class TfidfRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, doc_matrix: Any, doc_ids: Sequence[int]):
        self.vectorizer = vectorizer
        self.doc_matrix = doc_matrix
        self.doc_ids = np.asarray(doc_ids)

    @classmethod
    def fit(cls, corpus_texts: Sequence[str], doc_ids: Sequence[int]) -> "TfidfRetriever":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        doc_matrix = vectorizer.fit_transform(corpus_texts)
        return cls(vectorizer, doc_matrix, doc_ids)

    def search(self, query: str, k: int) -> List[int]:
        query_vec = self.vectorizer.transform([query])
        scores = (query_vec @ self.doc_matrix.T).toarray().ravel()
        top_indices = top_k_indices(scores, k)
        return [int(self.doc_ids[index]) for index in top_indices]

    def batch_search(self, queries: Sequence[str], k: int) -> List[List[int]]:
        query_vec = self.vectorizer.transform(list(queries))
        scores = query_vec @ self.doc_matrix.T
        results: List[List[int]] = []
        for row in range(scores.shape[0]):
            dense_scores = scores[row].toarray().ravel()
            top_indices = top_k_indices(dense_scores, k)
            results.append([int(self.doc_ids[index]) for index in top_indices])
        return results


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        tokenizer: Any,
        model: Any,
        doc_ids: Sequence[int],
        doc_embeddings: np.ndarray,
        device: torch.device,
        use_amp: bool,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.doc_ids = np.asarray(doc_ids)
        self.doc_embeddings = doc_embeddings.astype(np.float32)
        self.device = device
        self.use_amp = use_amp
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def fit(
        cls,
        corpus_texts: Sequence[str],
        doc_ids: Sequence[int],
        model_name: str,
        cache_dir: Path,
        device: torch.device,
        use_amp: bool,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> "DenseRetriever":
        ensure_dir(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        embedding_path = cache_dir / "corpus_embeddings.npy"
        id_path = cache_dir / "doc_ids.json"
        if embedding_path.exists() and id_path.exists():
            cached_doc_ids = json.loads(id_path.read_text(encoding="utf-8"))
            if cached_doc_ids == [int(doc_id) for doc_id in doc_ids]:
                doc_embeddings = np.load(embedding_path)
                return cls(
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    doc_ids=doc_ids,
                    doc_embeddings=doc_embeddings,
                    device=device,
                    use_amp=use_amp,
                    max_length=max_length,
                )
        doc_embeddings = encode_texts(
            texts=list(corpus_texts),
            tokenizer=tokenizer,
            model=model,
            device=device,
            use_amp=use_amp,
            batch_size=batch_size,
            max_length=max_length,
        )
        np.save(embedding_path, doc_embeddings)
        id_path.write_text(json.dumps([int(doc_id) for doc_id in doc_ids]), encoding="utf-8")
        return cls(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            device=device,
            use_amp=use_amp,
            max_length=max_length,
        )

    def search(self, query: str, k: int) -> List[int]:
        query_embedding = encode_texts(
            texts=[query],
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            use_amp=self.use_amp,
            batch_size=1,
            max_length=self.max_length,
        )[0]
        scores = self.doc_embeddings @ query_embedding
        top_indices = top_k_indices(scores, k)
        return [int(self.doc_ids[index]) for index in top_indices]

    def batch_search(self, queries: Sequence[str], k: int, batch_size: int = 32) -> List[List[int]]:
        query_embeddings = encode_texts(
            texts=list(queries),
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            use_amp=self.use_amp,
            batch_size=batch_size,
            max_length=self.max_length,
        )
        results: List[List[int]] = []
        for query_embedding in query_embeddings:
            scores = self.doc_embeddings @ query_embedding
            top_indices = top_k_indices(scores, k)
            results.append([int(self.doc_ids[index]) for index in top_indices])
        return results


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=np.int64)
    indices = np.argpartition(scores, -k)[-k:]
    return indices[np.argsort(scores[indices])[::-1]]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    use_amp: bool,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with get_amp_context(device, enabled=use_amp):
                model_output = model(**encoded)
                pooled = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
            outputs.append(normalized.cpu().numpy())
    return np.vstack(outputs).astype(np.float32)


def build_pair_examples(
    claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    negative_retriever: TfidfRetriever,
    negatives_per_claim: int = 3,
    retrieval_pool_k: int = 20,
    seed: int = DEFAULT_SEED,
) -> List[PairExample]:
    rng = random.Random(seed)
    corpus_doc_ids = list(corpus_by_id.keys())
    examples: List[PairExample] = []
    for claim_record in claims:
        claim_id = int(claim_record["id"])
        claim_text = claim_record["claim"]
        positives = gold_doc_labels(claim_record)
        used_doc_ids = set()
        for doc_id, label in positives.items():
            if doc_id not in corpus_by_id:
                continue
            examples.append(
                PairExample(
                    claim_id=claim_id,
                    doc_id=doc_id,
                    claim=claim_text,
                    abstract=doc_text(corpus_by_id[doc_id]),
                    label=label,
                )
            )
            used_doc_ids.add(doc_id)

        negative_candidates: List[int] = []
        for doc_id in claim_record.get("cited_doc_ids", []):
            doc_id = int(doc_id)
            if doc_id in corpus_by_id and doc_id not in positives and doc_id not in negative_candidates:
                negative_candidates.append(doc_id)
        for doc_id in negative_retriever.search(claim_text, retrieval_pool_k):
            if doc_id not in positives and doc_id not in negative_candidates:
                negative_candidates.append(doc_id)
        while len(negative_candidates) < negatives_per_claim:
            sampled_doc_id = int(rng.choice(corpus_doc_ids))
            if sampled_doc_id not in positives and sampled_doc_id not in negative_candidates:
                negative_candidates.append(sampled_doc_id)
        for doc_id in negative_candidates[:negatives_per_claim]:
            if doc_id not in corpus_by_id or doc_id in used_doc_ids:
                continue
            examples.append(
                PairExample(
                    claim_id=claim_id,
                    doc_id=doc_id,
                    claim=claim_text,
                    abstract=doc_text(corpus_by_id[doc_id]),
                    label="NOINFO",
                )
            )
    return examples


def pair_to_text(example: PairExample) -> str:
    return f"claim: {example.claim} [SEP] abstract: {example.abstract}"


class LexicalVerifier:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=120000,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=DEFAULT_SEED,
        )

    def fit(self, examples: Sequence[PairExample]) -> None:
        texts = [pair_to_text(example) for example in examples]
        labels = [LABEL_TO_ID[example.label] for example in examples]
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)

    def predict_proba(self, examples: Sequence[PairExample], batch_size: Optional[int] = None) -> np.ndarray:
        texts = [pair_to_text(example) for example in examples]
        features = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(features)


class EncodedPairDataset(Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: Sequence[int]):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TransformerVerifier:
    def __init__(
        self,
        model_name: str,
        output_dir: Path,
        device: torch.device,
        use_amp: bool,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.device = device
        self.use_amp = use_amp
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def train(
        cls,
        train_examples: Sequence[PairExample],
        output_dir: Path,
        device: torch.device,
        model_name: str = "google/bert_uncased_L-4_H-256_A-4",
        epochs: int = 3,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int = 256,
        use_amp: bool = True,
        num_workers: int = 0,
        seed: int = DEFAULT_SEED,
    ) -> Tuple["TransformerVerifier", Dict[str, Any]]:
        ensure_dir(output_dir)
        set_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        labels = np.array([LABEL_TO_ID[example.label] for example in train_examples], dtype=np.int64)
        train_idx, valid_idx = train_test_split(
            np.arange(len(train_examples)),
            test_size=0.1,
            random_state=seed,
            stratify=labels,
        )
        train_subset = [train_examples[index] for index in train_idx]
        valid_subset = [train_examples[index] for index in valid_idx]

        train_dataset = encode_pairs_for_dataset(train_subset, tokenizer, max_length=max_length)
        valid_dataset = encode_pairs_for_dataset(valid_subset, tokenizer, max_length=max_length)
        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_train_steps = math.ceil(len(train_loader) / max(gradient_accumulation_steps, 1)) * epochs
        warmup_steps = max(1, int(0.1 * total_train_steps))
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )
        class_counts = np.bincount([LABEL_TO_ID[item.label] for item in train_subset], minlength=3)
        class_weights = torch.tensor(
            len(train_subset) / (3 * np.maximum(class_counts, 1)),
            dtype=torch.float32,
            device=device,
        )

        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        history: List[Dict[str, float]] = []
        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_macro_f1 = -1.0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                labels_tensor = batch.pop("labels")
                with get_amp_context(device, enabled=amp_enabled):
                    logits = model(**batch).logits
                    loss = F.cross_entropy(logits, labels_tensor, weight=class_weights)
                    loss = loss / max(gradient_accumulation_steps, 1)
                scaler.scale(loss).backward()

                should_step = (
                    step % max(gradient_accumulation_steps, 1) == 0
                    or step == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item() * max(gradient_accumulation_steps, 1))

            valid_true, valid_pred = predict_with_model(
                model=model,
                data_loader=valid_loader,
                device=device,
                use_amp=amp_enabled,
            )
            valid_macro_f1 = f1_score(valid_true, valid_pred, average="macro", zero_division=0)
            epoch_record = {
                "epoch": float(epoch),
                "train_loss": total_loss / max(len(train_loader), 1),
                "valid_macro_f1": float(valid_macro_f1),
            }
            history.append(epoch_record)
            if valid_macro_f1 > best_macro_f1:
                best_macro_f1 = valid_macro_f1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("Transformer training did not produce a valid checkpoint.")

        model.load_state_dict(best_state)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return cls(
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            use_amp=amp_enabled,
            max_length=max_length,
        ), {
            "history": history,
            "best_valid_macro_f1": best_macro_f1,
        }

    def predict_proba(self, examples: Sequence[PairExample], batch_size: int = 16) -> np.ndarray:
        dataset = encode_pairs_for_dataset(examples, self.tokenizer, max_length=self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        self.model.eval()
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, self.device)
                batch.pop("labels")
                with get_amp_context(self.device, enabled=self.use_amp):
                    logits = self.model(**batch).logits
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                outputs.append(probabilities)
        return np.vstack(outputs)


def encode_pairs_for_dataset(
    examples: Sequence[PairExample],
    tokenizer: Any,
    max_length: int = 256,
) -> EncodedPairDataset:
    claims = [example.claim for example in examples]
    abstracts = [example.abstract for example in examples]
    labels = [LABEL_TO_ID[example.label] for example in examples]
    encodings = tokenizer(
        claims,
        abstracts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return EncodedPairDataset(encodings, labels)


def predict_with_model(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            with get_amp_context(device, enabled=use_amp):
                logits = model(**batch).logits
                predictions = torch.argmax(logits, dim=-1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(predictions.cpu().numpy().tolist())
    return np.array(all_true), np.array(all_pred)


def evaluate_retrieval(
    claims: Sequence[Dict[str, Any]],
    retrieved_doc_ids: Sequence[Sequence[int]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    evidence_claims = []
    reciprocal_ranks: List[float] = []
    per_k_hits = {int(k): [] for k in ks}
    evidence_coverage: List[float] = []
    for claim_record, retrieved in zip(claims, retrieved_doc_ids):
        positives = set(gold_doc_labels(claim_record).keys())
        if not positives:
            continue
        evidence_claims.append(int(claim_record["id"]))
        first_hit_rank = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in positives:
                first_hit_rank = 1.0 / rank
                break
        reciprocal_ranks.append(first_hit_rank)
        for k in ks:
            hit = any(doc_id in positives for doc_id in retrieved[:k])
            per_k_hits[int(k)].append(float(hit))
        coverage = len(set(retrieved[: max(ks)]) & positives) / len(positives)
        evidence_coverage.append(float(coverage))
    metrics: Dict[str, float] = {"n_evidence_claims": float(len(evidence_claims))}
    for k, values in per_k_hits.items():
        metrics[f"recall@{k}"] = float(np.mean(values)) if values else 0.0
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["evidence_coverage"] = float(np.mean(evidence_coverage)) if evidence_coverage else 0.0
    return metrics


def evaluate_classifier(
    verifier_name: str,
    verifier: Any,
    examples: Sequence[PairExample],
    output_dir: Path,
    eval_batch_size: int,
) -> Dict[str, Any]:
    probabilities = verifier.predict_proba(examples, batch_size=eval_batch_size)
    y_true = np.array([LABEL_TO_ID[example.label] for example in examples], dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    metrics = {
        "model": verifier_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for label_name, label_id in LABEL_TO_ID.items():
        metrics[f"f1_{label_name.lower()}"] = float(
            f1_score(y_true == label_id, y_pred == label_id, zero_division=0)
        )
    prediction_rows = []
    for example, predicted_id, proba in zip(examples, y_pred, probabilities):
        row = asdict(example)
        row["predicted_label"] = ID_TO_LABEL[int(predicted_id)]
        row["prob_noinfo"] = float(proba[LABEL_TO_ID["NOINFO"]])
        row["prob_supports"] = float(proba[LABEL_TO_ID["SUPPORTS"]])
        row["prob_refutes"] = float(proba[LABEL_TO_ID["REFUTES"]])
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(output_dir / f"{verifier_name}_oracle_predictions.csv", index=False)
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        label_names=["NOINFO", "SUPPORTS", "REFUTES"],
        output_path=output_dir / f"{verifier_name}_confusion_matrix.png",
        title=f"{verifier_name} Oracle Classification",
    )
    return metrics


def evaluate_end_to_end(
    claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    retrieved_doc_ids: Sequence[Sequence[int]],
    verifier: Any,
    name: str,
    output_dir: Path,
    eval_batch_size: int,
    ks: Sequence[int] = (3, 5),
) -> Dict[str, float]:
    metrics: Dict[str, List[float]] = {f"joint_hit@{k}": [] for k in ks}
    example_rows: List[Dict[str, Any]] = []
    for claim_record, retrieved in zip(claims, retrieved_doc_ids):
        gold = gold_doc_labels(claim_record)
        if not gold:
            continue
        pair_examples = [
            PairExample(
                claim_id=int(claim_record["id"]),
                doc_id=int(doc_id),
                claim=claim_record["claim"],
                abstract=doc_text(corpus_by_id[doc_id]),
                label=gold.get(int(doc_id), "NOINFO"),
            )
            for doc_id in retrieved
            if int(doc_id) in corpus_by_id
        ]
        probabilities = verifier.predict_proba(pair_examples, batch_size=eval_batch_size)
        predicted_ids = probabilities.argmax(axis=1)
        predicted_labels = [ID_TO_LABEL[int(label_id)] for label_id in predicted_ids]
        for k in ks:
            success = any(
                int(example.doc_id) in gold and predicted_labels[idx] == gold[int(example.doc_id)]
                for idx, example in enumerate(pair_examples[:k])
            )
            metrics[f"joint_hit@{k}"].append(float(success))
        example_rows.append(
            {
                "claim_id": int(claim_record["id"]),
                "claim": claim_record["claim"],
                "gold_doc_labels": json.dumps(gold),
                "retrieved_doc_ids": json.dumps([int(doc_id) for doc_id in retrieved[: max(ks)]]),
                "predicted_labels": json.dumps(predicted_labels[: max(ks)]),
            }
        )
    pd.DataFrame(example_rows).to_csv(output_dir / f"{name}_pipeline_predictions.csv", index=False)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_retrieval_curve(
    retrieval_results: Dict[str, Dict[str, float]],
    ks: Sequence[int],
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for retriever_name, metrics in retrieval_results.items():
        values = [metrics[f"recall@{k}"] for k in ks]
        plt.plot(list(ks), values, marker="o", label=retriever_name)
    plt.xticks(list(ks))
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("SciFact Retrieval Recall on Dev")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def dataset_summary(
    train_claims: Sequence[Dict[str, Any]],
    dev_claims: Sequence[Dict[str, Any]],
    corpus_by_id: Dict[int, Dict[str, Any]],
    train_examples: Sequence[PairExample],
    dev_examples: Sequence[PairExample],
) -> Dict[str, Any]:
    summary = {
        "n_corpus_docs": len(corpus_by_id),
        "n_train_claims": len(train_claims),
        "n_dev_claims": len(dev_claims),
        "n_train_evidence_claims": sum(1 for record in train_claims if gold_doc_labels(record)),
        "n_dev_evidence_claims": sum(1 for record in dev_claims if gold_doc_labels(record)),
        "n_train_pairs": len(train_examples),
        "n_dev_pairs": len(dev_examples),
    }
    for split_name, examples in [("train", train_examples), ("dev", dev_examples)]:
        label_distribution = pd.Series([example.label for example in examples]).value_counts().to_dict()
        summary[f"{split_name}_pair_label_distribution"] = label_distribution
    return summary


def save_metrics_tables(
    retrieval_results: Dict[str, Dict[str, float]],
    classifier_results: Dict[str, Dict[str, float]],
    pipeline_results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> None:
    pd.DataFrame.from_dict(retrieval_results, orient="index").round(4).to_csv(output_dir / "retrieval_metrics.csv")
    pd.DataFrame.from_dict(classifier_results, orient="index").round(4).to_csv(output_dir / "classification_metrics.csv")
    pd.DataFrame.from_dict(pipeline_results, orient="index").round(4).to_csv(output_dir / "pipeline_metrics.csv")


def save_summary_markdown(
    summary: Dict[str, Any],
    retrieval_results: Dict[str, Dict[str, float]],
    classifier_results: Dict[str, Dict[str, float]],
    pipeline_results: Dict[str, Dict[str, float]],
    runtime_stats: Dict[str, float],
    config: ExperimentConfig,
    device_info: Dict[str, Any],
    output_path: Path,
) -> None:
    lines = [
        "# SciFact Midway Experiment Summary",
        "",
        "## Models and Training Setup",
        f"- Preset: {config.preset}",
        f"- Retriever model: {config.retriever_model_name}",
        f"- Verifier model: {config.verifier_model_name}",
        f"- Device: {device_info['resolved_device']}",
        f"- AMP enabled: {device_info['amp_enabled']}",
        f"- Train batch size: {config.train_batch_size}",
        f"- Eval batch size: {config.eval_batch_size}",
        f"- Retriever batch size: {config.retriever_batch_size}",
        f"- Epochs: {config.epochs}",
        f"- Learning rate: {config.learning_rate}",
        f"- Max length: {config.max_length}",
        f"- Negatives per claim: {config.negatives_per_claim}",
        "",
        "## Dataset",
        f"- Corpus documents: {summary['n_corpus_docs']}",
        f"- Train claims: {summary['n_train_claims']}",
        f"- Dev claims: {summary['n_dev_claims']}",
        f"- Train evidence-bearing claims: {summary['n_train_evidence_claims']}",
        f"- Dev evidence-bearing claims: {summary['n_dev_evidence_claims']}",
        f"- Train document pairs: {summary['n_train_pairs']}",
        f"- Dev document pairs: {summary['n_dev_pairs']}",
        "",
        "## Retrieval",
    ]
    for name, metrics in retrieval_results.items():
        lines.append(
            f"- {name}: Recall@3={metrics['recall@3']:.4f}, Recall@5={metrics['recall@5']:.4f}, "
            f"Recall@10={metrics['recall@10']:.4f}, MRR={metrics['mrr']:.4f}"
        )
    lines.extend(["", "## Oracle Classification"])
    for name, metrics in classifier_results.items():
        lines.append(
            f"- {name}: Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}, "
            f"F1(NoInfo)={metrics['f1_noinfo']:.4f}, F1(Supports)={metrics['f1_supports']:.4f}, "
            f"F1(Refutes)={metrics['f1_refutes']:.4f}"
        )
    lines.extend(["", "## End-to-End Evidence Hit"])
    for name, metrics in pipeline_results.items():
        joint_keys = sorted(metrics.keys())
        metrics_text = ", ".join([f"{key}={metrics[key]:.4f}" for key in joint_keys])
        lines.append(f"- {name}: {metrics_text}")
    lines.extend(["", "## Runtime"])
    for name, value in runtime_stats.items():
        lines.append(f"- {name}: {value:.2f} seconds")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_all_experiments(
    project_root: Path,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[ExperimentConfig] = None,
) -> Dict[str, Any]:
    config = config or build_experiment_config("cpu_quick")
    set_seed(config.seed)
    project_root = project_root.resolve()
    data_dir = (data_dir or project_root / "data" / "data").resolve()
    output_dir = (output_dir or project_root / "results" / config.preset).resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "plots")
    ensure_dir(output_dir / "models")
    ensure_dir(output_dir / "cache")

    device = get_device(config.prefer_device)
    amp_enabled = config.use_amp and device.type == "cuda"
    device_info = {
        "requested_device": config.prefer_device,
        "resolved_device": str(device),
        "amp_enabled": amp_enabled,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }

    runtime_stats: Dict[str, float] = {}

    start = time.perf_counter()
    corpus_by_id, train_claims, dev_claims = load_scifact(data_dir)
    doc_ids, corpus_texts = corpus_texts_and_ids(corpus_by_id)
    runtime_stats["data_loading"] = time.perf_counter() - start

    start = time.perf_counter()
    tfidf_retriever = TfidfRetriever.fit(corpus_texts, doc_ids)
    train_pair_examples = build_pair_examples(
        train_claims,
        corpus_by_id,
        tfidf_retriever,
        negatives_per_claim=config.negatives_per_claim,
        retrieval_pool_k=config.retrieval_pool_k,
        seed=config.seed,
    )
    dev_pair_examples = build_pair_examples(
        dev_claims,
        corpus_by_id,
        tfidf_retriever,
        negatives_per_claim=config.negatives_per_claim,
        retrieval_pool_k=config.retrieval_pool_k,
        seed=config.seed,
    )
    runtime_stats["tfidf_and_pairs"] = time.perf_counter() - start

    start = time.perf_counter()
    lexical_verifier = LexicalVerifier()
    lexical_verifier.fit(train_pair_examples)
    runtime_stats["train_lexical_verifier"] = time.perf_counter() - start

    start = time.perf_counter()
    dense_retriever = DenseRetriever.fit(
        corpus_texts=corpus_texts,
        doc_ids=doc_ids,
        model_name=config.retriever_model_name,
        cache_dir=output_dir / "cache" / "retriever_embeddings",
        device=device,
        use_amp=amp_enabled,
        batch_size=config.retriever_batch_size,
        max_length=config.max_length,
    )
    runtime_stats["build_dense_retriever"] = time.perf_counter() - start

    start = time.perf_counter()
    transformer_verifier, transformer_train_info = TransformerVerifier.train(
        train_examples=train_pair_examples,
        output_dir=output_dir / "models" / "transformer_verifier",
        device=device,
        model_name=config.verifier_model_name,
        epochs=config.epochs,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_length=config.max_length,
        use_amp=amp_enabled,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    runtime_stats["train_transformer_verifier"] = time.perf_counter() - start

    claim_texts = [claim_record["claim"] for claim_record in dev_claims]

    start = time.perf_counter()
    tfidf_dev_retrieved = tfidf_retriever.batch_search(claim_texts, k=config.retrieval_top_k)
    runtime_stats["tfidf_dev_retrieval"] = time.perf_counter() - start

    start = time.perf_counter()
    dense_dev_retrieved = dense_retriever.batch_search(
        claim_texts,
        k=config.retrieval_top_k,
        batch_size=config.retriever_batch_size,
    )
    runtime_stats["dense_dev_retrieval"] = time.perf_counter() - start

    retrieval_results = {
        "tfidf": evaluate_retrieval(dev_claims, tfidf_dev_retrieved, ks=config.retrieval_ks),
        "dense": evaluate_retrieval(dev_claims, dense_dev_retrieved, ks=config.retrieval_ks),
    }

    start = time.perf_counter()
    classifier_results = {
        "logreg": evaluate_classifier(
            "logreg",
            lexical_verifier,
            dev_pair_examples,
            output_dir / "plots",
            eval_batch_size=config.eval_batch_size,
        ),
        "transformer": evaluate_classifier(
            "transformer",
            transformer_verifier,
            dev_pair_examples,
            output_dir / "plots",
            eval_batch_size=config.eval_batch_size,
        ),
    }
    runtime_stats["classifier_eval"] = time.perf_counter() - start

    start = time.perf_counter()
    pipeline_results = {
        "tfidf_plus_logreg": evaluate_end_to_end(
            dev_claims,
            corpus_by_id,
            tfidf_dev_retrieved,
            lexical_verifier,
            "tfidf_plus_logreg",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            ks=config.pipeline_ks,
        ),
        "tfidf_plus_transformer": evaluate_end_to_end(
            dev_claims,
            corpus_by_id,
            tfidf_dev_retrieved,
            transformer_verifier,
            "tfidf_plus_transformer",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            ks=config.pipeline_ks,
        ),
        "dense_plus_logreg": evaluate_end_to_end(
            dev_claims,
            corpus_by_id,
            dense_dev_retrieved,
            lexical_verifier,
            "dense_plus_logreg",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            ks=config.pipeline_ks,
        ),
        "dense_plus_transformer": evaluate_end_to_end(
            dev_claims,
            corpus_by_id,
            dense_dev_retrieved,
            transformer_verifier,
            "dense_plus_transformer",
            output_dir,
            eval_batch_size=config.eval_batch_size,
            ks=config.pipeline_ks,
        ),
    }
    runtime_stats["pipeline_eval"] = time.perf_counter() - start

    plot_retrieval_curve(
        retrieval_results=retrieval_results,
        ks=list(config.retrieval_ks),
        output_path=output_dir / "plots" / "retrieval_recall_curve.png",
    )
    save_metrics_tables(retrieval_results, classifier_results, pipeline_results, output_dir)

    summary = dataset_summary(train_claims, dev_claims, corpus_by_id, train_pair_examples, dev_pair_examples)
    summary["transformer_train_info"] = transformer_train_info

    full_summary = {
        "config": asdict(config),
        "device_info": device_info,
        "dataset_summary": summary,
        "retrieval_results": retrieval_results,
        "classifier_results": classifier_results,
        "pipeline_results": pipeline_results,
        "runtime_stats": runtime_stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    save_summary_markdown(
        summary=summary,
        retrieval_results=retrieval_results,
        classifier_results=classifier_results,
        pipeline_results=pipeline_results,
        runtime_stats=runtime_stats,
        config=config,
        device_info=device_info,
        output_path=output_dir / "midway_summary.md",
    )
    return full_summary
