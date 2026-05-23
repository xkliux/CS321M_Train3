import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Paths
# =========================

user = os.getlogin()

BASE_DIR = Path(rf"C:\Users\{user}\Desktop\Stanford\CS321M\Prediction_challenge_train3")
TRAIN_DIR = BASE_DIR / "train"
META_DIR = BASE_DIR / "Meta_Data"
SUBMISSION_DIR = BASE_DIR / "submission"
CACHE_DIR = BASE_DIR / "embedding_cache"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2048


# =========================
# Load config/artifacts
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_json(SUBMISSION_DIR / "config.json")

ENCODER_NAME = CONFIG["encoder_name"]
D_TEXT = int(CONFIG["embedding_dim"])
D_EXTRA = int(CONFIG["d_extra"])
EXTRA_FEATURES = CONFIG["extra_features"]

GLOBAL_MEAN = float(load_json(SUBMISSION_DIR / "global_mean.json")["global_mean"])
SUBJECT_MEAN = load_json(SUBMISSION_DIR / "subject_mean.json")
BENCHMARK_MEAN = load_json(SUBMISSION_DIR / "benchmark_mean.json")
CONDITION_MEAN = load_json(SUBMISSION_DIR / "condition_mean.json")
ITEM_MEAN = load_json(SUBMISSION_DIR / "item_mean.json")

SUBJECT_TO_IDX = load_json(SUBMISSION_DIR / "subject_to_idx.json")
BENCHMARK_TO_IDX = load_json(SUBMISSION_DIR / "benchmark_to_idx.json")
CONDITION_TO_IDX = load_json(SUBMISSION_DIR / "condition_to_idx.json")
# =========================
# Model
# =========================

class NCF(nn.Module):
    def __init__(self, d_item_text, d_extra, n_subjects, n_benchmarks, n_conditions):
        super().__init__()

        self.subject_emb = nn.Embedding(n_subjects, 32)
        self.benchmark_emb = nn.Embedding(n_benchmarks, 16)
        self.condition_emb = nn.Embedding(n_conditions, 8)

        input_dim = d_item_text + 32 + 16 + 8 + d_extra

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, item_x, subject_idx, benchmark_idx, condition_idx, extra):

        x = torch.cat([
            item_x,
            self.subject_emb(subject_idx),
            self.benchmark_emb(benchmark_idx),
            self.condition_emb(condition_idx),
            extra,
        ], dim=1)

        return self.net(x).squeeze(-1)


model = NCF(
    d_item_text=D_TEXT,
    d_extra=D_EXTRA,
    n_subjects=len(SUBJECT_TO_IDX),
    n_benchmarks=len(BENCHMARK_TO_IDX),
    n_conditions=len(CONDITION_TO_IDX),
).to(DEVICE)
model.load_state_dict(torch.load(SUBMISSION_DIR / "ncf_head.pt", map_location=DEVICE))
model.eval()


# =========================
# Helpers
# =========================

def clip_prob(x):
    return max(0.01, min(0.99, float(x)))


def load_embedding_map(name):
    safe_encoder_name = ENCODER_NAME.replace("/", "_")
    emb_path = CACHE_DIR / f"{name}_{safe_encoder_name}.npy"
    ids_path = CACHE_DIR / f"{name}_{safe_encoder_name}_ids.json"

    embeddings = np.load(emb_path)
    ids = load_json(ids_path)

    return dict(zip([str(x) for x in ids], embeddings, strict=False))


def build_extra_features(df):
    raw = {}

    raw["subject_prior"] = df["subject_content"].map(SUBJECT_MEAN).fillna(GLOBAL_MEAN).to_numpy()
    raw["benchmark_prior"] = df["benchmark"].map(BENCHMARK_MEAN).fillna(GLOBAL_MEAN).to_numpy()
    raw["condition_prior"] = df["condition"].map(CONDITION_MEAN).fillna(GLOBAL_MEAN).to_numpy()
    raw["item_prior"] = df["item_content"].map(ITEM_MEAN).fillna(GLOBAL_MEAN).to_numpy()

    raw["subject_minus_item"] = raw["subject_prior"] - raw["item_prior"]
    raw["subject_minus_benchmark"] = raw["subject_prior"] - raw["benchmark_prior"]

    X_extra = np.column_stack([raw[name] for name in EXTRA_FEATURES])
    return torch.tensor(X_extra, dtype=torch.float32), raw


def log_loss(y, p):
    p = np.clip(p, 0.05, 0.95)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def get_model_probs(loader, temperature=1.0):
    probs = []

    model.eval()

    with torch.no_grad():
        for item_x, subject_idx, benchmark_idx, condition_idx, extra in loader:
            item_x = item_x.to(DEVICE)
            subject_idx = subject_idx.to(DEVICE)
            benchmark_idx = benchmark_idx.to(DEVICE)
            condition_idx = condition_idx.to(DEVICE)
            extra = extra.to(DEVICE)

            logits = model(
                item_x,
                subject_idx,
                benchmark_idx,
                condition_idx,
                extra,
            )

            p = torch.sigmoid(logits / temperature).cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)


def render_subject_content(subject, fallback_subject_id):

    display_name = subject.get("display_name") or fallback_subject_id
    return f"Name: {display_name}"


# =========================
# Rebuild validation data
# =========================

print("Loading train data...")

train_files = sorted(TRAIN_DIR.glob("*.parquet"))
dfs = []

for f in train_files:
    temp = pd.read_parquet(f)
    temp["source_file"] = f.name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

print("Loading metadata...")

items = pd.read_parquet(META_DIR / "items.parquet")
subjects = pd.read_parquet(META_DIR / "subjects.parquet")
benchmarks = pd.read_parquet(META_DIR / "benchmarks.parquet")

items["item_content"] = items["content"].fillna("").astype(str)

subjects["subject_content"] = subjects.apply(
    lambda row: render_subject_content(row, row["subject_id"]),
    axis=1,
)

benchmarks["benchmark"] = benchmarks["benchmark_id"].fillna("").astype(str)

df = df.merge(items[["item_id", "item_content"]], on="item_id", how="left")
df = df.merge(subjects[["subject_id", "subject_content"]], on="subject_id", how="left")
df = df.merge(benchmarks[["benchmark_id", "benchmark"]], on="benchmark_id", how="left")

df = df.rename(columns={"test_condition": "condition", "response": "label"})

df["condition"] = df["condition"].fillna("none").astype(str)
df["item_content"] = df["item_content"].fillna("").astype(str)
df["subject_content"] = df["subject_content"].fillna("").astype(str)
df["benchmark"] = df["benchmark"].fillna("").astype(str)

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = (df["label"] > 0.5).astype(float)

df = df[
    (df["item_content"].str.len() > 0)
    & (df["subject_content"].str.len() > 0)
    & (df["benchmark"].str.len() > 0)
].copy()

print("Rows:", len(df))


# Same item split as training
unique_items = df["item_content"].drop_duplicates()

_, val_items = train_test_split(
    unique_items,
    test_size=0.1,
    random_state=69,
)

val_df = df[df["item_content"].isin(val_items)].copy()
print("Validation rows:", len(val_df))


# =========================
# Load cached embeddings
# =========================

print("Loading cached item embeddings...")

item_map = load_embedding_map("items")

X_text = torch.tensor(
    np.stack([item_map[str(x)] for x in val_df["item_id"]]),
    dtype=torch.float32,
)


subject_idx_tensor = torch.tensor(
    [
        SUBJECT_TO_IDX.get(x, 0)
        for x in val_df["subject_content"]
    ],
    dtype=torch.long,
)

benchmark_idx_tensor = torch.tensor(
    [
        BENCHMARK_TO_IDX.get(x, 0)
        for x in val_df["benchmark"]
    ],
    dtype=torch.long,
)

condition_idx_tensor = torch.tensor(
    [
        CONDITION_TO_IDX.get(x, 0)
        for x in val_df["condition"]
    ],
    dtype=torch.long,
)


X_extra, raw = build_extra_features(val_df)

y_val = val_df["label"].to_numpy().astype(float)

loader = DataLoader(
    TensorDataset(
        X_text,
        subject_idx_tensor,
        benchmark_idx_tensor,
        condition_idx_tensor,
        X_extra,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================
# Grid search
# =========================

print("Getting model probabilities...")

results = []

for temp in [1.0, 1.2, 1.5, 2.0]:
    p_model = get_model_probs(loader, temperature=temp)

    item_prior = raw["item_prior"]
    subject_prior = raw["subject_prior"]
    benchmark_prior = raw["benchmark_prior"]
    condition_prior = raw["condition_prior"]

    # 3-way search: model/item/subject
    for wm in np.arange(0.30, 0.91, 0.05):
        for wi in np.arange(0.00, 0.61, 0.05):
            ws = 1.0 - wm - wi

            if ws < -1e-9 or ws > 0.50:
                continue

            p = wm * p_model + wi * item_prior + ws * subject_prior
            loss = log_loss(y_val, p)

            results.append(
                {
                    "loss": loss,
                    "temp": temp,
                    "wm": wm,
                    "wi": wi,
                    "ws": ws,
                    "wb": 0.0,
                    "wc": 0.0,
                    "type": "model_item_subject",
                }
            )

    # 4-way search with tiny benchmark prior
    for wm in np.arange(0.40, 0.91, 0.05):
        for wi in np.arange(0.00, 0.51, 0.05):
            for ws in np.arange(0.00, 0.31, 0.05):
                wb = 1.0 - wm - wi - ws

                if wb < -1e-9 or wb > 0.15:
                    continue

                p = wm * p_model + wi * item_prior + ws * subject_prior + wb * benchmark_prior
                loss = log_loss(y_val, p)

                results.append(
                    {
                        "loss": loss,
                        "temp": temp,
                        "wm": wm,
                        "wi": wi,
                        "ws": ws,
                        "wb": wb,
                        "wc": 0.0,
                        "type": "model_item_subject_benchmark",
                    }
                )

results_df = pd.DataFrame(results).sort_values("loss")

print("\nTop 20 blends:")
print(results_df.head(20).to_string(index=False))

out_path = BASE_DIR / "blend_search_results.csv"
results_df.to_csv(out_path, index=False)

print("\nSaved:", out_path)