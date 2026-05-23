import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

######parameters start

user = os.getlogin()

BASE_DIR = Path(rf"C:\Users\{user}\Desktop\Stanford\CS321M\Prediction_challenge_train3")

TRAIN_DIR = BASE_DIR / "train"
META_DIR = BASE_DIR / "Meta_Data"
CACHE_DIR = BASE_DIR / "embedding_cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = BASE_DIR / "submission"
OUT_DIR.mkdir(exist_ok=True)
LOAD_ONLY = None #integer or None
# D_ID = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"

MAX_ROWS = 6_000_000
EPOCHS = 40
BATCH_SIZE = 2048
ENCODE_BATCH_SIZE = 512
patience = 3

######parameters end

##########helpers start

#build second order priors
def make_so_prior(df, col1, col2, label_col="label", sep="|||"):
    pair_mean = (
        df.groupby([col1, col2])[label_col]
        .mean()
        .apply(clip_prob)
        .to_dict()
    )

    pair_mean_json = {
        f"{k[0]}{sep}{k[1]}": v
        for k, v in pair_mean.items()
    }

    return pair_mean_json


def lookup_so_prior(pair_dict, a, b, global_mean, sep="|||"):
    return pair_dict.get(f"{a}{sep}{b}", global_mean)



#load embedding if exist, make if not
def get_or_compute_embedding_map(name, ids, texts, encoder):
    safe_encoder_name = ENCODER_NAME.replace("/", "_")
    emb_path = CACHE_DIR / f"{name}_{safe_encoder_name}.npy"
    ids_path = CACHE_DIR / f"{name}_{safe_encoder_name}_ids.json"

    ids = [str(x) for x in ids]
    texts = [str(x) for x in texts]
    current_text_map = dict(zip(ids, texts))

    if emb_path.exists() and ids_path.exists():
        print(f"Loading cached {name} embeddings...")
        embeddings = np.load(emb_path)

        with open(ids_path, "r", encoding="utf-8") as f:
            cached_ids = json.load(f)

        cached_map = dict(zip(cached_ids, embeddings))

        missing_ids = [x for x in ids if x not in cached_map]

        if not missing_ids:
            print(f"Using cached {name} embeddings.")
            return {x: cached_map[x] for x in ids}

        print(f"Only computing {len(missing_ids)} new {name} embeddings...")

        missing_texts = [current_text_map[x] for x in missing_ids]

        missing_embeddings = encoder.encode(missing_texts,batch_size=ENCODE_BATCH_SIZE,show_progress_bar=True,normalize_embeddings=True,)

        for x, emb in zip(missing_ids, missing_embeddings):
            cached_map[x] = emb

        updated_ids = list(cached_map.keys())
        updated_embeddings = np.stack([cached_map[x] for x in updated_ids])

        np.save(emb_path, updated_embeddings)

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(updated_ids, f)

        return {x: cached_map[x] for x in ids}

    print(f"Computing all {name} embeddings...")
    embeddings = encoder.encode(texts,batch_size=ENCODE_BATCH_SIZE,show_progress_bar=True,normalize_embeddings=True,)

    np.save(emb_path, embeddings)

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    return dict(zip(ids, embeddings))


def compute_or_load_idx(name, df, id_col):
    idx_path = CACHE_DIR / f"{name}_to_idx.json"

    # Load mapping if exists
    if idx_path.exists():
        print(f"Loading cached {name}_to_idx...")
        with open(idx_path, "r", encoding="utf-8") as f:
            id_to_idx = json.load(f)

    # make mapping and save it ifn ot
    else:
        print(f"Creating {name}_to_idx...")

        id_to_idx = {"__UNK__": 0}

        unique_ids = df[id_col].astype(str).drop_duplicates().tolist()

        for i, k in enumerate(unique_ids, start=1):
            id_to_idx[k] = i

        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(id_to_idx, f)

    # Map IDs to index, not found =  UNK
    idx_series = (df[id_col].astype(str).map(lambda x: id_to_idx.get(x, id_to_idx["__UNK__"])))

    return id_to_idx, idx_series

def join_text_fields(df, cols):
    parts = []
    for col in cols:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

# def render_subject_content(subject, fallback_subject_id):
#     display_name = subject.get("display_name") or fallback_subject_id
#
#     lines = [f"Name: {display_name}"]
#
#     optional_fields = (
#         ("provider", "Organization"),
#         ("params", "Parameters"),
#         ("family", "Family"),
#     )
#
#     for key, label in optional_fields:
#         value = subject.get(key)
#
#         if value:
#             lines.append(f"{label}: {value}")
#
#     return "\n".join(lines)
def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    return f"Name: {display_name}"

def build_extra_features(df,global_mean,subject_mean,benchmark_mean,condition_mean,item_mean,U=None,V=None, B=None,prior = True, gap = True, length = True, counts = True, similarity = True,interactions=True, so_priors=None,):
    features = {}
    ##########prior prob
    if prior:
        features["subject_prior"] = (df["subject_content"].map(subject_mean).fillna(global_mean).to_numpy())

        features["benchmark_prior"] = (df["benchmark"].map(benchmark_mean).fillna(global_mean).to_numpy())

        features["condition_prior"] = (df["condition"].map(condition_mean).fillna(global_mean).to_numpy())

        features["item_prior"] = df["item_content"].map(item_mean).fillna(global_mean).to_numpy()

    #######tensor diff
    if gap:
        features["subject_minus_item"] = (features["subject_prior"] - features["item_prior"])

        features["subject_minus_benchmark"] = (features["subject_prior"] - features["benchmark_prior"])

        ########feature lenght
    if length:
        item_len = df["item_content"].str.len().clip(0, 2000).to_numpy() / 2000.0
        subject_len = df["subject_content"].str.len().clip(0, 1000).to_numpy() / 1000.0
        benchmark_len = df["benchmark"].str.len().clip(0, 1000).to_numpy() / 1000.0

        features["item_len"] = item_len
        features["subject_len"] = subject_len
        features["benchmark_len"] = benchmark_len

        features["item_subject_len_ratio"] = item_len / (subject_len + 1e-6)

    # counts correct
    if counts:
        subject_count = df.groupby("subject_content")["label"].transform("count")
        benchmark_count = df.groupby("benchmark")["label"].transform("count")
        condition_count = df.groupby("condition")["label"].transform("count")
        item_count = df.groupby("item_id")["label"].transform("count")

        features["subject_count"] = (np.log1p(subject_count) / np.log1p(subject_count.max())).to_numpy()

        features["benchmark_count"] = (np.log1p(benchmark_count) / np.log1p(benchmark_count.max())).to_numpy()

        features["condition_count"] = (np.log1p(condition_count) / np.log1p(condition_count.max())).to_numpy()

        features["item_count"] = (np.log1p(item_count) / np.log1p(item_count.max())).to_numpy()


    ######sim & interaction terms on wiht UVB

    if similarity:
        if U is not None and V is not None:
            features["cos_subject_item"] = (F.cosine_similarity(U, V, dim=1).detach().cpu().numpy())

        if U is not None and B is not None:
            features["cos_subject_benchmark"] = (F.cosine_similarity(U, B, dim=1).detach().cpu().numpy())

        if V is not None and B is not None:
            features["cos_item_benchmark"] = (F.cosine_similarity(V, B, dim=1).detach().cpu().numpy())

    if interactions:
        if U is not None and V is not None:
            uv_mul = U * V
            uv_diff = torch.abs(U - V)

            features["uv_mul_mean"] = uv_mul.mean(dim=1).detach().cpu().numpy()
            features["uv_mul_sum"] = uv_mul.sum(dim=1).detach().cpu().numpy()
            features["uv_diff_mean"] = uv_diff.mean(dim=1).detach().cpu().numpy()
            features["uv_diff_sum"] = uv_diff.sum(dim=1).detach().cpu().numpy()

        if V is not None and B is not None:
            vb_mul = V * B
            vb_diff = torch.abs(V - B)

            features["vb_mul_mean"] = vb_mul.mean(dim=1).detach().cpu().numpy()
            features["vb_mul_sum"] = vb_mul.sum(dim=1).detach().cpu().numpy()
            features["vb_diff_mean"] = vb_diff.mean(dim=1).detach().cpu().numpy()
            features["vb_diff_sum"] = vb_diff.sum(dim=1).detach().cpu().numpy()

        if U is not None and B is not None:
            ub_mul = U * B
            ub_diff = torch.abs(U - B)

            features["ub_mul_mean"] = ub_mul.mean(dim=1).detach().cpu().numpy()
            features["ub_mul_sum"] = ub_mul.sum(dim=1).detach().cpu().numpy()
            features["ub_diff_mean"] = ub_diff.mean(dim=1).detach().cpu().numpy()
            features["ub_diff_sum"] = ub_diff.sum(dim=1).detach().cpu().numpy()

    if so_priors is not None:
        features["benchmark_condition_prior"] = np.array([
            lookup_so_prior(
                so_priors["benchmark_condition_prior"],
                b,
                c,
                global_mean,
            )
            for b, c in zip(df["benchmark"], df["condition"])
        ])

        features["subject_benchmark_prior"] = np.array([
            lookup_so_prior(
                so_priors["subject_benchmark_prior"],
                s,
                b,
                global_mean,
            )
            for s, b in zip(df["subject_content"], df["benchmark"])
        ])

        features["subject_condition_prior"] = np.array([
            lookup_so_prior(
                so_priors["subject_condition_prior"],
                s,
                c,
                global_mean,
            )
            for s, c in zip(df["subject_content"], df["condition"])
        ])

    # -------------------------#set output tensor

    feature_names = list(features.keys())

    X_extra = torch.tensor(np.column_stack([features[name] for name in feature_names]),dtype=torch.float32,)

    return X_extra, feature_names

# def get_probs_labels(model, loader):
#     model.eval()
#     probs_all = []
#     labels_all = []
#
#     with torch.no_grad():
#         for text_x, extra, yb in loader:
#             text_x = text_x.to(DEVICE)
#             extra = extra.to(DEVICE)
#
#             logits = model(text_x, extra)
#             probs = torch.sigmoid(logits).cpu().numpy()
#
#             probs_all.extend(probs)
#             labels_all.extend(yb.numpy())
#
#     return np.array(probs_all), np.array(labels_all)

def get_probs_labels(model, loader):
    model.eval()

    probs_all = []
    labels_all = []

    with torch.no_grad():

        for item_x, subject_idx, benchmark_idx, condition_idx, extra, yb in loader:

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

            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.extend(probs)
            labels_all.extend(yb.numpy())

    return np.array(probs_all), np.array(labels_all)


def save_training_artifacts(out_dir,encoder_name,X_text,X_extra,extra_feature_names,subject_to_idx,subject_mean,benchmark_mean,condition_mean,item_mean,so_priors,global_mean,benchmark_to_idx,condition_to_idx,):
    # config = {
    #     "encoder_name": encoder_name,
    #     "embedding_dim": int(U.shape[1]),
    #     "d_id": int(d_id),
    #     "d_extra": int(X_extra.shape[1]),
    #     "x_text_dim": int(X_text.shape[1]),
    #     "uses_benchmark_embedding": True,
    #     "uses_id_embeddings": False,
    #     "uses_item_id_embedding": False,
    #     "extra_features": extra_feature_names,
    #     "model_input_dim": int(X_text.shape[1] + X_extra.shape[1]),
    # }
    config = {
        "encoder_name": encoder_name,
        "embedding_dim": int(X_text.shape[1]),
        "d_extra": int(X_extra.shape[1]),
        "x_text_dim": int(X_text.shape[1]),

        "uses_id_embeddings": True,
        "uses_subject_id_embedding": True,
        "uses_benchmark_id_embedding": True,
        "uses_condition_embedding": True,

        "subject_embedding_dim": 32,
        "benchmark_embedding_dim": 16,
        "condition_embedding_dim": 8,

        "extra_features": extra_feature_names,
    }


    with open(out_dir / "so_priors.json", "w", encoding="utf-8") as f:
        json.dump(so_priors, f)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(out_dir / "subject_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(subject_to_idx, f)

    # with open(out_dir / "item_to_idx.json", "w", encoding="utf-8") as f:
    #     json.dump(item_to_idx, f)

    with open(out_dir / "global_mean.json", "w", encoding="utf-8") as f:
        json.dump({"global_mean": global_mean}, f)

    with open(out_dir / "subject_mean.json", "w", encoding="utf-8") as f:
        json.dump(subject_mean, f)

    with open(out_dir / "benchmark_mean.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_mean, f)

    with open(out_dir / "condition_mean.json", "w", encoding="utf-8") as f:
        json.dump(condition_mean, f)

    with open(out_dir / "item_mean.json", "w", encoding="utf-8") as f:
        json.dump(item_mean, f)

    with open(out_dir / "benchmark_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_to_idx, f)

    with open(out_dir / "condition_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(condition_to_idx, f)
    # with open(out_dir / "benchmark_condition_mean.json", "w", encoding="utf-8") as f:
    #     json.dump(benchmark_condition_mean_json, f)

    print("Saved config and lookup artifacts.")
    print(config)

def clip_prob(x):
    return max(0.01, min(0.99, float(x)))

# def compute_accuracy(model, loader):
#     model.eval()
#     total = 0
#     correct = 0
#
#     with torch.no_grad():
#         for text_x, extra, yb in loader:
#             text_x = text_x.to(DEVICE)
#             extra = extra.to(DEVICE)
#             yb = yb.to(DEVICE)
#
#             logits = model(text_x, extra)
#             probs = torch.sigmoid(logits)
#             preds = (probs > 0.5).float()
#
#             correct += (preds == yb).sum().item()
#             total += yb.size(0)
#
#     return correct / total
def compute_accuracy(model, loader):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():

        for item_x, subject_idx, benchmark_idx, condition_idx, extra, yb in loader:

            item_x = item_x.to(DEVICE)
            subject_idx = subject_idx.to(DEVICE)
            benchmark_idx = benchmark_idx.to(DEVICE)
            condition_idx = condition_idx.to(DEVICE)
            extra = extra.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(
                item_x,
                subject_idx,
                benchmark_idx,
                condition_idx,
                extra,
            )

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return correct / total
##########helpers end

###########Model NCF

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
        x = torch.cat(
            [
                item_x,
                self.subject_emb(subject_idx),
                self.benchmark_emb(benchmark_idx),
                self.condition_emb(condition_idx),
                extra,
            ],
            dim=1,
        )
        return self.net(x).squeeze(-1)

###########Model NCF

#load training data parquets

print("Loading official train shards...")

train_files = sorted(TRAIN_DIR.glob("*.parquet"))

if not train_files:
    raise FileNotFoundError(f"No train parquet files found in {TRAIN_DIR}")

#load from local
dfs = []

if LOAD_ONLY is not None:
    train_files=train_files[:LOAD_ONLY]

for f in train_files:
    print("Loading:", f.name)
    temp = pd.read_parquet(f)

    required_cols = {
        "subject_id",
        "item_id",
        "benchmark_id",
        "trial",
        "test_condition",
        "response",
    }

    missing = required_cols - set(temp.columns)
    if missing:
        print(f"{f} is with missing columns {missing}, skip loading")
        continue
        raise ValueError(f"{f.name} is missing columns: {missing}")

    temp["source_file"] = f.name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

print("Combined train shape:", df.shape)
print("Train columns:", df.columns.tolist())

##end training data loadiong

#load meta data for items, subjects and benchmarks

print("Loading metadata...")

items = pd.read_parquet(META_DIR / "items.parquet")
subjects = pd.read_parquet(META_DIR / "subjects.parquet")
benchmarks = pd.read_parquet(META_DIR / "benchmarks.parquet")

print("Items columns:", items.columns.tolist())
print("Subjects columns:", subjects.columns.tolist())
print("Benchmarks columns:", benchmarks.columns.tolist())

#end load meta data for items, subjects and benchmarks


####set loaded meta data into column as text to be embedded

items["item_content"] = items["content"].fillna("").astype(str)

# subjects["subject_content"] = subjects["display_name"].fillna("").astype(str)

# subjects["subject_content"] = (
#     subjects["display_name"].fillna("").astype(str)
#     + " "
#     + subjects["provider"].fillna("").astype(str)
#  + " "
#     + subjects["params"].fillna("").astype(str)
#     # + " "
#     # + subjects["release_date"].fillna("").astype(str)
#     #
#     # + " " + subjects["notes"].fillna("").astype(str)
# )
subjects["subject_content"] = subjects.apply(
    lambda row: render_subject_content(
        row,
        row["subject_id"],
    ),
    axis=1,
)
#benchmarks["benchmark"] = benchmarks["name"].fillna("").astype(str)

benchmarks["benchmark"] = benchmarks["benchmark_id"].fillna("").astype(str)

# benchmarks["benchmark"] = join_text_fields(
#     benchmarks,
#     [
#         #"benchmark_id",
#          "name",
#          # "description",
#          # "domain",
#          # "modality",
#         # "response_type",
#         # "response_scale",
#     ],
# )

####end set loaded meta data into column as text to be embedded

#########merge meta data and train data into big table

print("Merging metadata into train rows...")

df = df.merge(items[["item_id", "item_content"]],on="item_id",how="left",)

df = df.merge(subjects[["subject_id", "subject_content"]],on="subject_id",how="left",)

df = df.merge(benchmarks[["benchmark_id", "benchmark"]],on="benchmark_id",how="left",)

######### end merge meta data and train data into big table

#####  data clean up and rename
df = df.rename(columns={"test_condition": "condition","response": "label"})


df["condition"] = df["condition"].fillna("none").astype(str)
df["item_content"] = df["item_content"].fillna("").astype(str)
df["subject_content"] = df["subject_content"].fillna("").astype(str)
df["benchmark"] = df["benchmark"].fillna("").astype(str)

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])

# Binary  label
df["label"] = (df["label"] > 0.5).astype(float)

df = df[(df["item_content"].str.len() > 0)& (df["subject_content"].str.len() > 0)& (df["benchmark"].str.len() > 0)].copy()

print("Merged data shape:", df.shape)
print(df[["subject_content", "item_content", "benchmark", "condition", "label"]].head())

print("Label counts:")
print(df["label"].value_counts())
##### end  data clean up and rename


##optional sampling prevent data too big
if MAX_ROWS is not None and len(df) > MAX_ROWS:
    print(f"Sampling {MAX_ROWS} rows from {len(df)} rows...")
    df = df.sample(MAX_ROWS, random_state=69).reset_index(drop=True)

print("Training rows:", len(df))


############get priors and save

global_mean = clip_prob(df["label"].mean())

subject_mean = (df.groupby("subject_content")["label"].mean().apply(clip_prob).to_dict())
# subject_mean = (df.groupby("subject_id")["label"].mean().apply(clip_prob).to_dict())
benchmark_mean = (df.groupby("benchmark")["label"].mean().apply(clip_prob).to_dict())

condition_mean = (df.groupby("condition")["label"].mean().apply(clip_prob).to_dict())

item_mean = df.groupby("item_content")["label"].mean().apply(clip_prob).to_dict()
# item_mean = df.groupby("item_id")["label"].mean().apply(clip_prob).to_dict()

SO_PRIORS = {
    "benchmark_condition_prior": make_so_prior(df, "benchmark", "condition"),
    "subject_benchmark_prior": make_so_prior(df, "subject_content", "benchmark"),
    "subject_condition_prior": make_so_prior(df, "subject_content", "condition"),
}

with open(OUT_DIR / "global_mean.json", "w", encoding="utf-8") as f:
    json.dump({"global_mean": global_mean}, f)

with open(OUT_DIR / "subject_mean.json", "w", encoding="utf-8") as f:
    json.dump(subject_mean, f)

with open(OUT_DIR / "benchmark_mean.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_mean, f)

with open(OUT_DIR / "condition_mean.json", "w", encoding="utf-8") as f:
    json.dump(condition_mean, f)

with open(OUT_DIR / "item_mean.json", "w", encoding="utf-8") as f:
    json.dump(item_mean, f)

print("Saved priors.")


###encoding

print("Loading encoder...")
encoder = SentenceTransformer(ENCODER_NAME)

# print("Preparing unique subjects...")
# subject_table = (df[["subject_id", "subject_content"]].astype(str).drop_duplicates(subset=["subject_id"]).sort_values("subject_id"))
#
# subject_ids = subject_table["subject_id"].tolist()
# subject_texts = subject_table["subject_content"].tolist()
#
# subject_map = get_or_compute_embedding_map("subjects",subject_ids,subject_texts,encoder,)

print("Preparing unique items...")
item_table = (df[["item_id", "item_content"]].astype(str).drop_duplicates(subset=["item_id"]).sort_values("item_id"))

item_ids = item_table["item_id"].tolist()
item_texts = item_table["item_content"].tolist()

item_map = get_or_compute_embedding_map("items",item_ids,item_texts,encoder,)

# print("Preparing unique benchmarks...")
# benchmark_table = (df[["benchmark_id", "benchmark"]].astype(str).drop_duplicates(subset=["benchmark_id"]).sort_values("benchmark_id"))
#
# benchmark_ids = benchmark_table["benchmark_id"].tolist()
# benchmark_texts = benchmark_table["benchmark"].tolist()
#
# benchmark_map = get_or_compute_embedding_map("benchmarks",benchmark_ids,benchmark_texts,encoder,)
#
# ##############id embeddings


# subject_to_idx, df["subject_idx"] = compute_or_load_idx("subject",df,"subject_id")
#
# # item_to_idx, df["item_idx"] = compute_or_load_idx("item",df,"item_id")
#
# num_subjects = len(subject_to_idx)
# # num_items = len(item_to_idx)

##############id embeddings



#x_extra features with helper

# U = torch.tensor(np.stack([subject_map[str(x)] for x in df["subject_id"]]),dtype=torch.float32,)

# V = torch.tensor(np.stack([item_map[str(x)] for x in df["item_id"]]),dtype=torch.float32,)
#
# B = torch.tensor(np.stack([benchmark_map[str(x)] for x in df["benchmark_id"]]),dtype=torch.float32,)

# X_text = torch.cat([U, V, B], dim=1)
V = torch.tensor(
    np.stack([item_map[str(x)] for x in df["item_id"]]),
    dtype=torch.float32,
)

X_text = V
subject_to_idx, df["subject_idx"] = compute_or_load_idx("subject", df, "subject_id")
benchmark_to_idx, df["benchmark_idx"] = compute_or_load_idx("benchmark", df, "benchmark_id")
condition_to_idx, df["condition_idx"] = compute_or_load_idx("condition", df, "condition")

subject_idx_tensor = torch.tensor(df["subject_idx"].values, dtype=torch.long)
benchmark_idx_tensor = torch.tensor(df["benchmark_idx"].values, dtype=torch.long)
condition_idx_tensor = torch.tensor(df["condition_idx"].values, dtype=torch.long)

# X_extra, extra_feature_names= build_extra_features(df,global_mean,subject_mean,benchmark_mean,condition_mean,item_mean,U=U,V=V, B=B,prior = True, gap = True, length = False, counts = False, similarity = True, interactions=False, so_priors=None,)
# X_extra = torch.zeros((len(df), 0), dtype=torch.float32)
# extra_feature_names = []

X_extra, extra_feature_names = build_extra_features(
    df,
    global_mean,
    subject_mean,
    benchmark_mean,
    condition_mean,
    item_mean,
    U=None,
    V=None,
    B=None,
    prior=True,
    gap=True,
    length=False,
    counts=False,
    similarity=False,
    interactions=False,
    so_priors=None,
)

print(X_extra.shape)
print(extra_feature_names)


subject_idx_tensor = torch.tensor(
    df["subject_idx"].values,
    dtype=torch.long,
)

# item_idx_tensor = torch.tensor(
#     df["item_idx"].values,
#     dtype=torch.long,
# )

y = torch.tensor(df["label"].values, dtype=torch.float32)

print("X_text shape:", X_text.shape)
print("X_extra shape:", X_extra.shape)
print("y shape:", y.shape)
print("y min/max:", y.min().item(), y.max().item())


#train test split


#prevent leaks
unique_items = df["item_content"].drop_duplicates()

train_items, val_items = train_test_split(
    unique_items,
    test_size=0.2,
    random_state=69,
)

train_mask = df["item_content"].isin(train_items)
val_mask = df["item_content"].isin(val_items)

train_idx = df.index[train_mask].to_numpy()
val_idx = df.index[val_mask].to_numpy()


# train_ds = TensorDataset(
#     X_text[train_idx],
#     X_extra[train_idx],
#     y[train_idx],
# )
#
# val_ds = TensorDataset(
#     X_text[val_idx],
#     X_extra[val_idx],
#     y[val_idx],
# )

train_ds = TensorDataset(
    X_text[train_idx],
    subject_idx_tensor[train_idx],
    benchmark_idx_tensor[train_idx],
    condition_idx_tensor[train_idx],
    X_extra[train_idx],
    y[train_idx],
)

val_ds = TensorDataset(
    X_text[val_idx],
    subject_idx_tensor[val_idx],
    benchmark_idx_tensor[val_idx],
    condition_idx_tensor[val_idx],
    X_extra[val_idx],
    y[val_idx],
)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,)

val_loader = DataLoader(val_ds,batch_size=2048,shuffle=False,)


####model trainning

# model = NCF(
#     d_text=U.shape[1],
#     d_extra=X_extra.shape[1],
# ).to(DEVICE)

model = NCF(
    d_item_text=X_text.shape[1],
    d_extra=X_extra.shape[1],
    n_subjects=len(subject_to_idx),
    n_benchmarks=len(benchmark_to_idx),
    n_conditions=len(condition_to_idx),
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=5e-3,)

loss_fn = nn.BCEWithLogitsLoss()

best_val_loss = float("inf")

bad_epochs=0

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    # for text_x, extra, yb in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
    #     text_x = text_x.to(DEVICE)
    #     extra = extra.to(DEVICE)
    #     yb = yb.to(DEVICE)
    #
    #     optimizer.zero_grad()
    #
    #     logits = model(text_x, extra)
    #     loss = loss_fn(logits, yb)
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_loss_sum += loss.item() * yb.size(0)
    for item_x, subject_idx, benchmark_idx, condition_idx, extra, yb in tqdm(
            train_loader, desc=f"epoch {epoch + 1}"
    ):
        item_x = item_x.to(DEVICE)
        subject_idx = subject_idx.to(DEVICE)
        benchmark_idx = benchmark_idx.to(DEVICE)
        condition_idx = condition_idx.to(DEVICE)
        extra = extra.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(item_x, subject_idx, benchmark_idx, condition_idx, extra)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item() * yb.size(0)

    train_loss = train_loss_sum / len(train_ds)

    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for item_x, subject_idx, benchmark_idx, condition_idx, extra, yb in val_loader:
            item_x = item_x.to(DEVICE)
            subject_idx = subject_idx.to(DEVICE)
            benchmark_idx = benchmark_idx.to(DEVICE)
            condition_idx = condition_idx.to(DEVICE)
            extra = extra.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(
                item_x,
                subject_idx,
                benchmark_idx,
                condition_idx,
                extra,
            )

            loss = loss_fn(logits, yb)

            val_loss_sum += loss.item() * yb.size(0)

    val_loss = val_loss_sum / len(val_ds)

    print(
        f"epoch={epoch + 1}, "
        f"train_loss={train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, "
        f"train_acc={train_acc:.4f}, "
        f"val_acc={val_acc:.4f}"
    )

    #threshold tuning
    val_probs, val_labels = get_probs_labels(model, val_loader)

    best_acc = 0
    best_threshold = 0.5

    for t in np.linspace(0.05, 0.95, 91):
        preds = (val_probs > t).astype(float)
        acc = (preds == val_labels).mean()

        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    print("best threshold:", best_threshold)
    print("best val acc:", best_acc)
    # threshold tuning


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUT_DIR / "ncf_head.pt")
        print("Saved best model.")
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

##save config for submission

save_training_artifacts(
    out_dir=OUT_DIR,
    encoder_name=ENCODER_NAME,
    X_text=X_text,
    X_extra=X_extra,
    extra_feature_names=extra_feature_names,
    subject_to_idx=subject_to_idx,
    benchmark_to_idx=benchmark_to_idx,
    condition_to_idx=condition_to_idx,
    subject_mean=subject_mean,
    benchmark_mean=benchmark_mean,
    condition_mean=condition_mean,
    item_mean=item_mean,
    so_priors=SO_PRIORS,
    global_mean=global_mean,
)

print("Done.")
print("Best validation loss:", best_val_loss)
print("Saved files to:", OUT_DIR.resolve())

