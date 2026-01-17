import os
import pandas as pd
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import spacy
import networkx as nx
import dataclasses
import gc
import io
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import pathway as pw


#Configuration

class GlobalConfig:
    # Google Drive IDs for pathway configuration

    TRAIN_CSV_GDRIVE_ID = OBJECT_ID OF YOUR TRAIN.CSV FILE
    TEST_CSV_GDRIVE_ID = OBJECT_ID OF YOUR TEST.CSV FILE
    BOOKS_DIR_ID = OBJECT_ID OF THE BOOKS DIRECTORY
    CREDENTIALS_PATH = PATH OF YOUR GOOGLE CLOUD CREDETIALS JSON


    TRAIN_PKL = "processed_train_data_pretrained.pkl"
    SUBMISSION_FILE = "Results.csv"

    # Model Hyperparameters
    N_FOLDS = 5
    BATCH_SIZE = 4
    ACCUM_STEPS = 4
    EPOCHS = 25
    LR = 1e-4
    SEED = 42

    # Feature Config
    CHUNK_SIZE = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclasses.dataclass
class ModelConfig:
    n_embd: int = 128          # BDH embedding dimension
    n_head: int = 4            # Attention heads
    n_feat: int = 775          # 770 pretrained + 3 graph + 2 NLI
    seq_len: int = 16          # Sequence length for hypothesis tokens
    dropout: float = 0.3

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pathway data loading

def load_pathway_data():
    """Loads Train, Test, and Books from Google Drive via Pathway."""

    # Load Train CSV
    print("Loading train.csv...")
    try:
        train_table = pw.io.gdrive.read(
            object_id=GlobalConfig.TRAIN_CSV_GDRIVE_ID,
            service_user_credentials_file=GlobalConfig.CREDENTIALS_PATH,
            mode="static",
            format="binary"
        )
        train_result = pw.debug.table_to_pandas(train_table)
        csv_content = train_result['data'].iloc[0].decode('utf-8')
        train_df = pd.read_csv(io.StringIO(csv_content))
        print(f"Train Loaded: {len(train_df)} rows")
    except Exception as e:
        print(f"Error loading Train: {e}")
        train_df = pd.DataFrame()

    # Load Test CSV
    print("Loading test.csv...")
    try:
        test_table = pw.io.gdrive.read(
            object_id=GlobalConfig.TEST_CSV_GDRIVE_ID,
            service_user_credentials_file=GlobalConfig.CREDENTIALS_PATH,
            mode="static",
            format="binary"
        )
        test_result = pw.debug.table_to_pandas(test_table)
        csv_content = test_result['data'].iloc[0].decode('utf-8')
        test_df = pd.read_csv(io.StringIO(csv_content))
        print(f"Test Loaded: {len(test_df)} rows")
    except Exception as e:
        print(f"Error loading Test: {e}")
        test_df = pd.DataFrame()

    # Load Books Directory
    print("Loading Books Directory...")
    try:
        books_table = pw.io.gdrive.read(
            object_id=GlobalConfig.BOOKS_DIR_ID,
            service_user_credentials_file=GlobalConfig.CREDENTIALS_PATH,
            mode="static",
            format="binary",
            with_metadata=True
        )
        books_df = pw.debug.table_to_pandas(books_table)
        print(f"Books Loaded: {len(books_df)} files")
    except Exception as e:
        print(f"Error loading Books: {e}")
        books_df = pd.DataFrame()

    return train_df, test_df, books_df


#helper function for normalizing and finding books

def normalize_name(name):
    """Normalize book names for matching"""
    if not name: return ""
    return str(name).lower().replace(".txt", "").replace(" ", "").replace('"', '').replace("'", "")

def find_book_content_in_df(books_df, book_name):
    """
    Finds a book in the Pathway DataFrame and returns its text content.
    Returns: text_content (str) or None
    """
    if books_df.empty: return None

    target_clean = normalize_name(book_name)

    # Iterate to find match (exact match on normalized name)
    for idx, row in books_df.iterrows():
        file_raw = str(row['_metadata']['name'])
        file_clean = normalize_name(file_raw)
        if file_clean == target_clean:
            try:
                return row['data'].decode('utf-8', errors='ignore')
            except:
                return None
    return None


# Feature engineering and extraction

class UnifiedFeatureEngine:
    """
    Handles feature extraction combining Geometric (Embeddings), Graph, and NLI features.
    """
    def __init__(self, train_mean=None, train_std=None):
        print("Loading Feature Models...")

        # Geometric Encoder (Sentence Transformers)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.retriever = self.encoder
        print("Loaded sentence encoder (384-dim)")

        #  NLI Model
        self.nli = CrossEncoder('cross-encoder/nli-deberta-v3-base', device=device)
        print("Loaded NLI cross-encoder")

        # Graph/Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spacy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Normalization stats (for inference)
        self.train_mean = train_mean
        self.train_std = train_std
        print("Feature Engine Ready")

    def get_diverse_chunks(self, book_text):
        """Sample chunks from beginning, middle, and end."""
        chunk_size = GlobalConfig.CHUNK_SIZE
        book_len = len(book_text)
        if book_len < chunk_size:
            return [book_text]

        chunks = []
        # Beginning chunk
        chunks.extend([book_text[i:i+chunk_size] for i in range(0, int(book_len * 0.2), chunk_size)])
        # Middle chunk
        chunks.extend([book_text[i:i+chunk_size] for i in range(int(book_len * 0.4), int(book_len * 0.6), chunk_size)])
        # End chunk
        chunks.extend([book_text[i:i+chunk_size] for i in range(int(book_len * 0.8), book_len, chunk_size)])

        return chunks

    def get_graph_features(self, text):
        """Extract character interaction graph features (3 dims)."""
        G = nx.Graph()
        doc = self.nlp(text[:30000])

        for sent in doc.sents:
            chars = [e.text for e in sent.ents if e.label_ == "PERSON"]
            if len(chars) > 1:
                for i in range(len(chars)):
                    for j in range(i+1, len(chars)):
                        G.add_edge(chars[i], chars[j])

        if len(G) == 0:
            return [0.0, 0.0, 0.0]

        pr = list(nx.pagerank(G).values())
        return [
            np.mean(pr) if pr else 0,
            nx.density(G),
            len(G.edges) / max(1, len(G.nodes))
        ]

    def get_nli_features(self, backstory, book_text):
        """Semantic search + NLI (2 dims)."""
        try:
            if len(book_text) < 100: return [0.0, 0.0]

            full_chunks = self.get_diverse_chunks(book_text)
            if not full_chunks: return [0.0, 0.0]

            # Semantic Search
            query_emb = self.retriever.encode(backstory, convert_to_tensor=True)
            chunk_embs = self.retriever.encode(full_chunks, convert_to_tensor=True, batch_size=32, show_progress_bar=False)

            top_k = min(5, len(full_chunks))
            hits = util.semantic_search(query_emb, chunk_embs, top_k=top_k)[0]
            relevant_chunks = [full_chunks[hit['corpus_id']] for hit in hits]

            # NLI
            pairs = [(backstory, chunk) for chunk in relevant_chunks]
            scores = self.nli.predict(pairs)
            probs = torch.softmax(torch.tensor(scores), dim=1)

            # Aggregate Top contradictions
            contradiction_probs = probs[:, 0]
            k_contradict = min(3, len(contradiction_probs))
            topk_indices = contradiction_probs.topk(k_contradict).indices
            nli_probs = probs[topk_indices].mean(dim=0).tolist()

            return nli_probs[:2] # Contradiction, Entailment
        except:
            return [0.0, 0.0]

    def get_pretrained_embeddings(self, backstory, book_text):

        # Encode Backstory
        backstory_emb = self.encoder.encode(backstory, convert_to_tensor=True, show_progress_bar=False)

        # Encode Book (Beg/Mid/End)
        book_len = len(book_text)
        excerpts = [
            book_text[:2000],
            book_text[book_len//2-1000:book_len//2+1000],
            book_text[-2000:] if book_len > 2000 else book_text
        ]
        book_embs = self.encoder.encode(excerpts, convert_to_tensor=True, show_progress_bar=False)
        book_emb = book_embs.mean(dim=0)

        # Interaction Features
        cos_sim = torch.nn.functional.cosine_similarity(backstory_emb.unsqueeze(0), book_emb.unsqueeze(0)).item()
        l2_dist = torch.norm(backstory_emb - book_emb).item()

        combined = torch.cat([
            backstory_emb.cpu(),
            book_emb.cpu(),
            torch.tensor([cos_sim, l2_dist])
        ])
        return combined.numpy()

    def extract_all(self, backstory, book_text, normalize=False):
        """Master method to get the 775-dim vector."""
        emb_feats = self.get_pretrained_embeddings(backstory, book_text)
        graph_feats = self.get_graph_features(book_text)
        nli_feats = self.get_nli_features(backstory, book_text)

        all_features = np.concatenate([
            emb_feats,
            np.array(graph_feats),
            np.array(nli_feats)
        ])

        # Apply normalization if inference
        if normalize and self.train_mean is not None:
            all_features = (all_features - self.train_mean) / self.train_std

        return {
            'features': all_features,
            'nli_raw': nli_feats
        }

# BDH Model Architecture

class GraphFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_proj = nn.Linear(config.n_feat, config.n_embd)
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, features):
        f = self.feat_proj(features).unsqueeze(1).expand(-1, x.size(1), -1)
        combined = torch.cat([x, f], dim=-1)
        gate = self.gate(combined)
        fused = x * (1 - gate) + f * gate
        return self.dropout(fused)

class BabyDragonHatchling(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.n_feat, config.n_embd * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd * 2, config.n_embd)
        )
        self.hypothesis_tokens = nn.Parameter(torch.randn(1, config.seq_len, config.n_embd) * 0.02)
        self.fusion = GraphFusion(config)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=config.n_embd, nhead=config.n_head,
            dim_feedforward=config.n_embd * 4, dropout=config.dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, features):
        B = features.size(0)
        h_tokens = self.hypothesis_tokens.expand(B, -1, -1)
        x = self.fusion(h_tokens, features)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])


# Dataset and training Logic
class PretrainedDataset(Dataset):
    def __init__(self, data_list, mean=None, std=None):
        self.data = data_list
        feats = np.array([d['features'] for d in self.data])
        self.mean = feats.mean(axis=0) if mean is None else mean
        self.std = (feats.std(axis=0) + 1e-6) if std is None else std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        norm_feat = (np.array(item['features']) - self.mean) / self.std
        label = item.get('label', 0.0)
        if torch.is_tensor(label): label = label.item()

        return {
            'features': torch.tensor(norm_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

def train_one_fold(fold_idx, train_ds, val_ds, config):

    labels = [d['label'].item() for d in train_ds]
    counts = [labels.count(0), labels.count(1)]
    if counts[0] == 0: counts[0] = 1
    if counts[1] == 0: counts[1] = 1

    weights = [1.0/counts[int(l)] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=GlobalConfig.BATCH_SIZE, sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=GlobalConfig.BATCH_SIZE, shuffle=False)

    model = BabyDragonHatchling(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=GlobalConfig.LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    pos_weight = torch.tensor([counts[0] / max(1, counts[1])]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    best_thresh = 0.5
    patience = 8
    counter = 0

    print(f"\n Goal: Fold {fold_idx+1} Training...")

    for epoch in range(GlobalConfig.EPOCHS):
        model.train()
        for i, batch in enumerate(train_dl):
            loss = criterion(model(batch['features'].to(device)), batch['label'].to(device).unsqueeze(1))
            loss = loss / GlobalConfig.ACCUM_STEPS
            loss.backward()
            if (i+1) % GlobalConfig.ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = model(batch['features'].to(device))
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                trues.extend(batch['label'].cpu().numpy().flatten())

        # Threshold Search
        curr_best_f1 = 0
        curr_thresh = 0.5
        for t in np.linspace(0.2, 0.8, 61):
            f1 = f1_score(trues, (np.array(probs) > t).astype(int), zero_division=0)
            if f1 > curr_best_f1: curr_best_f1, curr_thresh = f1, t

        scheduler.step(curr_best_f1)

        if curr_best_f1 >= best_f1:
            best_f1, best_thresh = curr_best_f1, curr_thresh
            counter = 0
            torch.save({
                'model': model.state_dict(),
                'threshold': best_thresh,
                'config': config
            }, f"bdh_fold_{fold_idx}_best.pth")
        else:
            counter += 1
            if counter >= patience: break

    return best_f1


# Orchestration of Pipeline
def run_preprocessing(train_df, books_df):
    """Step 1: Convert raw books/backstories into feature vectors."""


    engine = UnifiedFeatureEngine()
    processed_data = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting Features"):
        book_name = row['book_name']

        # Find book content in Pathway DataFrame
        text = find_book_content_in_df(books_df, book_name)
        if not text:
            print(f"Missing book: {book_name}")
            continue

        # Label parsing
        lbl_str = str(row.get('label', 1)).lower()
        label_val = 0.0 if 'contra' in lbl_str else 1.0

        # Extract features (Unified 775-dim)
        feats = engine.extract_all(row.get('content', ''), text, normalize=False)
        feats['label'] = label_val
        processed_data.append(feats)

    with open(GlobalConfig.TRAIN_PKL, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved {len(processed_data)} samples to {GlobalConfig.TRAIN_PKL}")

def run_training():
    if not os.path.exists(GlobalConfig.TRAIN_PKL):
        print("Data not found. Run preprocessing first.")
        return

    print("\n Training BDH")

    with open(GlobalConfig.TRAIN_PKL, 'rb') as f:
        raw_data = pickle.load(f)

    if len(raw_data) == 0:
        print("No training data available.")
        return

    # Init main dataset to get stats
    full_ds = PretrainedDataset(raw_data)
    labels = [d['label'] for d in raw_data]

    skf = StratifiedKFold(n_splits=GlobalConfig.N_FOLDS, shuffle=True, random_state=GlobalConfig.SEED)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_data, labels)):
        train_sub = [raw_data[i] for i in train_idx]
        val_sub = [raw_data[i] for i in val_idx]

        # Create datasets using global mean/std from full_ds to ensure consistency
        train_ds = PretrainedDataset(train_sub, mean=full_ds.mean, std=full_ds.std)
        val_ds = PretrainedDataset(val_sub, mean=full_ds.mean, std=full_ds.std)

        f1 = train_one_fold(fold, train_ds, val_ds, ModelConfig())
        scores.append(f1)
        print(f"Fold {fold+1} Best F1: {f1:.2%}")

    print(f"\n Mean F1: {np.mean(scores):.2%} ± {np.std(scores):.2%}")


# Generation of submission on test csv

def run_inference(test_df, books_df):

    print("\n Inference")

    if test_df.empty:
        print("Test DF is empty.")
        return

    # Load stats from training data for normalization
    try:
        with open(GlobalConfig.TRAIN_PKL, 'rb') as f:
            train_data = pickle.load(f)
        train_feats = np.array([d['features'] for d in train_data])
        mean, std = train_feats.mean(axis=0), train_feats.std(axis=0) + 1e-6
    except:
        print("Training data missing, calculating generic stats (not ideal)")
        mean, std = None, None

    # Load Models
    models, thresholds = [], []
    for i in range(GlobalConfig.N_FOLDS):
        path = f"bdh_fold_{i}_best.pth"
        if os.path.exists(path):

            ckpt = torch.load(path, map_location=device, weights_only=False)

            m = BabyDragonHatchling(ckpt['config']).to(device)
            m.load_state_dict(ckpt['model'])
            m.eval()
            models.append(m)
            thresholds.append(ckpt.get('threshold', 0.5))

    if not models:
        print(" No models found!")
        return

    avg_thresh = np.mean(thresholds)
    print(f"Loaded {len(models)} models. Avg Threshold: {avg_thresh:.3f}")

    # Process
    engine = UnifiedFeatureEngine(train_mean=mean, train_std=std)
    results = []
    stats = {'consistent': 0, 'contradict': 0}

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        row_id = row['id']
        book_name = row['book_name']

        # Find book content
        text = find_book_content_in_df(books_df, book_name)

        # Fail-safe defaults
        if not text:
            results.append({'id': row_id, 'label': 1, 'rationale': 'Book missing'})
            stats['consistent'] += 1
            continue

        try:
            # Extract & Normalize
            extracted = engine.extract_all(row.get('content', ''), text, normalize=True)
            feat_tensor = torch.tensor(extracted['features'], dtype=torch.float32).unsqueeze(0).to(device)

            # Ensemble Prediction
            probs = []
            with torch.no_grad():
                for m in models:
                    probs.append(torch.sigmoid(m(feat_tensor)).item())

            avg_prob = np.mean(probs)
            pred = 1 if avg_prob > avg_thresh else 0

            # Rationale
            rat = f"Contradiction (Prob: {avg_prob:.2f})" if pred == 0 else f"Consistent (Prob: {avg_prob:.2f})"
            results.append({'id': row_id, 'label': pred, 'rationale': rat})

            if pred == 1: stats['consistent'] += 1
            else: stats['contradict'] += 1

        except Exception as e:
            results.append({'id': row_id, 'label': 1, 'rationale': str(e)})

    pd.DataFrame(results).to_csv(GlobalConfig.SUBMISSION_FILE, index=False)
    print(f"\n Stats: {stats}")
    print(f"Submission saved to {GlobalConfig.SUBMISSION_FILE}")

# Main orchestrator

if __name__ == "__main__":
    set_seed(GlobalConfig.SEED)

    # Load Data from Drive via Pathway
    train_df, test_df, books_df = load_pathway_data()

    if not train_df.empty and not books_df.empty:
        # Preprocess
        run_preprocessing(train_df, books_df)

        # Train
        run_training()

        # Inference
        run_inference(test_df, books_df)
    else:
        print("Failed to load initial data. Check GDrive IDs.")


