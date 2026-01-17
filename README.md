# BDH-Graph Augmented Narrative Reasoning (Track B)

## Team: deepmindset

This repository contains our solution for **Track B** of the Kharagpur Data Science Hackathon 2026.  
The task focuses on **global consistency checking** between a *hypothetical backstory* and a *full-length narrative (100k+ words)*.

Our approach goes beyond conventional RAG or long-context Transformers by explicitly modeling **causal and social structure** in narratives using **graph-based reasoning inspired by the Baby Dragon Hatchling (BDH) architecture**.

---

## 🚀 Problem Overview (Track B)

**Objective:**  
Given:
- A **Backstory**
- A **Full Novel / Narrative** (100k+ words)

Predict a binary label:
- `1` → Consistent  
- `0` → Contradictory  

**Key Challenge:**  
Transformers struggle to maintain **long-range state constraints** across very long documents. Most of the text is *noise*, while only a small subset contains **causal signals** that determine consistency.

---

## 💡 Why Track B?

We chose Track B to explore **local, distributed graph dynamics** instead of relying on black-box attention mechanisms.  
Track B aligns naturally with BDH principles, where **intelligence emerges from topology, sparsity, and state evolution**, not global attention.

---

## 🧠 Core Idea

We treat a narrative as a **dynamic causal graph**, not just text.

- **Characters = Nodes**
- **Interactions & events = Edges**
- **Contradictions = Structural violations in graph state**

By fusing:
- Semantic understanding (embeddings),
- Logical reasoning (NLI),
- Structural signals (graph topology),

we build a system that **reasons about narratives**, rather than merely scoring text similarity.

---

## 🏗️ System Architecture

### 1. Data Ingestion (Pathway)
- Streams large narrative files (local or Google Drive)
- Parses PDFs/TXT into clean text
- Handles long-context efficiently via streaming

### 2. Unified Feature Engine

**Branch A – Semantic**
- SentenceTransformer (`all-MiniLM-L6-v2`)
- 770-dimensional embeddings

**Branch B – Graph Topology**
- Entity extraction with SpaCy
- NetworkX graph construction
- Graph features:
  - PageRank mean
  - Graph density
  - Edge-to-node ratio
  - Connected components

**Branch C – Logic (NLI)**
- DeBERTa-v3 Cross-Encoder
- Probabilities for:
  - Entailment
  - Contradiction
  - Neutral

---

## 🔀 Graph Fusion Module

A **gated fusion mechanism** dynamically decides:
- When to trust **semantic/NLI signals**
- When to rely on **structural graph constraints**

This acts as a **structural veto system**:
- Semantic ambiguity → resolved by graph topology
- Local events → filtered as noise
- Causal events → reflected as graph changes

---

## 🐉 BDH Reasoning Core

We use a **Baby Dragon Hatchling–inspired model** with:

- **775D fused input**
  - 770 semantic
  - 3 graph topology
  - 2 NLI logic
- **Learnable hypothesis tokens**
  - Act as causal “queries” (death, travel, interaction, etc.)
- **Self-attention over fused state**
- **Sigmoid classifier for final prediction**

---

## 📚 Handling 100k+ Words

We apply a **Filter-and-Focus strategy**:

1. **Entity-centric filtering**
2. **Graph compression of entire novel**
3. **Top-k semantic chunk retrieval**
4. **Reduced NLI context (~2k tokens)**

This enables efficient long-context reasoning without re-reading the entire book.

---

## 📊 Training & Evaluation

- **Stratified K-Fold Cross Validation (N=5)**
- Ensemble inference across folds

### Results
- **Training Accuracy:** 91%
- **Validation F1 Score:** ~87%
- **Inference Time:** < 10 minutes (local GPU)
- **Stability:** Low variance across folds

---

## ⚠️ Limitations

- Multi-hop implicit logic may be missed
- Sarcasm and irony can confuse NLI
- Setting-based contradictions (non-entity) are harder to detect
- Extremely dense graphs may increase memory usage

---

## 🧩 Key Technologies

- **Pathway** – Streaming ETL & stateful processing
- **NetworkX** – Graph construction & topology
- **SpaCy** – Entity extraction
- **SentenceTransformers** – Semantic embeddings
- **DeBERTa v3** – Natural Language Inference
- **PyTorch** – Model implementation

---

## 🎯 Conclusion

Our solution shifts narrative consistency checking from **correlation to causality**.

By combining:
- Graph structure,
- Logical constraints,
- Semantic understanding,

we demonstrate that **long-context reasoning is better solved by stateful graph dynamics than brute-force attention**.

This architecture provides a scalable, interpretable, and production-ready blueprint for future long-context AI systems.

---

## 📌 References

- *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*
- Pathway BDH Reference Implementation
- Kharagpur Data Science Hackathon 2026 Problem Statement
