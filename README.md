# BDH-Graph Augmented Narrative Reasoning (Track B)

## Team: deepmindset

This repository contains our solution for **Track B** of the Kharagpur Data Science Hackathon 2026.  
The task focuses on **global consistency checking** between a *hypothetical backstory* and a *full-length narrative (100k+ words)*.

Our approach goes beyond conventional RAG or long-context Transformers by explicitly modeling **causal and social structure** in narratives using **graph-based reasoning inspired by the Baby Dragon Hatchling (BDH) architecture**.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d7c9576c-5cc8-49c5-9734-cd101d5b2af5" alt="BDH Architecture" style="border-radius: 50%; width: 300px; height: 300px; object-fit: cover;">
</div>


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

<img width="1716" height="919" alt="Screenshot 2026-01-17 200517" src="https://github.com/user-attachments/assets/b5b59de9-b2f4-42cc-9701-79cd165d7d4f" />

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

![WhatsApp Image 2026-01-16 at 12 58 47 PM](https://github.com/user-attachments/assets/2ade83a3-ae61-48fe-9d9c-61bcdde47792)

By fusing:
- Semantic understanding (embeddings),
- Logical reasoning (NLI),
- Structural signals (graph topology),

we build a system that **reasons about narratives**, rather than merely scoring text similarity.

---

## 🏗️ System Architecture   

![WhatsApp Image 2026-01-16 at 10 18 57 PM](https://github.com/user-attachments/assets/7f3c7fde-fb94-4d81-9331-28a76c3908e4)

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

![WhatsApp Image 2026-01-16 at 10 22 51 PM](https://github.com/user-attachments/assets/7ac327d7-adc6-471f-b246-7083b7a9a19d)

---

## 🔀 Graph Fusion Module

A **gated fusion mechanism** dynamically decides:

- When to trust **high-dimensional semantic embeddings (Branch A)**
- When to rely on **graph-derived topological features (Branch B)**

This acts as a **structural arbitration system**:
- Semantic ambiguity → resolved via graph topology
- Spurious local correlations → suppressed by structural context
- Causal or persistent patterns → reinforced through graph signals

The fusion is governed by a **learnable gate**:

$$h_{fused} = \sigma(W_g \cdot [x_{emb}, f_{graph}]) \odot x_{emb} + (1 - \sigma(W_g \cdot [x_{emb}, f_{graph}])) \odot f_{graph}$$

**Where:**
- $x_{emb}$ denotes the semantic embedding vector
- $f_{graph}$ denotes the graph-based structural feature vector
- $W_g$ is a learnable gating matrix
- $\sigma(\cdot)$ is the sigmoid activation function
- $\odot$ represents element-wise multiplication

This formulation enables **adaptive fusion**, allowing the model to shift between semantic reasoning and structural reasoning depending on confidence and context.



---

## 🐉 BDH Reasoning Core

![WhatsApp Image 2026-01-16 at 10 20 51 PM](https://github.com/user-attachments/assets/7b1561e8-333e-4745-ae49-1f35df45df1e)

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

<img width="1722" height="925" alt="Screenshot 2026-01-17 200621" src="https://github.com/user-attachments/assets/b553f759-33a7-4539-ba7c-ae3ef70eff61" />

---

## 📌 References

- *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*
- Pathway BDH Reference Implementation
- Kharagpur Data Science Hackathon 2026 Problem Statement
