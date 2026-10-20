# Latent Dirichlet Allocation (LDA) Topic Modeling Experiment

This repository contains an end-to-end implementation for **NLP Lab Experiment 4**:
> **Objective:** To apply Latent Dirichlet Allocation on a document collection and interpret the discovered topics using probabilistic and sampling-based methods.

---

## 📁 Project Structure

```
lda_topic_modeling/
├── data/
│   └── raw_documents.csv           # 25 curated multi-domain documents across AI, Space, Energy, Health, Finance
├── src/
│   ├── __init__.py
│   ├── preprocessor.py             # spaCy / NLTK text cleaner, lemmatizer, and bigram builder
│   ├── vectorizer.py               # CountVectorizer DTM & Gensim BoW dictionary representations
│   ├── lda_model.py                # Scikit-Learn & Gensim Variational Bayes LDA model
│   ├── gibbs_sampler.py            # Custom Collapsed Gibbs Sampling implementation of LDA
│   └── visualizer.py               # Publication-ready plots (bar charts, heatmaps, t-SNE, HTML report)
├── results/                        # Generated deliverables
│   ├── preprocessed_corpus.csv     # Deliverable: Preprocessed corpus (tokens & clean text)
│   ├── preprocessed_corpus.json    # Deliverable: Preprocessed corpus in JSON format
│   ├── topic_word_distributions.csv# Deliverable: Topic-word posterior distributions
│   ├── document_topic_distributions.csv # Deliverable: Document classification & dominant topics
│   ├── topic_bar_charts.png        # Deliverable: Topic keyword distribution charts
│   ├── document_topic_heatmap.png  # Deliverable: Document-topic composition heatmap
│   ├── topic_clusters_tsne.png     # Deliverable: 2D t-SNE projection of document clusters
│   ├── gibbs_sampling_convergence.png # Deliverable: Gibbs sampling log-likelihood convergence
│   └── topic_model_interactive.html # Deliverable: Interactive HTML dashboard
├── EXPERIMENT_4_LAB_REPORT.md      # Full academic report matching CO4 & Bloom's L3
├── lda_experiment_notebook.ipynb   # Interactive Jupyter Notebook for demonstration
├── run_experiment.py               # Master end-to-end execution script
└── README.md
```

---

## 🚀 Quickstart & Execution

### 1. Run the Complete Pipeline
To execute the preprocessing, train both Variational and Gibbs LDA models, and produce all deliverables:

```bash
python run_experiment.py
```

### 2. View Interactive Deliverable
Open `results/topic_model_interactive.html` in any web browser to explore discovered topics, keyword weights, and document classifications interactively.

### 3. Open the Jupyter Notebook
```bash
jupyter notebook lda_experiment_notebook.ipynb
```

---

## 📊 Deliverables Checklist

- [x] **Preprocessed corpus:** `results/preprocessed_corpus.csv` & `results/preprocessed_corpus.json`
- [x] **LDA model output:** Log-likelihood, perplexity, and fitted parameters in `run_experiment.py`
- [x] **Topic-word lists:** `results/topic_word_distributions.csv`
- [x] **Topic interpretation report:** `EXPERIMENT_4_LAB_REPORT.md`
- [x] **Visualizations:** Bar charts, Document-Topic heatmap, 2D t-SNE cluster projection, and interactive HTML dashboard.
