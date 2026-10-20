import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.preprocessor import TextPreprocessor
from src.vectorizer import DocumentTermRepresentation
from src.lda_model import LDATrainer
from src.gibbs_sampler import CollapsedGibbsLDA
from src.visualizer import TopicVisualizer


def run_lda_pipeline(data_path: str = "data/raw_documents.csv", output_dir: str = "results", n_topics: int = 5):
    print("=" * 75)
    print("  EXPERIMENT 4: LATENT DIRICHLET ALLOCATION (LDA) TOPIC MODELING")
    print("  Methods: Probabilistic (Variational Bayes) & Sampling-based (Gibbs)")
    print("=" * 75)
    
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load Document Collection
    # -------------------------------------------------------------
    print(f"\n[Step 1] Loading document corpus from: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw)} documents across {df_raw['domain'].nunique()} domains.")
    for domain, group in df_raw.groupby('domain'):
        print(f"  * {domain}: {len(group)} documents")

    # -------------------------------------------------------------
    # 2. Text Preprocessing (Deliverable: Preprocessed Corpus)
    # -------------------------------------------------------------
    print("\n[Step 2] Preprocessing text corpus (Cleaning, Tokenization, Lemmatization, Bigrams)...")
    preprocessor = TextPreprocessor(min_token_len=3)
    raw_texts = df_raw["text"].tolist()
    doc_titles = df_raw["title"].tolist()
    
    tokenized_docs, joined_docs = preprocessor.preprocess_corpus(raw_texts, use_bigrams=True)

    # Save Preprocessed Corpus Deliverable
    df_preprocessed = df_raw.copy()
    df_preprocessed["clean_tokens"] = [", ".join(tokens) for tokens in tokenized_docs]
    df_preprocessed["token_count"] = [len(tokens) for tokens in tokenized_docs]
    df_preprocessed["processed_text"] = joined_docs
    
    preprocessed_csv_path = os.path.join(output_dir, "preprocessed_corpus.csv")
    df_preprocessed.to_csv(preprocessed_csv_path, index=False)
    
    # Save JSON deliverable
    preprocessed_json_path = os.path.join(output_dir, "preprocessed_corpus.json")
    with open(preprocessed_json_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "doc_id": int(row["doc_id"]),
                "domain": row["domain"],
                "title": row["title"],
                "tokens": tokenized_docs[i]
            }
            for i, (_, row) in enumerate(df_raw.iterrows())
        ], f, indent=2)

    total_tokens = sum(len(d) for d in tokenized_docs)
    avg_tokens = np.mean([len(d) for d in tokenized_docs])
    print(f"[OK] Preprocessing complete. Total processed tokens: {total_tokens} (Avg {avg_tokens:.1f} tokens/doc)")
    print(f"  Sample processed doc 1 tokens:\n  {tokenized_docs[0][:10]}...")
    print(f"[OK] Saved preprocessed corpus to:\n  - {preprocessed_csv_path}\n  - {preprocessed_json_path}")

    # -------------------------------------------------------------
    # 3. Document-Term Representation (Outcome 2)
    # -------------------------------------------------------------
    print("\n[Step 3] Constructing Document-Term Matrix (DTM) and Bag-of-Words representation...")
    dtm_builder = DocumentTermRepresentation(min_df=1, max_df=0.90)
    dtm, vocab = dtm_builder.fit_transform_sklearn(joined_docs)
    gensim_dict, gensim_bow = dtm_builder.fit_transform_gensim(tokenized_docs)
    
    sparsity = dtm_builder.get_sparsity()
    print(f"[OK] DTM Shape: {dtm.shape[0]} documents x {dtm.shape[1]} unique vocabulary terms")
    print(f"[OK] Vocabulary size (V): {len(vocab)}")
    print(f"[OK] DTM Sparsity: {sparsity:.2f}%")
    
    top_words_df = dtm_builder.get_top_frequent_words(top_n=8)
    print("  Top frequent terms in corpus:")
    for _, r in top_words_df.iterrows():
        print(f"    * {r['Word']}: {int(r['Frequency'])} occurrences")

    # -------------------------------------------------------------
    # 4. Train LDA Model (Variational Bayes) (Outcome 3 & 4)
    # -------------------------------------------------------------
    print(f"\n[Step 4] Training LDA Model via Variational Inference (K = {n_topics} topics)...")
    lda_trainer = LDATrainer(n_topics=n_topics, learning_method="batch", max_iter=60, random_state=42)
    sklearn_lda = lda_trainer.train_sklearn(dtm, vocab, doc_topic_prior=0.1, topic_word_prior=0.01)
    
    metrics = lda_trainer.evaluate_model(dtm)
    print(f"[OK] Model Convergence Reached!")
    print(f"  - Log-Likelihood: {metrics['log_likelihood']:.2f}")
    print(f"  - Perplexity:     {metrics['perplexity']:.2f}")

    # Extract Topic-Word Lists (Deliverable)
    topic_words = lda_trainer.get_top_topic_words(top_n=10)
    topic_words_df = lda_trainer.get_topic_word_dataframe(top_n=10)
    topic_words_csv_path = os.path.join(output_dir, "topic_word_distributions.csv")
    topic_words_df.to_csv(topic_words_csv_path, index=False)
    print(f"[OK] Exported Topic-Word distributions to: {topic_words_csv_path}")

    # Automatic Dynamic Topic Labeling based on semantic keyword scoring
    domain_lexicons = {
        "Artificial Intelligence & Machine Learning": {
            "model", "neural", "network", "neural_network", "deep", "learning", "algorithm",
            "transformer", "vision", "training", "intelligence", "inference", "agent", "bias", "reinforcement_learning"
        },
        "Space Exploration & Astrophysics": {
            "space", "star", "telescope", "planet", "astronomy", "astronomer", "rover", "orbit",
            "cosmic", "universe", "gravitational", "galaxy", "lunar", "mars"
        },
        "Clean Energy & Climate Technologies": {
            "energy", "solar", "wind", "battery", "grid", "power", "clean", "hydrogen",
            "electricity", "turbine", "carbon", "storage", "photovoltaic"
        },
        "Biomedical Science & Gene Therapies": {
            "cell", "gene", "therapy", "disease", "patient", "immune", "crispr", "vaccine",
            "clinical", "medical", "dna", "mrna", "antigen", "cardiovascular"
        },
        "Macroeconomics & Financial Markets": {
            "market", "rate", "bank", "inflation", "liquidity", "price", "yield", "financial",
            "asset", "trading", "debt", "monetary", "recession", "central_bank"
        }
    }
    
    topic_labels: Dict[int, str] = {}
    assigned_domains = set()
    for topic_id, words in topic_words.items():
        best_domain = None
        best_score = -1.0
        for dom_name, dom_keywords in domain_lexicons.items():
            if dom_name in assigned_domains:
                continue
            score = sum(p for w, p in words if (w in dom_keywords or any(part in dom_keywords for part in w.split('_'))))
            if score > best_score:
                best_score = score
                best_domain = dom_name
        if best_domain and best_score > 0:
            topic_labels[topic_id] = best_domain
            assigned_domains.add(best_domain)
        else:
            topic_labels[topic_id] = f"Topic {topic_id + 1}"

    print("\n--- Discovered Topics & Top Keywords (Probabilities) ---")
    for topic_id, words in topic_words.items():
        label = topic_labels.get(topic_id, f"Topic {topic_id+1}")
        word_str = ", ".join([f"{w} ({p:.3f})" for w, p in words[:6]])
        print(f"  * Topic {topic_id + 1} [{label}]:\n    {word_str}")

    # Extract Document-Topic Distributions (Deliverable)
    doc_dist_df = lda_trainer.get_document_topic_dataframe(doc_titles, raw_texts)
    doc_dist_csv_path = os.path.join(output_dir, "document_topic_distributions.csv")
    doc_dist_df.to_csv(doc_dist_csv_path, index=False)
    print(f"[OK] Exported Document-Topic distributions to: {doc_dist_csv_path}")

    # -------------------------------------------------------------
    # 5. Sampling-based LDA: Collapsed Gibbs Sampling
    # -------------------------------------------------------------
    print(f"\n[Step 5] Training Sampling-based LDA using Collapsed Gibbs Sampling...")
    gibbs_lda = CollapsedGibbsLDA(n_topics=n_topics, alpha=0.5, beta=0.1, n_iter=100, random_state=42)
    gibbs_lda.fit(tokenized_docs, vocab=vocab)
    gibbs_topic_words = gibbs_lda.get_top_topic_words(top_n=6)
    print("[OK] Collapsed Gibbs Sampling completed 100 iterations.")
    print("  Gibbs Discovered Topics Sample:")
    for k, words in list(gibbs_topic_words.items())[:3]:
        print(f"    Topic {k+1}: {', '.join([w[0] for w in words])}")

    # -------------------------------------------------------------
    # 6. Generate Rich Visualizations (Deliverables)
    # -------------------------------------------------------------
    print("\n[Step 6] Generating Visualizations...")
    visualizer = TopicVisualizer(output_dir=output_dir)

    # Bar Charts
    bar_path = visualizer.plot_topic_word_distributions(topic_words, topic_labels=topic_labels, top_n=8)
    print(f"[OK] Saved Topic-Word Bar Charts: {bar_path}")

    # Heatmap
    heatmap_path = visualizer.plot_document_topic_heatmap(
        lda_trainer.doc_topic_dist,
        doc_titles=doc_titles,
        topic_labels=[f"T{k+1}: {topic_labels[k][:18]}.." for k in range(n_topics)]
    )
    print(f"[OK] Saved Document-Topic Heatmap: {heatmap_path}")

    # t-SNE Projection
    tsne_path = visualizer.plot_topic_clusters_tsne(
        lda_trainer.doc_topic_dist,
        doc_titles=doc_titles,
        domains=df_raw["domain"].tolist()
    )
    print(f"[OK] Saved 2D t-SNE Cluster Plot: {tsne_path}")

    # Gibbs Sampling Convergence
    gibbs_path = visualizer.plot_gibbs_convergence(gibbs_lda.log_likelihood_history)
    print(f"[OK] Saved Gibbs Convergence Plot: {gibbs_path}")

    # Interactive HTML Dashboard
    html_path = visualizer.generate_interactive_dashboard_html(
        topic_words,
        doc_dist_df,
        topic_labels=topic_labels,
        metrics=metrics,
        sparsity=sparsity,
        vocab_size=len(vocab)
    )
    print(f"[OK] Saved Interactive HTML Dashboard: {html_path}")

    print("\n" + "=" * 75)
    print("  ALL LDA DELIVERABLES GENERATED SUCCESSFULLY!")
    print("=" * 75)

    return {
        "preprocessed_csv": preprocessed_csv_path,
        "preprocessed_json": preprocessed_json_path,
        "topic_words_csv": topic_words_csv_path,
        "doc_dist_csv": doc_dist_csv_path,
        "visualizations": [bar_path, heatmap_path, tsne_path, gibbs_path, html_path],
        "metrics": metrics
    }


if __name__ == "__main__":
    run_lda_pipeline()
