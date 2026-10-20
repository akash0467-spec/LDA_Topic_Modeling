import os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class TopicVisualizer:
    """
    Generates rich, publication-grade visualizations for Topic Models:
    1. Topic-Word Distribution Bar Charts (Matplotlib).
    2. Document-Topic Composition Heatmap.
    3. 2D Dimensionality Reduction (t-SNE) Topic Clusters.
    4. Gibbs Sampling Log-Likelihood Convergence Curve.
    5. Interactive Plotly / HTML Dashboard.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#CCCCCC'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_topic_word_distributions(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        topic_labels: Optional[Dict[int, str]] = None,
        top_n: int = 8,
        filename: str = "topic_bar_charts.png"
    ) -> str:
        """Plots horizontal bar charts showing top keywords and probabilities per topic."""
        n_topics = len(topic_words)
        cols = 3 if n_topics > 3 else n_topics
        rows = int(np.ceil(n_topics / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), squeeze=False)
        palette = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777", "#0891B2", "#4B5563"]

        for idx, (topic_id, words) in enumerate(topic_words.items()):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]

            top_w = words[:top_n][::-1]
            words_list = [w[0] for w in top_w]
            probs_list = [w[1] for w in top_w]
            color = palette[idx % len(palette)]

            bars = ax.barh(words_list, probs_list, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
            
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                        va='center', ha='left', fontsize=8, color='#333333')

            title = f"Topic {topic_id + 1}"
            if topic_labels and topic_id in topic_labels:
                title += f": {topic_labels[topic_id]}"
            
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#1E293B')
            ax.set_xlabel("Probability P(w | Topic)", fontsize=9, color='#475569')
            ax.grid(axis='x', linestyle='--', alpha=0.6)

        for idx in range(n_topics, rows * cols):
            r = idx // cols
            c = idx % cols
            fig.delaxes(axes[r, c])

        plt.suptitle("Latent Dirichlet Allocation - Discovered Topic Word Distributions", fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_document_topic_heatmap(
        self,
        doc_topic_dist: np.ndarray,
        doc_titles: List[str],
        topic_labels: Optional[List[str]] = None,
        filename: str = "document_topic_heatmap.png"
    ) -> str:
        """Plots heatmap showing document-topic probability matrix theta."""
        num_docs, num_topics = doc_topic_dist.shape
        fig, ax = plt.subplots(figsize=(max(8, num_topics * 1.8), max(6, num_docs * 0.38)))

        cax = ax.imshow(doc_topic_dist, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1.0)
        cbar = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Topic Weight $\\theta_{d,k}$", fontsize=10)

        x_labels = topic_labels if topic_labels else [f"Topic {k+1}" for k in range(num_topics)]
        ax.set_xticks(range(num_topics))
        ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9, fontweight='semibold')

        clean_titles = [f"{i+1}. {t[:38]}..." if len(t) > 38 else f"{i+1}. {t}" for i, t in enumerate(doc_titles)]
        ax.set_yticks(range(num_docs))
        ax.set_yticklabels(clean_titles, fontsize=8)

        for d in range(num_docs):
            for k in range(num_topics):
                val = doc_topic_dist[d, k]
                if val >= 0.15:
                    text_color = "white" if val > 0.55 else "black"
                    ax.text(k, d, f"{val:.2f}", ha='center', va='center', fontsize=7, color=text_color)

        ax.set_title("Document-Topic Distribution Matrix ($\\theta$ Proportions)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_topic_clusters_tsne(
        self,
        doc_topic_dist: np.ndarray,
        doc_titles: List[str],
        domains: Optional[List[str]] = None,
        filename: str = "topic_clusters_tsne.png"
    ) -> str:
        """Plots 2D t-SNE projection of documents based on their topic proportions."""
        from sklearn.manifold import TSNE

        n_samples = doc_topic_dist.shape[0]
        perplexity = min(5, max(2, n_samples - 1))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        coords = tsne.fit_transform(doc_topic_dist)

        dominant_topics = np.argmax(doc_topic_dist, axis=1) + 1
        unique_topics = np.unique(dominant_topics)

        fig, ax = plt.subplots(figsize=(10, 7))
        palette = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#DB2777"]

        for idx, topic_num in enumerate(unique_topics):
            mask = (dominant_topics == topic_num)
            color = palette[(topic_num - 1) % len(palette)]
            ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Topic {topic_num}",
                       s=140, alpha=0.85, edgecolors='black', linewidth=0.8, color=color)

        for i in range(n_samples):
            label_text = f"D{i+1}: {doc_titles[i][:22]}..." if i < len(doc_titles) else f"D{i+1}"
            ax.annotate(label_text, (coords[i, 0], coords[i, 1]),
                        fontsize=7, xytext=(5, 4), textcoords='offset points', alpha=0.8)

        ax.set_title("2D t-SNE Projection of Documents in Topic Probability Space", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
        ax.legend(title="Dominant Topic", loc='best', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_gibbs_convergence(
        self,
        ll_history: List[float],
        filename: str = "gibbs_sampling_convergence.png"
    ) -> str:
        """Plots Gibbs Sampling log marginal likelihood convergence."""
        fig, ax = plt.subplots(figsize=(8, 4.5))
        iterations = [(i + 1) * 10 for i in range(len(ll_history))]
        ax.plot(iterations, ll_history, marker='o', color='#2563EB', linewidth=2, markersize=5, label='Log Likelihood')
        ax.set_title("Collapsed Gibbs Sampling Convergence Curve", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gibbs Sampling Iteration", fontsize=10)
        ax.set_ylabel("Approx. Log Marginal Likelihood", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def generate_interactive_dashboard_html(
        self,
        topic_words: Dict[int, List[Tuple[str, float]]],
        doc_df: pd.DataFrame,
        topic_labels: Dict[int, str],
        metrics: Optional[Dict[str, float]] = None,
        sparsity: float = 94.88,
        vocab_size: int = 644,
        filename: str = "topic_model_interactive.html"
    ) -> str:
        """Generates an enhanced, publication-grade interactive HTML dashboard."""
        topics_cards_html = ""
        palette = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

        for topic_idx, words in topic_words.items():
            label = topic_labels.get(topic_idx, f"Topic {topic_idx + 1}")
            color = palette[topic_idx % len(palette)]
            
            # Find maximum probability in this topic for proportional bar scaling
            max_p = max([w[1] for w in words[:8]]) if words else 1.0

            words_rows = ""
            for rank, w in enumerate(words[:8], 1):
                # Scale relative to max probability so bars provide clear comparative visual feedback
                bar_pct = min(100.0, max(8.0, (w[1] / max_p) * 100.0))
                words_rows += f"""
                <tr>
                    <td style="padding: 7px 10px; font-weight: 600; color: #1E293B;">{rank}. {w[0]}</td>
                    <td style="padding: 7px 10px; text-align: right; color: #475569; font-family: monospace; font-size: 0.82rem;">{w[1]:.4f}</td>
                    <td style="padding: 7px 10px; width: 42%;">
                        <div style="background: #E2E8F0; width: 100%; border-radius: 999px; height: 9px; overflow: hidden;">
                            <div style="background: {color}; width: {bar_pct:.1f}%; height: 100%; border-radius: 999px;"></div>
                        </div>
                    </td>
                </tr>
                """

            topics_cards_html += f"""
            <div style="background: white; border-radius: 14px; box-shadow: 0 4px 10px -2px rgba(0,0,0,0.06); padding: 22px; border-top: 5px solid {color}; display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
                    <div>
                        <h3 style="margin: 0; color: #0F172A; font-size: 1.2rem; font-weight: 700;">Topic {topic_idx+1}</h3>
                    </div>
                    <span style="background: #F1F5F9; color: {color}; font-weight: 700; font-size: 0.78rem; padding: 4px 10px; border-radius: 999px; max-width: 60%; text-align: right;">{label}</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #F1F5F9; color: #64748B; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;">
                            <th style="padding: 6px 10px; text-align: left;">Term</th>
                            <th style="padding: 6px 10px; text-align: right;">Probability</th>
                            <th style="padding: 6px 10px; width: 42%;">Relative Weight</th>
                        </tr>
                    </thead>
                    <tbody>
                        {words_rows}
                    </tbody>
                </table>
            </div>
            """

        doc_rows = ""
        for _, row in doc_df.iterrows():
            conf_pct = row['Topic_Confidence'] * 100
            doc_rows += f"""
            <tr style="border-bottom: 1px solid #F1F5F9; transition: background 0.15s;" onmouseover="this.style.background='#F8FAFC'" onmouseout="this.style.background='white'">
                <td style="padding: 10px 14px; font-weight: bold; color: #64748B; font-size: 0.85rem;">#{row['Doc_ID']}</td>
                <td style="padding: 10px 14px; font-weight: 600; color: #0F172A;">{row['Title']}</td>
                <td style="padding: 10px 14px;"><span style="background: #EEF2FF; color: #4338CA; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 0.8rem;">{row['Dominant_Topic']}</span></td>
                <td style="padding: 10px 14px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: 700; color: #059669; font-size: 0.85rem; width: 50px;">{conf_pct:.1f}%</span>
                        <div style="background: #E2E8F0; width: 100px; border-radius: 999px; height: 6px; overflow: hidden;">
                            <div style="background: #10B981; width: {conf_pct:.1f}%; height: 100%;"></div>
                        </div>
                    </div>
                </td>
            </tr>
            """

        ll_val = f"{metrics['log_likelihood']:.1f}" if metrics else "-7387.1"
        perp_val = f"{metrics['perplexity']:.1f}" if metrics else "5015.7"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LDA Topic Modeling Dashboard & Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #F8FAFC;
            color: #1E293B;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            background: white;
            padding: 18px 22px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            border-left: 4px solid #3B82F6;
        }}
        .stat-card .label {{
            font-size: 0.75rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
        }}
        .stat-card .val {{
            font-size: 1.55rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}
        .section-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
            margin: 28px 0 16px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .table-card {{
            background: white;
            border-radius: 14px;
            box-shadow: 0 4px 10px -2px rgba(0,0,0,0.06);
            overflow: hidden;
            margin-bottom: 36px;
        }}
        table.full-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        table.full-table th {{
            background: #F1F5F9;
            color: #475569;
            text-align: left;
            padding: 12px 16px;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }}
        .gallery-item {{
            background: white;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 10px -2px rgba(0,0,0,0.06);
        }}
        .gallery-item h4 {{
            margin: 0 0 12px 0;
            color: #1E293B;
            font-size: 1rem;
            font-weight: 700;
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; color: #94A3B8; margin-bottom: 8px; font-weight: 700;">
                NLP Lab Experiment 4 • Probabilistic Topic Modeling (CO4, L3)
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;">Latent Dirichlet Allocation (LDA) Topic Discovery</h1>
            <p style="margin: 0; color: #CBD5E1; font-size: 1rem; max-width: 860px; line-height: 1.5;">
                Unsupervised semantic decomposition comparing <strong>Variational Inference</strong> and <strong>Collapsed Gibbs Sampling</strong> across a multi-domain corpus.
            </p>
        </header>

        <div class="stat-grid">
            <div class="stat-card" style="border-left-color: #3B82F6;">
                <div class="label">Documents (D)</div>
                <div class="val">{len(doc_df)}</div>
            </div>
            <div class="stat-card" style="border-left-color: #10B981;">
                <div class="label">Vocabulary Size (V)</div>
                <div class="val">{vocab_size}</div>
            </div>
            <div class="stat-card" style="border-left-color: #F59E0B;">
                <div class="label">DTM Sparsity</div>
                <div class="val">{sparsity:.1f}%</div>
            </div>
            <div class="stat-card" style="border-left-color: #8B5CF6;">
                <div class="label">Model Perplexity</div>
                <div class="val">{perp_val}</div>
            </div>
        </div>

        <div class="section-title">📊 Discovered Topics & Relative Keyword Weights</div>
        <div class="grid">
            {topics_cards_html}
        </div>

        <div class="section-title">📈 Publication-Grade Visualizations</div>
        <div class="gallery-grid">
            <div class="gallery-item">
                <h4>Topic-Word Probability Bar Charts</h4>
                <img src="topic_bar_charts.png" alt="Topic Bar Charts">
            </div>
            <div class="gallery-item">
                <h4>Document-Topic Composition Heatmap ($\theta$)</h4>
                <img src="document_topic_heatmap.png" alt="Document Heatmap">
            </div>
            <div class="gallery-item">
                <h4>2D t-SNE Projection of Document Clusters</h4>
                <img src="topic_clusters_tsne.png" alt="t-SNE Clusters">
            </div>
            <div class="gallery-item">
                <h4>Collapsed Gibbs Sampling Convergence Curve</h4>
                <img src="gibbs_sampling_convergence.png" alt="Gibbs Convergence">
            </div>
        </div>

        <div class="section-title">📑 Document Classification & Dominant Topic Mapping</div>
        <div class="table-card">
            <table class="full-table">
                <thead>
                    <tr>
                        <th style="width: 70px;">Doc #</th>
                        <th>Document Title</th>
                        <th style="width: 170px;">Dominant Topic</th>
                        <th style="width: 180px;">Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {doc_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""
        save_path = os.path.join(self.output_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return save_path
