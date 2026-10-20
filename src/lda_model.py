import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional


class LDATrainer:
    """
    Trains and evaluates Latent Dirichlet Allocation (LDA) models using
    Variational Inference (Scikit-learn LatentDirichletAllocation & Gensim LdaModel).
    
    Provides:
    - Model training with customizable hyperparameters (n_topics, alpha, eta/beta)
    - Perplexity and Log-Likelihood calculation
    - Coherence Score evaluation
    - Optimal K grid search
    - Topic-word extraction and document-topic distributions
    """

    def __init__(self, n_topics: int = 5, learning_method: str = "batch", max_iter: int = 50, random_state: int = 42):
        self.n_topics = n_topics
        self.learning_method = learning_method
        self.max_iter = max_iter
        self.random_state = random_state
        self.sklearn_model = None
        self.gensim_model = None
        self.vocab: List[str] = []
        self.doc_topic_dist: Optional[np.ndarray] = None

    def train_sklearn(self, dtm: np.ndarray, vocab: List[str], doc_topic_prior: Optional[float] = None, topic_word_prior: Optional[float] = None) -> Any:
        """
        Trains Scikit-learn LatentDirichletAllocation model.
        doc_topic_prior = alpha
        topic_word_prior = beta/eta
        """
        from sklearn.decomposition import LatentDirichletAllocation

        self.vocab = vocab
        self.sklearn_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
            learning_method=self.learning_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
            evaluate_every=5
        )
        self.doc_topic_dist = self.sklearn_model.fit_transform(dtm)
        return self.sklearn_model

    def train_gensim(self, corpus: Any, dictionary: Any, alpha: Any = "auto", eta: Any = "auto", passes: int = 15) -> Any:
        """Trains Gensim LdaModel if gensim is available."""
        try:
            from gensim.models import LdaModel
            self.gensim_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.n_topics,
                random_state=self.random_state,
                passes=passes,
                alpha=alpha,
                eta=eta,
                per_word_topics=True
            )
            return self.gensim_model
        except Exception as e:
            print(f"Gensim LdaModel training skipped: {e}")
            return None

    def evaluate_model(self, dtm: np.ndarray) -> Dict[str, float]:
        """Calculates Log-Likelihood and Perplexity for the trained scikit-learn model."""
        if self.sklearn_model is None:
            raise ValueError("Scikit-learn model not fitted.")
        ll = float(self.sklearn_model.score(dtm))
        perp = float(self.sklearn_model.perplexity(dtm))
        return {
            "log_likelihood": ll,
            "perplexity": perp
        }

    def get_top_topic_words(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extracts the top N keywords and their normalized probabilistic weights for each topic.
        """
        if self.sklearn_model is None:
            raise ValueError("Model has not been trained.")
        
        # topic_word distributions: shape (n_topics, n_features)
        components = self.sklearn_model.components_
        # Normalize to probability distribution
        topic_word_probs = components / components.sum(axis=1)[:, np.newaxis]
        
        topic_words = {}
        for topic_idx, topic in enumerate(topic_word_probs):
            top_indices = topic.argsort()[::-1][:top_n]
            topic_words[topic_idx] = [(self.vocab[i], float(topic[i])) for i in top_indices]
        
        return topic_words

    def get_topic_word_dataframe(self, top_n: int = 10) -> pd.DataFrame:
        """Generates a structured DataFrame with columns: Topic_ID, Rank, Word, Probability."""
        topic_words = self.get_top_topic_words(top_n=top_n)
        rows = []
        for topic_idx, words in topic_words.items():
            for rank, (word, prob) in enumerate(words, start=1):
                rows.append({
                    "Topic_ID": f"Topic {topic_idx + 1}",
                    "Rank": rank,
                    "Keyword": word,
                    "Probability": round(prob, 4)
                })
        return pd.DataFrame(rows)

    def get_document_topic_dataframe(self, doc_titles: List[str], raw_texts: List[str]) -> pd.DataFrame:
        """
        Produces document-topic distribution table showing:
        - Document Title
        - Topic proportions
        - Dominant Topic
        - Confidence Score
        """
        if self.doc_topic_dist is None:
            raise ValueError("Document-topic distribution unavailable.")
        
        num_docs = self.doc_topic_dist.shape[0]
        data = []
        for i in range(num_docs):
            dist = self.doc_topic_dist[i]
            dom_topic = int(np.argmax(dist)) + 1
            conf = float(np.max(dist))
            row = {
                "Doc_ID": i + 1,
                "Title": doc_titles[i] if i < len(doc_titles) else f"Doc {i+1}",
                "Dominant_Topic": f"Topic {dom_topic}",
                "Topic_Confidence": round(conf, 4),
            }
            for k in range(self.n_topics):
                row[f"Topic_{k+1}_Weight"] = round(float(dist[k]), 4)
            data.append(row)
        
        return pd.DataFrame(data)

    @staticmethod
    def evaluate_topic_coherence(texts: List[List[str]], dictionary: Any, corpus: Any, topic_words: List[List[str]]) -> Optional[float]:
        """Calculates Gensim CoherenceModel C_v score if gensim is available."""
        try:
            from gensim.models.coherencemodel import CoherenceModel
            cm = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            return float(cm.get_coherence())
        except Exception:
            return None
