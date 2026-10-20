import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class CollapsedGibbsLDA:
    """
    A pure NumPy implementation of Latent Dirichlet Allocation (LDA) using
    Collapsed Gibbs Sampling (Griffiths and Steyvers, 2004).

    Mathematical Foundation:
    - Document-topic distribution: theta_d ~ Dir(alpha)
    - Topic-word distribution: phi_k ~ Dir(beta)
    - Sampling update formula for word token i (word v in doc d):
      P(z_i = k | z_{-i}, w, alpha, beta) proportional to:
      [(n_{k,-i}^{(v)} + beta) / (n_{k,-i}^{(.)} + V * beta)] *
      [(n_{d,-i}^{(k)} + alpha) / (n_{d,-i}^{(.)} + K * alpha)]
    """

    def __init__(self, n_topics: int = 5, alpha: float = 0.5, beta: float = 0.1, n_iter: int = 150, random_state: int = 42):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.random_state = random_state

        self.vocab: List[str] = []
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.V: int = 0  # Vocabulary size
        self.D: int = 0  # Number of documents

        # Topic assignment per token in corpus: z[d][i]
        self.z: List[List[int]] = []
        # Documents as token indices: docs[d][i]
        self.docs_tokens: List[List[int]] = []

        # Count matrices:
        # n_dk[d, k]: number of tokens in doc d assigned to topic k
        self.n_dk: Optional[np.ndarray] = None
        # n_kv[k, v]: number of times word v is assigned to topic k
        self.n_kv: Optional[np.ndarray] = None
        # n_k[k]: total tokens assigned to topic k
        self.n_k: Optional[np.ndarray] = None

        # Estimated parameters
        self.phi: Optional[np.ndarray] = None      # Shape (n_topics, V): topic-word distributions
        self.theta: Optional[np.ndarray] = None    # Shape (D, n_topics): document-topic distributions
        self.log_likelihood_history: List[float] = []

    def fit(self, tokenized_docs: List[List[str]], vocab: Optional[List[str]] = None) -> "CollapsedGibbsLDA":
        np.random.seed(self.random_state)
        
        # Build vocabulary
        if vocab is None:
            unique_words = sorted(list({w for doc in tokenized_docs for w in doc}))
            self.vocab = unique_words
        else:
            self.vocab = list(vocab)

        self.V = len(self.vocab)
        self.D = len(tokenized_docs)
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for i, w in enumerate(self.vocab)}

        # Convert tokens to integers
        self.docs_tokens = []
        for doc in tokenized_docs:
            doc_idx = [self.word2id[w] for w in doc if w in self.word2id]
            self.docs_tokens.append(doc_idx)

        # Initialize count matrices
        K = self.n_topics
        V = self.V
        D = self.D
        self.n_dk = np.zeros((D, K), dtype=int)
        self.n_kv = np.zeros((K, V), dtype=int)
        self.n_k = np.zeros(K, dtype=int)
        self.z = []
        self.log_likelihood_history = []

        # Step 1: Random topic initialization
        for d in range(D):
            z_d = []
            for v in self.docs_tokens[d]:
                k = np.random.randint(0, K)
                z_d.append(k)
                self.n_dk[d, k] += 1
                self.n_kv[k, v] += 1
                self.n_k[k] += 1
            self.z.append(z_d)

        # Step 2: Collapsed Gibbs Sampling Iterations
        for iteration in range(self.n_iter):
            for d in range(D):
                doc_len = len(self.docs_tokens[d])
                for i in range(doc_len):
                    v = self.docs_tokens[d][i]
                    k = self.z[d][i]

                    # Exclude token i from counts (the "-i" term)
                    self.n_dk[d, k] -= 1
                    self.n_kv[k, v] -= 1
                    self.n_k[k] -= 1

                    # Compute conditional probability distribution over topics
                    # P(z_i = k | z_{-i}, w) \propto (n_kv + beta)/(n_k + V*beta) * (n_dk + alpha)
                    p_z = (self.n_kv[:, v] + self.beta) / (self.n_k + V * self.beta) * (self.n_dk[d, :] + self.alpha)
                    p_z = p_z / np.sum(p_z)  # Normalize probabilities

                    # Sample new topic assignment
                    new_k = np.random.choice(K, p=p_z)

                    # Update topic assignment and counts
                    self.z[d][i] = new_k
                    self.n_dk[d, new_k] += 1
                    self.n_kv[new_k, v] += 1
                    self.n_k[new_k] += 1

            # Monitor log-likelihood convergence every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration == self.n_iter - 1:
                ll = self._compute_log_likelihood()
                self.log_likelihood_history.append(ll)

        # Step 3: Compute Posterior Estimates of phi and theta
        self.phi = (self.n_kv + self.beta) / (self.n_k[:, np.newaxis] + V * self.beta)
        doc_lengths = np.array([len(doc) for doc in self.docs_tokens])
        self.theta = (self.n_dk + self.alpha) / (doc_lengths[:, np.newaxis] + K * self.alpha)

        return self

    def _compute_log_likelihood(self) -> float:
        """Approximates total log marginal likelihood log P(W | Z)."""
        V = self.V
        K = self.n_topics
        ll = 0.0
        # Word-topic distribution likelihood
        for k in range(K):
            ll += np.sum(np.log(np.arange(1, self.n_kv[k, :].sum() + 1) + self.beta))
        return float(ll)

    def get_top_topic_words(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Returns top_n words and their posterior probability for each topic."""
        if self.phi is None:
            raise ValueError("Model has not been fitted yet.")
        topic_words = {}
        for k in range(self.n_topics):
            top_word_indices = np.argsort(self.phi[k, :])[::-1][:top_n]
            topic_words[k] = [(self.id2word[idx], float(self.phi[k, idx])) for idx in top_word_indices]
        return topic_words

    def get_document_topics_dataframe(self, doc_titles: Optional[List[str]] = None) -> pd.DataFrame:
        """Returns a formatted DataFrame containing document-topic distribution theta."""
        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")
        cols = [f"Topic_{k+1}" for k in range(self.n_topics)]
        df = pd.DataFrame(self.theta, columns=cols)
        if doc_titles is not None and len(doc_titles) == self.D:
            df.insert(0, "Document_Title", doc_titles)
        df["Dominant_Topic"] = np.argmax(self.theta, axis=1) + 1
        df["Max_Probability"] = np.max(self.theta, axis=1)
        return df
