from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class DocumentTermRepresentation:
    """
    Constructs Document-Term representations for topic modeling.
    Supports:
    - Scikit-learn CountVectorizer Document-Term Matrix (sparse/dense array)
    - Gensim Dictionary & doc2bow corpus representation
    - Matrix sparsity and vocabulary statistics
    """

    def __init__(self, min_df: int = 1, max_df: float = 0.95):
        self.min_df = min_df
        self.max_df = max_df
        self.vocab: List[str] = []
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.dtm: Optional[np.ndarray] = None
        self.gensim_dictionary = None
        self.gensim_corpus = None

    def fit_transform_sklearn(self, joined_docs: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Fits CountVectorizer and returns dense Document-Term Matrix and feature names."""
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r'\b[a-zA-Z_]{3,}\b'
        )
        dtm_sparse = vectorizer.fit_transform(joined_docs)
        self.dtm = dtm_sparse.toarray()
        self.vocab = vectorizer.get_feature_names_out().tolist()
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for i, w in enumerate(self.vocab)}
        return self.dtm, self.vocab

    def fit_transform_gensim(self, tokenized_docs: List[List[str]]) -> Tuple[Any, List[List[Tuple[int, int]]]]:
        """Constructs Gensim Dictionary and Bag-of-Words (BoW) corpus."""
        try:
            from gensim.corpora import Dictionary
            self.gensim_dictionary = Dictionary(tokenized_docs)
            # Filter extreme frequencies
            self.gensim_dictionary.filter_extremes(no_below=self.min_df, no_above=self.max_df)
            self.gensim_corpus = [self.gensim_dictionary.doc2bow(doc) for doc in tokenized_docs]
            return self.gensim_dictionary, self.gensim_corpus
        except ImportError:
            # Fallback custom gensim-compatible dictionary structure
            unique_words = sorted(list({tok for doc in tokenized_docs for tok in doc}))
            self.word2id = {w: i for i, w in enumerate(unique_words)}
            self.id2word = {i: w for i, w in enumerate(unique_words)}
            
            bow_corpus = []
            for doc in tokenized_docs:
                counts: Dict[int, int] = {}
                for tok in doc:
                    if tok in self.word2id:
                        idx = self.word2id[tok]
                        counts[idx] = counts.get(idx, 0) + 1
                bow_corpus.append(sorted(list(counts.items())))
            self.gensim_corpus = bow_corpus
            return self.word2id, bow_corpus

    def get_sparsity(self) -> float:
        """Calculates matrix sparsity percentage: 100 * (1 - non_zeros / total_cells)"""
        if self.dtm is None:
            return 0.0
        total_elements = self.dtm.shape[0] * self.dtm.shape[1]
        non_zero_elements = np.count_nonzero(self.dtm)
        sparsity = (1.0 - (non_zero_elements / total_elements)) * 100.0
        return sparsity

    def get_top_frequent_words(self, top_n: int = 15) -> pd.DataFrame:
        """Returns the most frequent words across the document collection."""
        if self.dtm is None:
            return pd.DataFrame()
        total_counts = self.dtm.sum(axis=0)
        word_freq = [(self.vocab[i], total_counts[i]) for i in range(len(self.vocab))]
        word_freq.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(word_freq[:top_n], columns=["Word", "Frequency"])
