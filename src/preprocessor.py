import re
import string
from typing import List, Tuple, Dict, Any
import pandas as pd

# Load spaCy if available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    SPACY_AVAILABLE = True
except Exception:
    nlp = None
    SPACY_AVAILABLE = False

# Default English stopwords list for maximum offline resilience
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't", 'also', 'use', 'using', 'used', 'make', 'may', 'well', 'many',
    'new', 'one', 'two', 'via', 'e', 'g', 'al', 'et'
}

# Add NLTK stopwords if available
try:
    import nltk
    from nltk.corpus import stopwords
    nltk_stops = set(stopwords.words('english'))
    DEFAULT_STOPWORDS.update(nltk_stops)
except Exception:
    pass


class TextPreprocessor:
    """
    A robust text preprocessing pipeline tailored for Topic Modeling (LDA).
    Performs:
    1. Lowercasing & noise removal (regex, punctuation, numbers).
    2. Tokenization.
    3. Lemmatization using spaCy en_core_web_sm.
    4. POS filtering (optional, prioritizing NOUN, PROPN, ADJ, VERB).
    5. Stopword filtering.
    6. Bigram generation for domain-specific collocations.
    """

    def __init__(self, min_token_len: int = 3, allowed_pos: Tuple[str, ...] = ("NOUN", "ADJ", "VERB", "PROPN")):
        self.min_token_len = min_token_len
        self.allowed_pos = allowed_pos
        self.stopwords = set(DEFAULT_STOPWORDS)

    def clean_raw_text(self, text: str) -> str:
        """Removes URLs, HTML tags, numbers, and punctuation from raw text."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Remove numbers and digits
        text = re.sub(r'\d+', ' ', text)
        # Remove punctuation
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        # Collapse extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenizes and lemmatizes text using spaCy or fallback regex."""
        cleaned = self.clean_raw_text(text)
        
        if SPACY_AVAILABLE and nlp is not None:
            doc = nlp(cleaned)
            tokens = []
            for token in doc:
                lemma = token.lemma_.strip()
                pos = token.pos_
                # Filter tokens
                if (
                    len(lemma) >= self.min_token_len
                    and lemma not in self.stopwords
                    and (not self.allowed_pos or pos in self.allowed_pos)
                    and lemma.isalpha()
                ):
                    tokens.append(lemma)
            return tokens
        else:
            # Fallback simple tokenization & basic stemming/filtering
            raw_tokens = cleaned.split()
            tokens = [
                tok for tok in raw_tokens
                if len(tok) >= self.min_token_len
                and tok not in self.stopwords
                and tok.isalpha()
            ]
            return tokens

    def create_bigrams(self, tokenized_docs: List[List[str]], min_count: int = 2) -> List[List[str]]:
        """
        Creates bigrams for frequent adjacent tokens (e.g., 'neural', 'network' -> 'neural_network').
        Uses Gensim Phrases if available, else a count-based n-gram linker.
        """
        try:
            from gensim.models import Phrases
            from gensim.models.phrases import Phraser
            bigram_model = Phrases(tokenized_docs, min_count=min_count, threshold=1.0)
            bigram_phraser = Phraser(bigram_model)
            return [bigram_phraser[doc] for doc in tokenized_docs]
        except Exception:
            # Fallback bigram detector
            pair_counts: Dict[Tuple[str, str], int] = {}
            for doc in tokenized_docs:
                for i in range(len(doc) - 1):
                    pair = (doc[i], doc[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            frequent_pairs = {k for k, v in pair_counts.items() if v >= min_count}
            
            new_docs = []
            for doc in tokenized_docs:
                new_doc = []
                skip = False
                for i in range(len(doc)):
                    if skip:
                        skip = False
                        continue
                    if i < len(doc) - 1 and (doc[i], doc[i+1]) in frequent_pairs:
                        new_doc.append(f"{doc[i]}_{doc[i+1]}")
                        skip = True
                    else:
                        new_doc.append(doc[i])
                new_docs.append(new_doc)
            return new_docs

    def preprocess_corpus(self, texts: List[str], use_bigrams: bool = True) -> Tuple[List[List[str]], List[str]]:
        """
        Processes a collection of documents.
        Returns:
        - List of tokenized documents (list of strings for each document)
        - List of joined string representations (convenient for CountVectorizer)
        """
        tokenized_corpus = [self.tokenize_and_lemmatize(t) for t in texts]
        if use_bigrams:
            tokenized_corpus = self.create_bigrams(tokenized_corpus)
        
        joined_corpus = [" ".join(tokens) for tokens in tokenized_corpus]
        return tokenized_corpus, joined_corpus
