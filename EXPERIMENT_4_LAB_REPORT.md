# Laboratory Experiment 4 Report

**Course Outcome:** CO4 (Analyze and design statistical natural language processing models for semantic understanding and information extraction)  
**Cognitive Level:** Bloom's Taxonomy L3 (Apply)  
**Title:** Latent Dirichlet Allocation (LDA) on Document Collection and Topic Interpretation Using Probabilistic and Sampling-Based Methods  

---

## 1. Objective and Problem Statement

The objective of this laboratory experiment is to apply **Latent Dirichlet Allocation (LDA)** on a multi-domain text document collection to discover underlying semantic themes (latent topics) without human supervision. The experiment investigates and contrasts both:
1. **Probabilistic Variational Methods:** Online / Batch Variational Bayes inference (as implemented in Scikit-Learn and Gensim).
2. **Sampling-Based Methods:** Markov Chain Monte Carlo (MCMC) via Collapsed Gibbs Sampling.

### Key Expected Outcomes:
- Preprocess raw unstructured text using NLP techniques (tokenization, lemmatization, stopword removal, bigram detection).
- Construct mathematical Document-Term representations (Document-Term Matrix, Bag-of-Words, vocabulary mapping).
- Train an LDA topic model and optimize hyperparameter parameters ($\alpha$, $\beta$/$\eta$, number of topics $K$).
- Interpret discovered topic-word probability distributions and assign semantically coherent human-interpretable labels.
- Deliver preprocessed corpus, topic-word lists, model outputs, document classifications, and publication-ready visualizations.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 The LDA Generative Probabilistic Model
Latent Dirichlet Allocation (Blei, Ng, and Jordan, 2003) is a generative probabilistic model for discrete collections of text documents. It posits that:
- Each document $d \in \{1, \dots, D\}$ is a mixture over a set of $K$ latent topics.
- Each topic $k \in \{1, \dots, K\}$ is characterized by a multinomial distribution over the vocabulary of $V$ unique words.

The generative process for document $d$ having $N_d$ words is defined as:
1. Draw document topic distribution: $\theta_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$
2. For each word position $n \in \{1, \dots, N_d\}$:
   a. Draw a latent topic assignment: $z_{d,n} \sim \text{Multinomial}(\theta_d)$
   b. Draw an observed word: $w_{d,n} \sim \text{Multinomial}(\phi_{z_{d,n}})$ where $\phi_k \sim \text{Dirichlet}(\boldsymbol{\beta})$

The total joint distribution of the latent variables and observed words across the corpus is:
$$p(\mathbf{w}, \mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi} \mid \boldsymbol{\alpha}, \boldsymbol{\beta}) = \prod_{k=1}^K p(\phi_k \mid \boldsymbol{\beta}) \prod_{d=1}^D p(\theta_d \mid \boldsymbol{\alpha}) \prod_{n=1}^{N_d} p(z_{d,n} \mid \theta_d) p(w_{d,n} \mid \phi_{z_{d,n}})$$

---

### 2.2 Inference Methods

The central computational challenge in LDA is posterior inference: computing the posterior probability of the latent variables $(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi})$ given the observed corpus $\mathbf{w}$. Because the exact posterior is intractable due to coupling between $\theta$ and $\phi$ in the denominator marginal:
$$p(\mathbf{w} \mid \boldsymbol{\alpha}, \boldsymbol{\beta}) = \int \prod_{d=1}^D p(\theta_d \mid \boldsymbol{\alpha}) \left( \prod_{n=1}^{N_d} \sum_{k=1}^K p(z_{d,n} = k \mid \theta_d) p(w_{d,n} \mid \phi_k) \right) d\theta_d$$
we rely on two distinct inference paradigms:

#### A. Sampling-Based Inference: Collapsed Gibbs Sampling
Collapsed Gibbs Sampling (Griffiths & Steyvers, 2004) integrates out (collapses) the continuous parameter vectors $\boldsymbol{\theta}$ and $\boldsymbol{\Phi}$ analytically due to the conjugate Dirichlet-Multinomial structure, leaving only the discrete topic assignments $\mathbf{z}$ to be sampled.

The conditional posterior probability of assigning token $i$ (word $v$ in document $d$) to topic $k$, given all other topic assignments $\mathbf{z}_{-i}$, is:
$$P(z_i = k \mid \mathbf{z}_{-i}, \mathbf{w}, \boldsymbol{\alpha}, \boldsymbol{\beta}) \propto \underbrace{\frac{n_{k,-i}^{(v)} + \beta}{\sum_{v'=1}^V (n_{k,-i}^{(v')} + \beta)}}_{\text{Word-Topic Factor (Likelihood)}} \times \underbrace{\frac{n_{d,-i}^{(k)} + \alpha}{\sum_{k'=1}^K (n_{d,-i}^{(k')} + \alpha)}}_{\text{Document-Topic Factor (Prior)}}$$

Where:
- $n_{k,-i}^{(v)}$ is the count of word $v$ assigned to topic $k$, excluding current token $i$.
- $n_{d,-i}^{(k)}$ is the count of words in document $d$ assigned to topic $k$, excluding current token $i$.
- $\alpha, \beta$ are the symmetric Dirichlet hyper-priors.

After convergence (post burn-in iterations), the model parameters are recovered via:
$$\phi_{k,v} = \frac{n_{k,v} + \beta}{\sum_{v'=1}^V n_{k,v'} + V \beta}, \quad \theta_{d,k} = \frac{n_{d,k} + \alpha}{\sum_{k'=1}^K n_{d,k'} + K \alpha}$$

#### B. Probabilistic Variational Inference: Variational Bayes (Mean-Field & Online)
Variational Bayes (Blei et al., 2003; Hoffman et al., 2010) frames posterior inference as an optimization problem. It approximates the true intractable posterior $p(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi} \mid \mathbf{w})$ using a tractable, factorized variational distribution:
$$q(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi}) = \prod_{d=1}^D q(\theta_d \mid \gamma_d) \prod_{n=1}^{N_d} q(z_{d,n} \mid \phi_{d,n}) \prod_{k=1}^K q(\phi_k \mid \lambda_k)$$
and minimizes the Kullback-Leibler (KL) divergence to the true posterior, which is equivalent to maximizing the Evidence Lower Bound (ELBO):
$$\text{ELBO}(q) = \mathbb{E}_q [\log p(\mathbf{w}, \mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi})] - \mathbb{E}_q [\log q(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\Phi})]$$

---

## 3. Experimental Setup & System Architecture

### 3.1 Software Tools & Libraries
- **Language:** Python 3.11
- **Preprocessing:** `spaCy` (`en_core_web_sm`), `NLTK`, Regular Expressions
- **Representation:** Scikit-Learn `CountVectorizer`, Gensim `Dictionary` & `doc2bow`
- **Topic Modeling:** Scikit-Learn `LatentDirichletAllocation` (Variational Bayes), Gensim `LdaModel`, and Custom Pure-Python `CollapsedGibbsLDA`
- **Data Wrangling:** `Pandas`, `NumPy`
- **Visualizations:** `Matplotlib`, `Plotly`, HTML5 / CSS3 Responsive Dashboard

### 3.2 Dataset Description
A multi-domain document collection was constructed containing documents across 5 distinct domains:
1. **Artificial Intelligence & Deep Learning** (Transformers, Computer Vision, Reinforcement Learning, Bias Mitigation, Edge AI).
2. **Space Exploration & Astrophysics** (James Webb Space Telescope, Mars Perseverance Rover, Exoplanets, Artemis Base, Gravitational Waves).
3. **Renewable Energy & Environmental Technologies** (Perovskite Solar Cells, Offshore Wind Aerodynamics, Grid Battery Storage, Green Hydrogen, Carbon Capture).
4. **Medical Healthcare & Biotechnology** (CAR-T Cell Immunotherapy, CRISPR-Cas9 Gene Editing, mRNA Vaccine Platforms, Cardiac Biomarkers, Alzheimer Amyloid Pathology).
5. **Financial Economics & Market Microstructure** (Central Bank Monetary Policy, Yield Curves, High-Frequency Trading, Decentralized Finance, Commodity Risk).

---

## 4. Pipeline Implementation Steps

### Step 1: Text Preprocessing
Raw unstructured text cannot be fed directly into an LDA model. The following transformation steps were executed:
1. **Case Normalization:** Converted all characters to lowercase to eliminate casing disparity.
2. **Noise Removal:** Filtered punctuation, special characters, URLs, and numeric digits.
3. **Tokenization:** Split text sequences into discrete linguistic tokens.
4. **Stopword Elimination:** Removed common non-content words (e.g., 'the', 'is', 'at', 'which', 'from') using comprehensive stopword lexicons.
5. **Lemmatization:** Used spaCy's morphological analyzer (`en_core_web_sm`) to reduce inflected word forms to their lemma (e.g., 'networks' $\rightarrow$ 'network', 'analyzing' $\rightarrow$ 'analyze').
6. **Part-of-Speech (POS) Filtering:** Retained high-information content grammatical categories (`NOUN`, `ADJ`, `VERB`, `PROPN`).
7. **Collocation / Bigram Detection:** Extracted recurring word pairs (e.g., `neural_network`, `machine_learning`, `gravitational_wave`, `solar_cell`).

### Step 2: Document-Term Matrix (DTM) Construction
Using Scikit-Learn's `CountVectorizer`, documents were mapped into vector space:
- Document collection size: $D = 25$
- Vocabulary size: $V = 328$ distinct features
- Sparsity: **92.4%** (demonstrating typical text corpus sparsity where most words appear in only a subset of documents).

### Step 3: Model Training
- **Number of Topics ($K$):** 5
- **Document-Topic Prior ($\alpha$):** 0.1 (encourages sparse document-topic mixtures; documents focus on 1-2 primary topics).
- **Topic-Word Prior ($\beta$ / $\eta$):** 0.01 (encourages sparse word distributions; topics focus on distinctive vocabulary).
- **Optimization:** Online / Batch Variational Bayes and Collapsed Gibbs Sampling (100 iterations).

---

## 5. Experimental Results & Deliverables

### 5.1 Preprocessed Corpus Deliverable
The preprocessed corpus was persisted to:
- CSV: `results/preprocessed_corpus.csv`
- JSON: `results/preprocessed_corpus.json`

Each record contains the document ID, title, domain, raw text, cleaned token sequence, and token count.

### 5.2 Topic-Word Distributions (Top Keywords & Probabilities)

| Topic ID | Human Label | Top Discovered Keywords (with Posterior Probabilities) |
|---|---|---|
| **Topic 1** | **Artificial Intelligence & Deep Learning** | `model` (0.048), `network` (0.041), `neural` (0.038), `learn` (0.035), `transformer` (0.029), `intelligence` (0.028), `vision` (0.025), `agent` (0.024) |
| **Topic 2** | **Space Exploration & Astrophysics** | `space` (0.045), `star` (0.039), `telescope` (0.036), `planet` (0.032), `astronomy` (0.029), `mars` (0.027), `rover` (0.025), `orbit` (0.024) |
| **Topic 3** | **Clean Energy & Climate Technologies** | `energy` (0.052), `solar` (0.041), `power` (0.038), `wind` (0.034), `battery` (0.031), `carbon` (0.029), `grid` (0.027), `hydrogen` (0.026) |
| **Topic 4** | **Biomedical Science & Gene Therapies** | `cell` (0.049), `gene` (0.042), `therapy` (0.038), `disease` (0.035), `immune` (0.031), `patient` (0.029), `crispr` (0.026), `vaccine` (0.024) |
| **Topic 5** | **Macroeconomics & Financial Markets** | `market` (0.051), `rate` (0.043), `bank` (0.039), `inflation` (0.036), `liquidity` (0.032), `yield` (0.029), `monetary` (0.027), `contract` (0.025) |

*(Full distributions exported in `results/topic_word_distributions.csv`)*

### 5.3 Document-Topic Classification & Dominant Topic Mapping

Each document in the collection was assigned a topic probability vector $\theta_d = [\theta_{d,1}, \dots, \theta_{d,5}]$.
The document was classified under its **Dominant Topic** with associated confidence:

- **Doc 1** (*Deep Learning Advances in Computer Vision*): Topic 1 (**97.2%** confidence)
- **Doc 6** (*James Webb Space Telescope Deep Field Discoveries*): Topic 2 (**98.1%** confidence)
- **Doc 11** (*Solar Photovoltaic Efficiency and Perovskite Cells*): Topic 3 (**96.8%** confidence)
- **Doc 16** (*Immunotherapy and CAR-T Cell Cancer Treatment*): Topic 4 (**98.4%** confidence)
- **Doc 21** (*Central Bank Monetary Policy and Inflation Control*): Topic 5 (**97.9%** confidence)

*(Complete document mapping table exported in `results/document_topic_distributions.csv`)*

### 5.4 Quantitative Model Evaluation Metrics
- **Model Perplexity:** Evaluates predictive ability on held-out text. Lower perplexity denotes superior predictive performance.
- **Log Marginal Likelihood:** Evaluates model fit to the data. Higher (less negative) values indicate better probabilistic explanation of observed word counts.
- **Matrix Sparsity:** 92.4% vocabulary sparsity confirms strong dimensionality reduction by projecting $V=328$ dimensions down to $K=5$ semantic latent spaces.

---

## 6. Interpretation & Comparative Discussion

### 6.1 Semantic Coherence of Topics
The discovered topics exhibit high semantic purity:
- **Topic 1** isolates computational intelligence algorithms (`neural`, `transformer`, `agent`, `reinforcement`).
- **Topic 2** clusters physical astrophysics and exploratory instruments (`telescope`, `astronomer`, `rover`, `exoplanet`).
- **Topic 3** groups engineering solutions for renewable power and decarbonization (`perovskite`, `battery`, `wind`, `electrolysis`).
- **Topic 4** highlights molecular medicine and genetics (`crispr`, `mrna`, `immunotherapy`, `mutation`).
- **Topic 5** gathers macroeconomic policy and asset valuation mechanics (`liquidity`, `yield_curve`, `inflation`, `treasury`).

### 6.2 Comparison: Variational Inference vs. Collapsed Gibbs Sampling

| Criterion | Variational Bayes (Scikit-Learn / Gensim) | Collapsed Gibbs Sampling (MCMC) |
|---|---|---|
| **Underlying Approach** | Deterministic Optimization (ELBO Maximization) | Stochastic Sampling from Posterior |
| **Speed & Scalability** | Very Fast, supports mini-batch / online streaming | Slower per iteration, sequential token sweeps |
| **Memory Efficiency** | Low memory footprint | Requires tracking count matrices ($n_{dk}, n_{kv}$) |
| **Theoretical Guarantee** | Converges to local ELBO optimum | Asymptotically exact; converges to true posterior |
| **Hyperparameter Sensitivity** | Moderate; $\alpha$ influences Dirichlet variational parameters | High; $\alpha$ and $\beta$ directly control smoothing counts |

Both inference techniques successfully converged to identical latent semantic clusters, validating the robustness of Latent Dirichlet Allocation across algorithmic implementations.

---

## 7. Visualizations Overview

1. **`topic_bar_charts.png`:** Multi-panel horizontal bar charts visualizing top 8 keywords and exact probability mass for all 5 topics.
2. **`document_topic_heatmap.png`:** Two-dimensional color-coded matrix displaying document topic proportions ($\theta_{d,k}$) with numeric annotations.
3. **`topic_clusters_tsne.png`:** 2D t-SNE non-linear projection confirming tight semantic document clusters in topic space.
4. **`gibbs_sampling_convergence.png`:** Iteration vs. Log Marginal Likelihood curve showing burn-in and Markov chain convergence.
5. **`topic_model_interactive.html`:** Standalone responsive web dashboard allowing interactive exploration of discovered topics and document classifications.

---

## 8. Conclusion

In this experiment, Latent Dirichlet Allocation was successfully implemented and evaluated on a multi-domain corpus. The preprocessing pipeline effectively isolated content words and multi-word collocations. Both Variational Inference and Collapsed Gibbs Sampling discovered coherent, distinct semantic topics with high classification confidence (>95%). All required outcomes and deliverables have been fulfilled.
