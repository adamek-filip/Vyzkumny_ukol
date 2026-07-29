# Predicting Conspiracy Beliefs from Unstructured Text: A Psychometric Machine Learning Pipeline

## 1. Project Objective and Scope
The fundamental goal of this computational research project is to construct predictive models mapping unstructured psychological text and baseline sociodemographic variables into latent attitudinal structures. Specifically, this pipeline predicts respondents' attitudes toward four distinct conspiracy theories:

*   **Shadow Elites (`w2_conspir_a`):** The belief that the world is run by hidden elites and politicians are merely puppets.
*   **Organized Migration (`w2_conspir_b`):** The belief that illegal migration is an organized plot to replace Europeans.
*   **COVID-19 Vaccine Coverup (`w2_conspir_c`):** The belief that severe harmful consequences of COVID-19 vaccination are deliberately suppressed.
*   **NATO War Provocation (`w2_conspir_d`):** The belief that Russia was pushed into the Ukraine war by hostile NATO behavior.

The target variables ($Y$) are measured on a standard 5-point Likert scale. We extract three distinct sets of variables from the CAB panel to construct and compare our models: the baseline sociodemographic and psychological features ($X_{survey}$) and the unstructured text responses ($X_{text}$).

---

## 2. Computational Methodology

To rigorously isolate the predictive capacity of linguistic artifacts against traditional survey metrics, the architecture utilizes three distinct mapping methods.

### Method 0: Baseline Mapping via Traditional Survey Variables
Before analyzing semantically independent texts, we establish a baseline benchmark. We utilize regularized regression to map standard survey features (e.g., resilience, institutional trust, political ideology, threat perception) to the target conspiracy attitudes. 
*   **Why this approach?** This step isolates the variance explained purely by standard psychometric surveying, establishing a strict lower bound for predictive performance that our text-based models must surpass to demonstrate utility.

### Method 1: Explicit Text Mapping via Mathematization
This approach utilizes a Large Language Model (LLM) strictly as an unsupervised feature extractor, projecting the unstructured text $X_{text}$ into 55 predefined, human-interpretable linguistic dimensions. We apply a continuous regularized regression model paired with ordinal thresholding.

This architecture treats the ordinal target as a continuous variable $y \in \mathbb{R}$, training a standard linear regression that minimizes the Mean Squared Error (MSE):

$$ \mathcal{J}(\beta) = \frac{1}{2N} \sum_{i=1}^N (y_i - \mathbf{x}_i^T \beta)^2 + \alpha \left( \rho \|\beta\|_1 + \frac{1-\rho}{2} \|\beta\|_2^2 \right) $$

During the inner cross-validation loop, the optimal hyperparameters ($\alpha$ and $\rho$) are selected using Negative Mean Squared Error (NMSE).
*   **Dynamic Quantile Thresholding:** To discretize the continuous predictions back into ordinal classes, we implement Dynamic Quantile Thresholding via Empirical Cumulative Distribution Function matching. 

### Method 2: Implicit Mapping via In-Context Learning (ICL)
Instead of forcing text into predefined dimensions, this method utilizes the LLM as an end-to-end classifier by leveraging its native attention mechanism to compute similarities directly in its high-dimensional latent space. The model is provided with a system prompt detailing the psychometric context and 15 stratified, labeled historical examples.

---

## 3. Repository Structure

The codebase is modularized to separate data hygiene, explicit modeling, and implicit LLM inference.

### Data Preparation Pipeline (Method 0 & 1)
*   `demographic_data_prep_initial.py`: Handles the primary loading and alignment of the CAB panel waves, establishing the master ID reference and left-joining data to prevent accidental cohort dropping.
*   `demographic_ordinal_2.py`: Unifies ordinal scales and constructs the global design matrix, employing aggressive global standardization and specific categorical mapping dictionaries.
*   `demographic_data_prep.py`: Filters the tabular dataset to strictly match the respondents present in the NLP dataset, ensuring exact cohort alignment for valid cross-validation comparisons.
*   `JSON_creation.py`: Processes the target conspiracy CSV files into a single, cleaned JSON architecture optimized for stratified cross-validation.

### Explicit Modeling Pipeline (Method 0 & 1)
*   `elastic_net_demographic.py` & `FINAL_elastic_net_2.py`: Executes the 10x10 Nested Stratified Cross-Validation for the continuous Elastic Net model. These scripts handle internal imputation and scaling to prevent data leakage and implement the Dynamic Quantile Thresholding logic.
*   `ordinal_reg_demographic.py` & `FINAL_ordinal_reg_4.py`: Implements the custom `ElasticNetOrdinalRegression` estimator utilizing the L-BFGS-B optimization algorithm to directly model the ordinal nature of the targets.

### Implicit Modeling Pipeline (Method 2)
*    `shadow_elites.py`, `migration.py`,  `covid.py`,  `russia_nato.py`: The end-to-end In-Context Learning API pipeline. It dynamically generates stratified master contexts for Key-Value caching, executes decoupled API calls to a local LLM, and forces a strict JSON schema output. It mathematically encodes the classification decision as a probability vector distribution for downstream divergence evaluation.
*   `visualization.py`: Contains the layout engines for rendering global classification metrics (Accuracy, QWK) and generating structural confusion matrix heatmaps.

---

## 4. Execution Requirements
*   **Scientific Computing:** The pipelines require standard Python data science libraries (`pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`).
*   **Local LLM Infrastructure:** Method 2 requires a local instance of Ollama running `gemma4:31b` to execute the prompts without transmitting sensitive psychological text to external endpoints. Ensure the API is listening on `http://localhost:11434/api/generate` before initiating `migration.py`.
