import json
import os
import re
import requests
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.special import softmax

# CONFIGURATION
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:31b"

def load_data(filepath: str, json_dim_key: str, target_dim: str) -> tuple[pd.DataFrame, str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = data[json_dim_key]["data"]
    theory_desc = data[json_dim_key].get("theory_description", "Undefined theory.")
    
    df = pd.DataFrame(records)
    
    # Dynamically rename the generic JSON 'score' to your specific dimension name
    if 'score' in df.columns:
        df = df.rename(columns={'score': target_dim})
    else:
        raise KeyError(f"Expected 'score' key in JSON records, but it was not found.")
        
    df = df[['id', 'text', target_dim]] 
    return df, theory_desc

def format_prompt(training_data: pd.DataFrame, test_text: str, target_dim: str, max_examples: int = 15) -> str:
    """Node 2.2: Construct the prompt bypassing the DataFrame index with explicit context."""
    theory_descriptions = {
        "hidden_elites": "Svět ve skutečnosti řídí skryté elity, které jsou mocnější než jednotlivé vlády, politici jsou jenom jejich loutkami. (The world is actually run by hidden elites who are more powerful than individual governments, politicians are just their puppets).",
        "migration": "Illegal migration is an organized plot to replace Europeans.",
        "vaccines": "COVID-19 vaccine harms are being suppressed.",
        "russia_nato": "Russia was forced into the Ukraine war by NATO's behavior."
    }
    
    theory_description = theory_descriptions.get(target_dim, "Undefined target theory.")
    
    sys_instruction = f"""You are an expert political psychologist and psychometrician. Your task is to analyze projective text responses and predict the author's latent attitude toward a specific conspiracy theory.

CONTEXT (THE STIMULUS):
The respondents were asked the following projective question in Czech: 

"Imagine waking up as the Prime Minister of the Czech Republic. What would you change in the Czech Republic? Please write at least five sentences that describe your ideas or actions. This will help us find out what people think is really important. Please write a coherent text of at least five sentences." (Czech: "Představte si, že jste se probudil jako premiér ČR. Co byste změnil? Napište alespoň 5 vět.")

TARGET VARIABLE:
Predict the respondent's attitude toward the following statement:
"{theory_description}"

MEASUREMENT SCALE:
1 = Strongly Agree
2 = Agree
3 = Neutral
4 = Disagree
5 = Strongly Disagree

INSTRUCTIONS:
Below are historical examples of respondent texts and their true latent scores on this exact scale. 
Analyze the semantic, linguistic, and rhetorical patterns in these examples. Use this context to score the final target text. 
Output ONLY a single integer (1, 2, 3, 4, or 5). Do not provide explanations, formatting, or additional text."""
    
    examples = "HISTORICAL EXAMPLES:\n"
    
    # Convert relational matrix into native dictionaries to sever Index dependency
    for row in training_data.to_dict(orient='records'):
        examples += f"Text: {row['text']}\nScore: {row[target_dim]}\n\n"
        
    final_query = f"TARGET TEXT TO SCORE:\nText: {test_text}\nScore: [["
    return f"{sys_instruction}\n\n{examples}{final_query}"

def query_ollama(prompt: str) -> tuple[int, np.ndarray, np.ndarray, str]:
    """
    Node 2.3 & 2.4: Execute decoupled API call via /api/generate.
    Returns: (predicted_score, probability_vector, raw_logs, raw_text)
    """
    schema = {
        "type": "object",
        "properties": {
            "prediction": {
                "type": "integer",
                "enum": [1, 2, 3, 4, 5]
            }
        },
        "required": ["prediction"]
    }

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,        
        "top_logprobs": 10,
        "format": schema,     # Enforce the strict schema
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_gpu": 99,
            "num_predict": 10,   # Give it slightly more room for the JSON wrapper
            "stop": ["]]", "}", "```"] # Aggressive stop tokens
        }
    }
    
    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=900)
    response.raise_for_status()
    
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Ollama Error: {data['error']}")
        
    raw_text = data.get("response", "")
    
    # Parse the forced JSON schema
    try:
        response_dict = json.loads(raw_text)
        predicted_score = int(response_dict.get("prediction"))
    except (json.JSONDecodeError, ValueError, TypeError):
        # Aggressive Fallback if the model breaks schema but outputs a digit
        match = re.findall(r'([1-5])', raw_text)
        if not match:
            raise ValueError(f"Model failed to generate a valid target score.\nOutput trace: {raw_text[:120]}")
        predicted_score = int(match[-1])
        
    
    # Mathematical Representation for Downstream KL Divergence Evaluation
    # Encodes the hard classification decision as a valid probability vector distribution
    probabilities = np.zeros(5)
    probabilities[predicted_score - 1] = 1.0

    logprobs_array = data.get("logprobs")
    
    if logprobs_array:
        target_distribution = None
        
        # Scan backwards through the generated tokens to find the exact step 
        # where the model emitted our regex-matched digit
        for step in reversed(logprobs_array):
            token_str = step.get("token", "").strip(" \n\r\t*[]")
            if token_str == str(predicted_score):
                target_distribution = step.get("top_logprobs", [])
                break
                
        if target_distribution:
            # Isolate the log-probabilities for the 1-5 scale
            logprobs_dict = {1: -np.inf, 2: -np.inf, 3: -np.inf, 4: -np.inf, 5: -np.inf}
            
            for candidate in target_distribution:
                cand_token = candidate.get("token", "").strip(" \n\r\t*[]")
                cand_logprob = candidate.get("logprob", -np.inf)
                
                if cand_token.isdigit() and int(cand_token) in logprobs_dict:
                    logprobs_dict[int(cand_token)] = cand_logprob
                    
            raw_logs = np.array([logprobs_dict[i] for i in range(1, 6)])
            
            # Convert the logarithmic space back to a clean 0.0 - 1.0 distribution
            if not np.all(raw_logs == -np.inf):
                probabilities = softmax(raw_logs)

    return predicted_score, probabilities, raw_logs, raw_text

def process_dimension(df: pd.DataFrame, target_dim: str, n_splits=10, permutations=3, max_examples=15) -> dict:
    """Executes 10-fold Stratified CV, permutation averaging, and performance analysis."""
    X = df['text'].tolist()
    y = df[target_dim].to_numpy(dtype=int)
    
    total_records = len(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_ids = []
    all_predictions = []
    all_expected_values = []
    all_true_labels = []
    all_prob_distributions = []
    all_log_distributions = []
    all_raw_1 = []
    all_raw_2 = []
    all_raw_3 = []
    
    processed_count = 0
    live_backup_csv = f"live_checkpoint_{target_dim}.csv"
    if not os.path.exists(live_backup_csv):
        columns = ['id', 'true_label', 'pred_argmax', 'pred_expected', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 'prob_5', 'raw_1', 'raw_2', 'raw_3']
        pd.DataFrame(columns=columns).to_csv(live_backup_csv, index=False)
        print(f"[SYSTEM] Created live backup file: {live_backup_csv}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nProcessing Fold {fold+1}/{n_splits}...")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # ---------------------------------------------------------
        # 1. OPTIMIZATION: Generate the Master Permutations
        # ---------------------------------------------------------
        master_permutations = []
        n_per_class = max_examples // 5

        for _ in range(permutations):
            balanced_examples = []
            for score in [1, 2, 3, 4, 5]:
                class_pool = train_df[train_df[target_dim] == score]
                if len(class_pool) >= n_per_class:
                    sampled_class = class_pool.sample(n=n_per_class)
                else:
                    sampled_class = class_pool.sample(n=n_per_class, replace=True)
                balanced_examples.append(sampled_class)
                
            shuffled_train = pd.concat(balanced_examples).sample(frac=1).reset_index(drop=True)
            master_permutations.append(shuffled_train)
        
        print(f"Generated {permutations} Stratified Master Contexts for KV Caching.")

        # ---------------------------------------------------------
        # 2. TESTING LOOP: With Active Telemetry
        # ---------------------------------------------------------
        fold_tracker = {}
        for test_row in test_df.to_dict(orient='records'):
            fold_tracker[test_row['id']] = {
                'text': test_row['text'],
                'true_label': test_row[target_dim],
                'prob_matrices': [],
                'logprob_matrices': [],
                'raw_responses': []
            }

        total_fold_rows = len(fold_tracker)

        # OUTER LOOP
        for perm_idx, perm_train in enumerate(master_permutations):
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Loading Master Context {perm_idx + 1}/3 into GPU VRAM...")
            
            # INNER LOOP
            for row_idx, (target_id, data) in enumerate(fold_tracker.items(), 1):
                prompt = format_prompt(perm_train, data['text'], target_dim, max_examples=max_examples)
            
                try:
                    pred_score, prob_vector, raw_logs, raw_text = query_ollama(prompt)
                    data['prob_matrices'].append(prob_vector)
                    data['logprob_matrices'].append(raw_logs)
                    data['raw_responses'].append(raw_text.strip())
                except Exception as e:
                    print(f"\nWarning: Token processing failed for ID {target_id}. Error: {e}")
                    data['prob_matrices'].append(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
                    data['logprob_matrices'].append(np.array([-1.609, -1.609, -1.609, -1.609, -1.609]))
                    data['raw_responses'].append(f"ERROR: {e}")

                # ACTIVE TELEMETRY: Print inline progress so you know the script hasn't hung
                print(f"\r  -> Inferencing row {row_idx}/{total_fold_rows}", end="", flush=True)
            print() # Clear the carriage return line after the context loop finishes

        # ---------------------------------------------------------
        # 3. AGGREGATION LOOP
        # ---------------------------------------------------------
        for target_id, data in fold_tracker.items():
            avg_probs = np.mean(data['prob_matrices'], axis=0)
            argmax_pred = np.argmax(avg_probs) + 1
            expected_val = np.dot(avg_probs, np.array([1, 2, 3, 4, 5]))
            
            avg_logprobs = np.log(avg_probs + 1e-12)
        
            r1 = data['raw_responses'][0] if len(data['raw_responses']) > 0 else ""
            r2 = data['raw_responses'][1] if len(data['raw_responses']) > 1 else ""
            r3 = data['raw_responses'][2] if len(data['raw_responses']) > 2 else ""
        
            all_ids.append(target_id)
            all_predictions.append(argmax_pred)
            all_expected_values.append(expected_val)
            all_true_labels.append(data['true_label'])
            all_prob_distributions.append(avg_probs)
            all_log_distributions.append(avg_logprobs)
            all_raw_1.append(r1)
            all_raw_2.append(r2)
            all_raw_3.append(r3)
        
            live_row = pd.DataFrame([{
                'id': target_id,
                'true_label': data['true_label'],
                'pred_argmax': argmax_pred,
                'pred_expected': expected_val,
                'logprob_1': avg_logprobs[0],
                'logprob_2': avg_logprobs[1],
                'logprob_3': avg_logprobs[2],
                'logprob_4': avg_logprobs[3],
                'logprob_5': avg_logprobs[4],
                'raw_1': r1,
                'raw_2': r2,
                'raw_3': r3
            }])
            
            live_row.to_csv(live_backup_csv, mode='a', header=False, index=False)
        
            processed_count += 1
            if processed_count % 10 == 0:
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{current_time}] TOTAL PIPELINE PROGRESS: {processed_count}/{total_records} records complete.")

    # Calculate Out-of-Fold Metrics
    kappa = cohen_kappa_score(all_true_labels, all_predictions, weights='quadratic')
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[1, 2, 3, 4, 5])
    logprob_matrix = np.array(all_log_distributions)
    
    cm_df = pd.DataFrame(
        cm, 
        index=[f"True {i}" for i in range(1, 6)], 
        columns=[f"Pred {i}" for i in range(1, 6)]
    )

    print(f"\n=== Final Metrics for Dimension: {target_dim} ===")
    print(f"Quadratic Weighted Kappa (k_w): {kappa:.4f}\n")
    print("Out-of-Fold Confusion Matrix:")
    print("-" * 45)
    print(cm_df)
    print("-" * 45, "\n")

    return {
        "id": all_ids,
        "true_labels": all_true_labels,
        "predictions_argmax": all_predictions,
        "predictions_expected_value": all_expected_values,
        "cohen_kappa": kappa,
        "logprob_1": logprob_matrix[:, 0],
        "logprob_2": logprob_matrix[:, 1],
        "logprob_3": logprob_matrix[:, 2],
        "logprob_4": logprob_matrix[:, 3],
        "logprob_5": logprob_matrix[:, 4],
        "raw_1": all_raw_1,
        "raw_2": all_raw_2,
        "raw_3": all_raw_3
    }

if __name__ == "__main__":
    # Alignment mapping configuration
    json_subspace_key = "vaccines_conspir_c"
    logical_dimension = "vaccines"
    
    print(f"Initializing evaluation tracking for subspace: {logical_dimension}...")
    df_data, theory_desc = load_data("latent_psychometric_data.json", json_subspace_key, logical_dimension)

    # Execute statistical evaluation matrix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = process_dimension(df_data, logical_dimension, n_splits=10, permutations=3)
    
    # Save Out-of-Fold arrays for target validation
    df_results = pd.DataFrame(results)
    csv_out = f"Vaccines.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"Pipeline complete. Statistical trace exported to: {csv_out}")