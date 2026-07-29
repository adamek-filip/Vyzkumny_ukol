import pandas as pd
import numpy as np
from scipy import stats
import csv

def invert_likert_dataset(df):
    # Load the dataset
    # df = pd.read_csv(file_path)
    
    # Create a copy to preserve the original data structure
    df_inv = df.copy()
    
    # 1. Invert the ground truth and aggregated predictions (6 - value)
    df_inv['true_labels'] = 6 - df['true_labels']
    df_inv['predictions_argmax'] = 6 - df['predictions_argmax']
    df_inv['predictions_expected_value'] = 6 - df['predictions_expected_value']
    
    # 2. Swap the log probabilities
    # 1 swaps with 5, 2 swaps with 4, 3 stays the same
    df_inv['logprob_1'] = df['logprob_5']
    df_inv['logprob_2'] = df['logprob_4']
    df_inv['logprob_3'] = df['logprob_3'] 
    df_inv['logprob_4'] = df['logprob_2']
    df_inv['logprob_5'] = df['logprob_1']
                
    return df_inv

def true_williams_t_test(y_true, y_pred1, y_pred2):
    n = len(y_true)
    if n <= 3:
        raise ValueError("Sample size must be greater than 3.")
        
    r_jk, _ = stats.pearsonr(y_true, y_pred1)
    r_jh, _ = stats.pearsonr(y_true, y_pred2)
    r_kh, _ = stats.pearsonr(y_pred1, y_pred2)
    
    # Determinant |R|
    det_R = 1 - r_jk**2 - r_jh**2 - r_kh**2 + (2 * r_jk * r_jh * r_kh)
    
    if det_R <= 0:
        return np.nan, np.nan, r_jk, r_jh
        
    # Mean of the two correlations being compared
    r_bar = (r_jk + r_jh) / 2.0
    
    # Williams' T2 Denominator components
    term1 = 2 * ((n - 1) / (n - 3)) * det_R
    term2 = (r_bar**2) * ((1 - r_kh)**3)
    
    # Williams' T2 Statistic
    t_stat = (r_jk - r_jh) * np.sqrt( ((n - 1) * (1 + r_kh)) / (term1 + term2) )
    
    # Two-tailed p-value with n-3 degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 3))
    
    return t_stat, p_value, r_jk, r_jh, r_kh

# ==========================================
# 1. Load the Data
# ==========================================
df_m1 = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\nato_M1.csv")
df_m2 = pd.read_csv(
    r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\ICL_Russia_NATO_FINAL.csv",
    encoding="utf-8-sig",
    quoting=csv.QUOTE_NONE,
    sep=","
)

df_inverted = invert_likert_dataset(df_m2)

# ==========================================
# 2. Merge Dataframes on ID
# ==========================================
# WARNING: Change 'participant_id' to the actual name of the ID column in your CSVs.
# If the column names differ between CSVs, use: left_on='ID_M1', right_on='ID_M2'
m1_id_col = 'participant_id'
m2_id_col = 'id' # Adjust this to match your ICL dataset's ID column name

# THE FIX: Cast to string AND strip out any literal double quotes
df_m1[m1_id_col] = df_m1[m1_id_col].astype(str).str.replace('"', '')
df_inverted[m2_id_col] = df_inverted[m2_id_col].astype(str).str.replace('"', '')

# Perform the merge
if m1_id_col == m2_id_col:
    merged_df = pd.merge(df_m1, df_inverted, on=m1_id_col, how='inner')
else:
    merged_df = pd.merge(df_m1, df_inverted, left_on=m1_id_col, right_on=m2_id_col, how='inner')

# ==========================================
# 3. Extract Aligned Arrays
# ==========================================
# We only need one y_true since the merge guarantees they are the same person
y_true = merged_df['y_true'].values
y_pred1 = merged_df['y_pred_cont'].values
y_pred2 = pd.to_numeric(merged_df['predictions_argmax'], errors='coerce').values


t_stat, p_val, r_m1, r_m2, r_m3 = true_williams_t_test(y_true, y_pred1, y_pred2)

print("Target: Conspiracy D (NATO War Provocation)")
print("-" * 45)
print(f"Total participants matched : {len(merged_df)}")
print(f"Valid rows analyzed        : {len(y_true)}")
print(f"Method 1 Pearson r         : {r_m1:.3f}")
print(f"Method 2 Pearson r         : {r_m2:.3f}")
print(f"Connected Pearson r         : {r_m3:.3f}")
print(f"Williams' t-statistic      : {t_stat:.3f}")
print(f"p-value                    : {p_val:.3f}")