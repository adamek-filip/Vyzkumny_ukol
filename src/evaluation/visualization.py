import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# 1. Load your Out-of-Fold results
# Replace with your actual checkpoint filename
csv_filename = r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\ICL_Hidden_elites_FINAL.csv"
df = pd.read_csv(csv_filename,
                encoding="utf-8-sig",        # Automatically strips the \ufeff BOM from the header
                quoting=csv.QUOTE_NONE,      # CRITICAL: Disables text qualification mode
                sep=","                      # Forces strict splitting on every comma
)

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

# --- Example Usage ---
df_inverted = invert_likert_dataset(df)

# 2. Extract the clean vectors
y_true = df_inverted['true_labels']
y_pred = df_inverted['predictions_argmax']

# 3. Apply the boolean mask (Listwise Deletion) from our earlier solution
valid_mask = ~y_true.isna() & ~y_pred.isna()
y_true_clean = y_true[valid_mask]
y_pred_clean = y_pred[valid_mask]

cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[1, 2, 3, 4, 5])
# print("Confusion Matrix Successfully Computed:\n", cm)

# Define the psychometric scale for the axes
tick_labels = [
    "1",
    "2",
    "3",
    "4",
    "5"
]

global_accuracy = accuracy_score(y_true_clean, y_pred_clean)
global_mse = mean_squared_error(y_true_clean, y_pred_clean)
pearson_corr, _ = pearsonr(y_true_clean, y_pred_clean)
spearman_corr, _ = spearmanr(y_true_clean, y_pred_clean)

# 1. Initialize Figure with the State-of-the-Art Layout Engine
# 'layout="constrained"' replaces the need for tight_layout later
plt.figure(figsize=(8, 6), layout='constrained')
    
sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=tick_labels, 
        yticklabels=tick_labels
    )

# 2. Apply structural mapping 
plt.ylabel("True Value", fontweight='bold', labelpad=10)
plt.xlabel("Gemma 31B Predicted Score", fontweight='bold', labelpad=10)
#plt.xticks(rotation=45, ha='right')
#plt.yticks(rotation=0)

# 3. Apply Hierarchical Text Rendering
# The constrained engine will naturally push the heatmap down to fit this text within Y=1.0.
# We remove the y=1.02 parameter.
plt.suptitle("Method 2: Shadow Elites", fontsize=16, fontweight='bold')

header_text = (
        f"Accuracy: {global_accuracy:.3f}  |  "
        f"QWK: {df['cohen_kappa'].iloc[0]:.3f}  |  "
        f"Spearman ρ: {spearman_corr:.3f}  |  "
        f"Pearson r: {pearson_corr:.3f}"
    )

# The pad parameter creates calculated space specifically between this subtitle and the top of the heatmap
plt.title(header_text, pad=15, fontsize=10)

# 4. Render the plot (Note: plt.tight_layout() is strictly removed)
plt.show()

print("\n1. Global Classification Metrics (Dynamic Thresholds):")
print(f"  Accuracy:                      {global_accuracy:.4f}")

print("\n2. Global Regression Metrics (Continuous Floats):")
print(f"  Mean Squared Error (MSE):      {global_mse:.4f}")
print(f"  Pearson Correlation (Linear):  {pearson_corr:.4f}")
print(f"  Spearman Correlation (Rank):   {spearman_corr:.4f}")
print("="*55)