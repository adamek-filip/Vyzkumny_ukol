import pandas as pd
import numpy as np


df1 = pd.read_spss(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\PNS 2406 CoRe CAB W1_FINAL.sav")
df2 = pd.read_spss(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\PNS 2409 CoRe CAB W2_FINAL.sav")
df3 = pd.read_spss(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\PNS 2411 CoRe CAB W3_FINAL.sav")

# 1. Standardize IDs across ALL files first (as we did before)
for df in [df1, df2, df3]:
    df['ID'] = df['ID'].astype(str).str.strip()

# 2. Find the 'Master' ID list to ensure no one is dropped by mistake
all_ids = set(df1['ID']).union(set(df2['ID'])).union(set(df3['ID']))

# 3. Create a master ID reference dataframe
master_df = pd.DataFrame({'ID': list(all_ids)})

columns_from_wave1 = ["ID", "w1_IDE_2", "w1_IDE_5b", "w1_IDE_19", "w1_IDE_8", "w1_VZD", "w1_B9a", "w1_B21"]

columns_from_wave2 = ["ID", "w2_conspir_a", "w2_conspir_b", "w2_conspir_c", "w2_conspir_d", "w2_childvax_pomocna"]

columns_from_wave3= [
    "pol_trust_1", "pol_trust_2", "pol_trust_5", "pol_trust_8", "pol_trust_9", "gen_trust",
    "sat_dem", "pol_interest", "left_right",
    "anx_1", "anx_2", "anx_3", "anx_4", "anger_1", "fear_1", "fear_3",
    "resil_1", "resil_2", "resil_3", "resil_4", "resil_5", "resil_6",
    "sub_inc", "fin_hshld", "econ_ret", "feltsec",
    "threat_1", "threat_2", "threat_3", "threat_4", "threat_5", "threat_6",
    "mig_att_1", "ref_att_1", "issue_sal_1", "issue_sal_2", "issue_sal_3", "issue_sal_4", "issue_sal_6"
]

w3_features = ["ID"] + [f"w3_{base}" for base in columns_from_wave3]

# Combine lists and remove duplicates while preserving exact order
wave1_columns_unique = list(dict.fromkeys(columns_from_wave1))
wave2_columns_unique = list(dict.fromkeys(columns_from_wave2))
wave3_columns_unique = list(dict.fromkeys(w3_features))

# Extract subsets
df_w1 = df1[wave1_columns_unique].copy()
df_w2 = df2[wave2_columns_unique].copy()
df_w3 = df3[wave3_columns_unique].copy()

# 4. Merge sequentially onto the master list (Left Join)
# This guarantees that even if a user is missing from Wave 1, 
# they are kept in the dataset as long as they appear in the master list.
df_combined = pd.merge(master_df, df_w1, on='ID', how='left')
df_combined = pd.merge(df_combined, df_w2, on='ID', how='left')
df_combined = pd.merge(df_combined, df_w3, on='ID', how='left')


# ==========================================
# 3. FEATURE ENGINEERING (AGE BINNING)
# ==========================================
# Coerce age to numeric to clear any textual artifacts
df_combined['w1_IDE_2'] = pd.to_numeric(df_combined['w1_IDE_2'], errors='coerce')

# right=False creates intervals [a, b). 
# np.inf ensures the top bin never breaks if max age shifts in future data.
bins = [0, 30, 50, 65, np.inf]
labels = ['15-29', '30-49', '50-64', '65+']

df_combined['age_group'] = pd.cut(
    df_combined['w1_IDE_2'], 
    bins=bins, 
    labels=labels, 
    right=False, 
    include_lowest=True
)

# Drop the continuous variable to prevent multicollinearity in models
df_combined = df_combined.drop(columns=['w1_IDE_2'])

print("\nFinal DataFrame constructed successfully.")
print(f"Total rows (N): {df_combined.shape[0]}")
print(f"Total columns (Features): {df_combined.shape[1]}")
print(df1.shape[0], df2.shape[0])  # Original vs. Merged row counts

# df_combined.to_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\Demographic_all_w_wave3.csv", index=False)






# ISSUE CHECK

DF_TO_CHECK = df_combined.copy()  # Use a copy to avoid modifying the original DataFrame 
STAGE_NAME = "Merged Dataframe (df_combined)" # Give it a name for your console output

suspect_ids = [11112308, 11112439, 11113320, 11113889, 11113969, 11113970, 11113971]
target_col = "w2_conspir_a"

print(f"\n--- AUDIT: {STAGE_NAME} ---")

if target_col not in DF_TO_CHECK.columns:
    print(f"Result: The column '{target_col}' DOES NOT EXIST in this dataframe.")
else:
    # Standardize the IDs in the dataframe temporarily for a safe comparison
    safe_ids = DF_TO_CHECK['ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    for user_id in suspect_ids:
        mask = (safe_ids == str(user_id))
        
        if not mask.any():
            print(f"ID {user_id}: DROPPED (Row completely missing)")
        else:
            val = DF_TO_CHECK.loc[mask, target_col].values[0]
            if pd.isna(val):
                print(f"ID {user_id}: NaN (Missing Data)")
            else:
                print(f"ID {user_id}: '{val}' (Type: {type(val).__name__})")
print("-" * 35)


