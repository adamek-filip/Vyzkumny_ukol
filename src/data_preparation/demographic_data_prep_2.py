import pandas as pd

df_nlp = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\LLM_conspir_d_300.csv")
df_demographic = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\Demographic_all_w_wave3.csv")


def align_cohorts(df_tabular, df_nlp, id_col='ID'):
    """
    Filters the tabular dataset to strictly match the respondents present in the NLP dataset.
    
    Parameters:
    df_tabular (pd.DataFrame): The full survey dataset (features).
    df_nlp (pd.DataFrame): The dataset containing the LLM predictions/texts.
    id_col (str): The name of the identifier column.
    
    Returns:
    pd.DataFrame: A filtered tabular dataframe with matching IDs, sorted to align with df_nlp.
    """
    # 1. Verify ID columns exist
    if id_col not in df_tabular.columns or id_col not in df_nlp.columns:
        raise KeyError(f"Column '{id_col}' must exist in both DataFrames.")
        
    # 2. Extract the unique IDs from the NLP dataset
    target_ids = df_nlp[id_col].unique()
    
    # 3. Filter the tabular dataset
    df_matched = df_tabular[df_tabular[id_col].isin(target_ids)].copy()
    
    # 4. Optional but recommended: Sort the tabular data to perfectly match the NLP data's row order.
    # This guarantees that when you run Cross-Validation, Fold 1 in Baseline = Fold 1 in LLM.
    df_matched.set_index(id_col, inplace=True)
    df_matched = df_matched.reindex(target_ids).reset_index()
    
    print(f"Original tabular shape: {df_tabular.shape}")
    print(f"Matched tabular shape: {df_matched.shape}")
    print(f"Missing IDs in tabular data: {len(target_ids) - len(df_matched)}")
    
    return df_matched


df = align_cohorts(df_demographic, df_nlp, id_col='ID')
df.to_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\Demographic_conspir_d.csv", index=False)
# df = df.set_index('ID')



print("\n--- 2. RAW VALUES IN 'w3_issue_sal_6' ---")
if 'w3_issue_sal_6' in df.columns:
    print(df['w3_issue_sal_6'].value_counts(dropna=False).head(10))
else:
    print("CRITICAL ERROR: 'w3_issue_sal_6' does not exist in df3.")
    
print("\n--- 3. RAW VALUES IN 'w3_pol_trust_1' ---")
if 'w3_pol_trust_1' in df.columns:
    print(df['w3_pol_trust_1'].value_counts(dropna=False).head(10))


suspect_ids = df.loc[df['w2_conspir_a'].isna(), 'ID'].tolist()

print(f"Found {len(suspect_ids)} suspect IDs.")

# 2. Extract their raw, untouched answers directly from the original df2 (Wave 2)
raw_evidence = df.loc[df['ID'].isin(suspect_ids), ['ID', 'w2_conspir_a']].copy()

# 3. Print the exact string representation and data type
print("\n--- RAW EVIDENCE DIAGNOSTIC ---")
for index, row in raw_evidence.iterrows():
    val = row['w2_conspir_a']
    print(f"ID: {row['ID']} | Raw Value: '{val}' | Type: {type(val)}")





def print_mapping_templates_for_script(df, exclude_cols=None):
    """
    Extracts unique strings from non-numeric columns and prints valid 
    Python dictionary syntax for direct copy-pasting.
    """
    if exclude_cols is None:
        # Exclude the ID column and your LLM targets by default
        exclude_cols = ['ID']
        
    cols_to_process = [c for c in df.columns if c not in exclude_cols]
    
    print("# ==========================================")
    print("# GENERATED MAPPING TEMPLATES")
    print("# ==========================================\n")
    
    for col in cols_to_process:
        # 1. Skip strictly numeric columns (e.g., float64, int64)
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # 2. Get unique values, dropping standard NaNs
        unique_vals = df[col].dropna().unique()
        
        # 3. Filter out global missing indicators to keep the dictionaries clean
        missing_indicators = ["NEVÍ", "Neví", "", " ", "nan"]
        clean_vals = [v for v in unique_vals if str(v).strip() not in missing_indicators]
        
        # If the column only contained numbers as strings and NaNs, skip it
        if not clean_vals:
            continue
            
        # 4. Print valid Python dictionary syntax
        print(f"MAP_{col.upper()} = {{")
        for val in clean_vals:
            # Escape quotes inside the string to prevent syntax errors
            safe_val = str(val).replace('"', '\\"')
            print(f'    "{safe_val}": 0.0,  # TODO: Assign continuous float')
        print("}\n")

# Execute the function on your raw dataset
print_mapping_templates_for_script(df)


