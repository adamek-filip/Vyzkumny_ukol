import pandas as pd
import numpy as np
import re

df_aligned = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\Demographic_conspir_d.csv")
print(f"Initial shape of aligned DataFrame: {df_aligned.shape}")
# ==========================================
# 1. UNIFIED ORDINAL SCALES
# ==========================================

MAP_AGREEMENT = {
    "Rozhodně nesouhlasím": 1.0,
    "Spíše nesouhlasím": 2.0,
    "Ani souhlas, ani nesouhlas": 3.0, "Tak napůl": 3.0,
    "Spíše souhlasím": 4.0,
    "Rozhodně souhlasím": 5.0
}

MAP_TRUST = {
    "Rozhodně nedůvěřuji": 1.0,
    "Spíše nedůvěřuji": 2.0,
    "Spíše důvěřuji": 3.0,
    "Rozhodně důvěřuji": 4.0
}

MAP_SALIENCE = {
    "Málo důležité": 1.0,   
    "Středně důležité": 2.0,
    "Hodně důležité": 3.0
}

MAP_FREQUENCY_ANX = {
    "Vůbec ne": 1.0,
    "Několik dní": 2.0,
    "Více než polovinu dní": 3.0,
    "Téměř každý den": 4.0
}

MAP_RETROSPECTIVE = {
    "Hodně zhoršila": 1.0, "Hodně zhoršil": 1.0,
    "Spíše zhoršila": 2.0, "Spíše zhoršil": 2.0,
    "Nezměnila": 3.0,      "Nezměnil": 3.0,
    "Spíše zlepšila": 4.0, "Spíše zlepšil": 4.0,
    "Hodně zlepšila": 5.0, "Hodně zlepšil": 5.0
}

MAP_BINARY = {
    "Ne": 0.0,
    "Ano": 1.0
}

# Unique scales that don't repeat
MAP_SAT_DEM = {
    "Velmi nespokojen/a": 1.0, "Spíše nespokojen/a": 2.0,
    "Spíše spokojen/a": 3.0, "Velmi spokojen/a": 4.0
}

MAP_FELTSEC = {
    "Vůbec ne bezpečně": 1.0, "Nepříliš bezpečně": 2.0,
    "Poměrně bezpečně": 3.0, "Velmi bezpečně": 4.0
}

MAP_SUB_INC = {
    "Se současným příjmem se vychází velice těžko.": 1.0,
    "Se současným příjmem se vychází těžko.": 2.0,
    "Se současným příjmem lze vyjít.": 3.0,
    "Se současným příjmem se žije pohodlně.": 4.0
}

MAP_VZD = {
    "ZŠ + BEZ MAT": 1.0,
    "S MAT": 2.0,
    "VŠ": 3.0
}

MAP_POL_INTEREST = {
    "Vůbec ne": 1.0, "Jen trochu": 2.0,
    "Docela": 3.0, "Velmi": 4.0
}

MAP_GENDER = {
    'Muž': 0, 
    'Žena': 1
}

MAP_URBANIZATION = {
    'Malá vesnice, osada, samota': 1,
    'Velká vesnice': 2,
    'Malé město': 3,
    'Středně velké město': 4,
    'Předměstí velkého města': 5,
    'Velké město': 6
}


# ==========================================
# 2. COLUMN CATEGORIZATION
# ==========================================

# REMOVED conspiracies from here so they aren't processed by the main automation
COLS_AGREEMENT = ["w3_resil_1", "w3_resil_2", "w3_resil_3", "w3_resil_4", "w3_resil_5", "w3_resil_6", 
                  "w3_mig_att_1"]

COLS_TRUST = ["w3_pol_trust_1", "w3_pol_trust_2", "w3_pol_trust_5", "w3_pol_trust_8", "w3_pol_trust_9"]
COLS_SALIENCE = ["w3_issue_sal_1", "w3_issue_sal_2", "w3_issue_sal_3", "w3_issue_sal_4", "w3_issue_sal_6"]
COLS_ANXIETY = ["w3_anx_1", "w3_anx_2", "w3_anx_3", "w3_anx_4"]
COLS_RETROSPECTIVE = ["w3_fin_hshld", "w3_econ_ret"]
COLS_BINARY = [] 

COLS_REGEX = ["w3_threat_1", "w3_threat_2", "w3_threat_3", "w3_threat_4", "w3_threat_5", "w3_threat_6", 
              "w3_gen_trust", "w3_anger_1", "w3_left_right", "w3_fear_1", "w3_fear_3", "w3_ref_att_1"]

COLS_NOMINAL = ["w1_IDE_8", "w1_B21", "w1_B9a", "w1_IDE_5b", "w1_IDE_19", "age_group", "w2_childvax_pomocna"]

# ADDED: Target variables isolated so they aren't dummified or swept
COLS_TARGETS = ["w2_conspir_a", "w2_conspir_b", "w2_conspir_c", "w2_conspir_d"]


def extract_leading_number(val, nevi_constant=0.0):
    """Regex extractor for scales like '7 Cítím silnou obavu'."""
    val_str = str(val).strip()
    if pd.isna(val) or val_str == "" or val_str.lower() == "nan":
        return np.nan
    
    if val_str == "NEVÍ":
        return nevi_constant
        
    match = re.search(r'\d+', val_str)
    return float(match.group()) if match else np.nan

def construct_design_matrix(df, nevi_constant=0.0):
    X = df.copy()
    STRICT_NEVI = "NEVÍ"
    
    # 1. Aggressive Global Standardization
    string_cols = X.select_dtypes(include=['object', 'string']).columns
    
    for col in string_cols:
        X[col] = X[col].str.strip()
        X[col] = X[col].replace(r'(?i)^\s*nev[íi]m?\s*$', STRICT_NEVI, regex=True)
    
    X.replace(["", " "], np.nan, inplace=True)    
    
    # 2. Apply Consolidated Maps
    mapping_groups = [
        (COLS_AGREEMENT, MAP_AGREEMENT),
        (COLS_TRUST, MAP_TRUST),
        (COLS_SALIENCE, MAP_SALIENCE),
        (COLS_ANXIETY, MAP_FREQUENCY_ANX),
        (COLS_RETROSPECTIVE, MAP_RETROSPECTIVE),
        (COLS_BINARY, MAP_BINARY)
    ]
    
    for columns, mapping in mapping_groups:
        safe_mapping = {**mapping, STRICT_NEVI: nevi_constant}
        for col in columns:
            if col in X.columns:
                is_nevi = (X[col] == STRICT_NEVI)
                if is_nevi.any():
                    X[f"{col}_is_nevi"] = is_nevi.astype(int)
                X[col] = X[col].map(safe_mapping)
                
    # 3. Apply Unique Maps
    unique_maps = {
        "w3_sat_dem": MAP_SAT_DEM, "w3_feltsec": MAP_FELTSEC,
        "w3_sub_inc": MAP_SUB_INC, "w1_VZD": MAP_VZD,
        "w3_pol_interest": MAP_POL_INTEREST, 
        "w1_IDE_8": MAP_GENDER, "w1_IDE_19": MAP_URBANIZATION
    }
    
    for col, mapping in unique_maps.items():
        if col in X.columns:
            safe_mapping = {**mapping, STRICT_NEVI: nevi_constant}
            is_nevi = (X[col] == STRICT_NEVI)
            if is_nevi.any():
                X[f"{col}_is_nevi"] = is_nevi.astype(int)
            X[col] = X[col].map(safe_mapping)
            
    # 4. Apply Regex Extraction
    for col in COLS_REGEX:
        if col in X.columns:
            is_nevi = (X[col] == STRICT_NEVI)
            if is_nevi.any():
                X[f"{col}_is_nevi"] = is_nevi.astype(int)
            X[col] = X[col].apply(lambda x: extract_leading_number(x, nevi_constant))
            
    # 5. The Global Sweep 
    # UPDATED: Explicitly EXCLUDE nominal AND target columns
    sweep_cols = [col for col in string_cols if col not in COLS_NOMINAL and col not in COLS_TARGETS]
    
    for col in sweep_cols:
        if col in X.columns:
            is_nevi = (X[col] == STRICT_NEVI)
            if is_nevi.any():
                X[f"{col}_is_nevi"] = is_nevi.astype(int)
                X.loc[is_nevi, col] = str(nevi_constant)
            
            X[col] = pd.to_numeric(X[col], errors='coerce')

    return X


# Final Execution of the Base Matrix
df_design = construct_design_matrix(df_aligned)

# Dummify only the nominal columns
df_design = pd.get_dummies(
    df_design, 
    columns=[col for col in COLS_NOMINAL if col in df_design.columns], 
    drop_first=True,
    dtype=float
)


# ==========================================
# 3. SELECT TARGET, MAP, AND EXPORT SINGLE DATASET
# ==========================================

# 1. Define which target you want to keep for this specific export
CHOSEN_TARGET = "w2_conspir_d"  # Update this to 'w2_conspir_b', 'c', or 'd' as needed

# 2. Map the chosen target to the 1-5 scale
if CHOSEN_TARGET in df_design.columns:
    df_design[CHOSEN_TARGET] = df_design[CHOSEN_TARGET].map(MAP_AGREEMENT)

# 3. Identify and drop the other conspiracy targets
targets_to_drop = [col for col in COLS_TARGETS if col != CHOSEN_TARGET and col in df_design.columns]
df_design = df_design.drop(columns=targets_to_drop)

# 4. Export the final dataset
letter = CHOSEN_TARGET[-1] # Extracts 'a', 'b', 'c', or 'd' for the filename
# out_path = rf"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\df_design_{letter}_demog.csv"

# df_design.to_csv(out_path, index=False)

# print(f"Successfully mapped {CHOSEN_TARGET} and exported to: {out_path}")
# print(f"Dropped unused targets: {targets_to_drop}")

# ==========================================
# DIAGNOSTIC: MISSINGNESS LEDGER
# ==========================================
def audit_missingness(df_final):
    """Calculates the volume and percentage of missing values per feature."""
    missing_counts = df_final.isna().sum()
    missing_pct = (missing_counts / len(df_final)) * 100
    
    report = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct
    })
    
    report = report[report['Missing_Count'] > 0].sort_values(by='Missing_Count', ascending=False)
    print(f"\nTotal features with missing values: {len(report)} / {df_final.shape[1]}\n")
    print(report.round(2).to_string())

# Run diagnostic on the base design matrix before the split
# audit_missingness(df_design)