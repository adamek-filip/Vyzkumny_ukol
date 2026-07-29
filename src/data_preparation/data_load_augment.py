import pandas as pd


def filter_by_letter_count(df: pd.DataFrame, text_column: str = 'w4_llm_text', threshold: int = 300) -> pd.DataFrame:
    """
    Filters a DataFrame by counting strictly alphabetic characters in a text column.
    Omits all spaces, punctuation (.,!?), and numbers.
    """
    # 1. Force string type to prevent silent failures on nulls or pure-numeric rows
    text_series = df[text_column].astype(str)
    # 2. Count only alphabetic characters using regex: [^\W_] matches letters, excluding spaces, punctuation, and underscores
    letter_counts = text_series.str.count(r'[^\W_]')
    
    # 3. Create the boolean mask and filter the DataFrame
    mask = letter_counts >= threshold
    filtered_df = df[mask].copy()
    
    return filtered_df

def truncate_upper_bound(df: pd.DataFrame, column: str = 'w2_conspir_a', threshold: float = 20.0, keep_nans: bool = False) -> pd.DataFrame:
   
    y = pd.to_numeric(df[column], errors='coerce')
    mask = (y <= threshold)  
    filtered_df = df[mask].copy()
    
    return filtered_df



df = pd.read_csv("D:/Programovani/VYZKUMAK/data/ALL_DATA_w_55_and_conspir.csv")

df_filtered_chars = filter_by_letter_count(df, text_column='w4_llm_text', threshold=300)
df_ready_for_model = truncate_upper_bound(df_filtered_chars, threshold=20, keep_nans=False)


columns_to_omit = ['w4_llm_text']
df_pruned = df_ready_for_model.drop(columns=columns_to_omit, errors='ignore')