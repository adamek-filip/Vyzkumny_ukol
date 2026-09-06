import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import joblib
from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# Context Manager for inner CV tracking
# ==========================================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into a tqdm progress bar.
    Intercepts the parallel processing engine Scikit-Learn uses under the hood.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# ==========================================
# Core CV Execution
# ==========================================
def run_nested_cv(X: pd.DataFrame, y: pd.Series, feature_names: list):
    """
    Executes a 10x10 Nested Cross-Validation for Linear Regression on Ordinal targets.
    Incorporates empirical CDF matching for dynamic thresholding.
    """
    cv_params = {'n_splits': 10, 'shuffle': True, 'random_state': 42}
    outer_cv = StratifiedKFold(**cv_params)
    inner_cv = StratifiedKFold(**cv_params)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('enet', ElasticNet(max_iter=10000, random_state=42))
    ])
    
    param_grid = {
        'enet__alpha': np.logspace(-4, 2, 10),
        'enet__l1_ratio': np.linspace(0.01, 1.0, 10) 
    }

    # Extract unique sorted classes directly from the data
    classes = np.sort(np.unique(y))
    aggregated_cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    global_y_true, global_y_pred_cont, global_y_pred_disc = [], [], []
    fold_coefs_list = []
    global_ids = []

    print("Beginning 10x10 Nested CV Process with Dynamic Thresholding...\n")
    
    outer_splits = list(outer_cv.split(X, y))
    
    with tqdm(total=len(outer_splits), desc="Outer CV Progress", unit="fold") as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Set verbose=0 to silence default scikit-learn output
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0 
            )
            
            # Calculate total fits for the inner progress bar
            n_candidates = len(param_grid['enet__alpha']) * len(param_grid['enet__l1_ratio'])
            total_fits = n_candidates * inner_cv.get_n_splits()
            
            # Execute GridSearch wrapped in our tqdm context manager
            with tqdm_joblib(tqdm(desc=f"Tuning Fold {fold_idx}/10", total=total_fits, leave=False, unit="fit")):
                grid.fit(X_train, y_train)


            best_pipe = grid.best_estimator_

            # Dynamically request the feature names that survived the imputer
            surviving_features = best_pipe.named_steps['scaler'].get_feature_names_out(
                best_pipe.named_steps['imputer'].get_feature_names_out(feature_names)
            )
            
            # Map coefficients to their actual features for this specific fold
            current_coefs = pd.Series(
                best_pipe.named_steps['enet'].coef_, 
                index=surviving_features
            )
            fold_coefs_list.append(current_coefs)

            # ---------------------------------------------------------
            # THE DYNAMIC THRESHOLDING LOGIC
            # ---------------------------------------------------------
            # 1. Get the continuous predictions on the training data
            preds_train_cont = grid.predict(X_train)
            
            # 2. Calculate the empirical cumulative distribution of the training target
            class_proportions = y_train.value_counts(normalize=True).sort_index()
            cum_probs = class_proportions.cumsum().values
            
            # 3. Find the exact quantiles in the training predictions. 
            # We drop the final probability (1.0) because K classes only need K-1 thresholds.
            thresholds = np.percentile(preds_train_cont, cum_probs[:-1] * 100)
            
            # 4. Predict on the unseen test data
            preds_test_cont = grid.predict(X_test)
            
            # 5. Map test predictions to classes using the learned boundaries
            bin_indices = np.digitize(preds_test_cont, bins=thresholds)
            preds_test_disc = classes[bin_indices]
            # ---------------------------------------------------------

            # Aggregate Metrics
            fold_cm = confusion_matrix(y_test, preds_test_disc, labels=classes)
            aggregated_cm += fold_cm
            
            global_y_true.extend(y_test.values)
            global_y_pred_cont.extend(preds_test_cont)
            global_y_pred_disc.extend(preds_test_disc)
            global_ids.extend(X_test.index.tolist())
            
            # Clean terminal output using tqdm.write
            tqdm.write(
                f"Fold {fold_idx:02d} | "
                f"Best Params: Alpha={grid.best_params_['enet__alpha']:.4f}, "
                f"L1_Ratio={grid.best_params_['enet__l1_ratio']:.2f} | "
                f"Learned Thresholds: {np.round(thresholds, 2)}"
            )
            pbar.update(1)

    # Compute Global Metrics
    global_y_true = np.array(global_y_true)
    global_y_pred_cont = np.array(global_y_pred_cont)
    global_y_pred_disc = np.array(global_y_pred_disc)
    global_ids = np.array(global_ids)

    df_results = pd.DataFrame({
    'participant_id': global_ids,
    'y_true': global_y_true,
    'y_pred_cont': global_y_pred_cont,
    'y_pred_disc': global_y_pred_disc
    })
    
    df_results.to_csv(conspir +"_M1.csv", index=False)

    global_accuracy = accuracy_score(global_y_true, global_y_pred_disc)
    global_mse = mean_squared_error(global_y_true, global_y_pred_cont)
    global_r2 = r2_score(global_y_true, global_y_pred_cont)
    pearson_corr, _ = pearsonr(global_y_true, global_y_pred_cont)
    spearman_corr, _ = spearmanr(global_y_true, global_y_pred_cont)
    global_qwk = cohen_kappa_score(global_y_true, global_y_pred_disc, labels=classes, weights='quadratic')

    coef_matrix = pd.DataFrame(fold_coefs_list).fillna(0.0)
    
    mean_coefs = coef_matrix.mean(axis=0)
    std_coefs = coef_matrix.std(axis=0)
    
    coef_df = pd.DataFrame({
        'Feature': mean_coefs.index,
        'Mean_Beta': mean_coefs.values,
        'Std_Beta': std_coefs.values,
        'Abs_Mean_Beta': np.abs(mean_coefs.values)
    }).sort_values(by='Abs_Mean_Beta', ascending=False).reset_index(drop=True)

    # Print Final Report
    print("\n" + "="*55)
    print("NESTED CV RESULTS (AGGREGATED ACROSS ALL FOLDS)")
    print("="*55)
    
    print("\n1. Global Classification Metrics (Dynamic Thresholds):")
    print(f"  Accuracy:                      {global_accuracy:.4f}")
    print(f"  Quadratic Weighted Kappa:      {global_qwk:.4f}")
    
    print("\n2. Global Regression Metrics (Continuous Floats):")
    print(f"  Mean Squared Error (MSE):      {global_mse:.4f}")
    print(f"  R-Squared (R2):                {global_r2:.4f}")
    print(f"  Pearson Correlation (Linear):  {pearson_corr:.4f}")
    print(f"  Spearman Correlation (Rank):   {spearman_corr:.4f}")
    print("="*55)

    # Visualization
    plt.figure(figsize=(8, 6))
    string_classes = [str(c) for c in classes]
    
    sns.heatmap(
        aggregated_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=string_classes, 
        yticklabels=string_classes
    )
    plt.xlabel('Elastic Net Predicted Value', fontweight='bold')
    plt.ylabel('True Value', fontweight='bold')
    plt.suptitle('Method 1: ' + conspir, fontsize=16, fontweight='bold')

    header_text = (
        f"Accuracy: {global_accuracy:.3f}  |  "
        f"QWK: {global_qwk:.3f}  |  "
        f"Spearman \u03c1: {spearman_corr:.3f}  |  "
        f"Pearson r: {pearson_corr:.3f}"
    )

    plt.title(header_text, pad=10, fontsize=10)
    plt.tight_layout()
    plt.show()

    CODEBOOK_MAP = {
        'X1_1': 'Policy Domain Focus: Economic',
        'X1_2': 'Policy Domain Focus: Social welfare',
        'X1_3': 'Policy Domain Focus: Infrastructure',
        'X1_4': 'Policy Domain Focus: Environmental',
        'X1_5': 'Policy Domain Focus: Security/Defense',
        'X1_6': 'Policy Domain Focus: Education',
        'X1_7': 'Policy Domain Focus: Healthcare',
        'X2_1': 'Economic Orientation: State intervention preference',
        'X2_2': 'Economic Orientation: Market solution preference',
        'X2_3': 'Economic Orientation: Mixed approach',
        'X3_1': 'Problem-Solution Balance: Problem identification',
        'X3_2': 'Problem-Solution Balance: Solution proposal',
        'X3_3': 'Problem-Solution Balance: Implementation specificity',
        'X4_1': 'Target Groups Mentioned: Elderly',
        'X4_2': 'Target Groups Mentioned: Working class',
        'X4_3': 'Target Groups Mentioned: Middle class',
        'X4_4': 'Target Groups Mentioned: Youth',
        'X4_5': 'Target Groups Mentioned: Businesses',
        'X4_6': 'Target Groups Mentioned: Other specific groups',
        'X5_1': 'Language Style: Formality level',
        'X5_2': 'Language Style: Technical terminology use',
        'X5_3': 'Language Style: Colloquial expressions',
        'X5_4': 'Language Style: Emotional content',
        'X6_1': 'Argumentation Style: Ideological',
        'X6_2': 'Argumentation Style: Pragmatic',
        'X6_3': 'Argumentation Style: Expert-based',
        'X6_4': 'Argumentation Style: Experience-based',
        'X7_1': 'Certainty Markers: Commitment to statements',
        'X7_2': 'Certainty Markers: Use of hedging',
        'X7_3': 'Certainty Markers: Conditional statements',
        'X8_1': 'Time Orientation: Past focus',
        'X8_2': 'Time Orientation: Present focus',
        'X8_3': 'Time Orientation: Future focus',
        'X9_1': 'Appeal Types: Logical appeal',
        'X9_2': 'Appeal Types: Emotional appeal',
        'X9_3': 'Appeal Types: Authority appeal',
        'X9_4': 'Appeal Types: Moral appeal',
        'X10_1': 'Position Taking: Critical of status quo',
        'X10_2': 'Position Taking: Supportive of status quo',
        'X10_3': 'Position Taking: Reform-oriented',
        'X11_1': 'Agency Attribution: Personal agency ("I will")',
        'X11_2': 'Agency Attribution: Collective agency ("We should")',
        'X11_3': 'Agency Attribution: System responsibility',
        'X12_1': 'Scope of Change Proposed: Incremental',
        'X12_2': 'Scope of Change Proposed: Systemic',
        'X12_3': 'Scope of Change Proposed: Revolutionary',
        'X13_1': 'Political Context Awareness: Local issues',
        'X13_2': 'Political Context Awareness: National issues',
        'X13_3': 'Political Context Awareness: International issues',
        'X14_1': 'Implementation Level: Local',
        'X14_2': 'Implementation Level: Regional',
        'X14_3': 'Implementation Level: National',
        'X15_1': 'Resource Consideration: Cost awareness',
        'X15_2': 'Resource Consideration: Funding sources mentioned',
        'X15_3': 'Resource Consideration: Implementation feasibility',
    }

    # 2. Extract top 15 features from your model coefficients
    top_features = coef_df.head(15).copy()

    # 3. Create formatted, wrapped labels for the plot axis
    wrapped_labels = []
    for _, row in top_features.iterrows():
        feat_name = row['Feature']
        description = CODEBOOK_MAP.get(feat_name, 'Label not found')
        
        full_text = f"{feat_name}: {description}"
        
        # Wrap text at 45 characters max per line using newline breaks (\n)
        wrapped_text = "\n".join(textwrap.wrap(full_text, width=45))
        wrapped_labels.append(wrapped_text)

    # Assign wrapped configurations back to a plotting helper column
    top_features['Plot_Label'] = wrapped_labels

    # 4. Generate the optimized plot layout
    plt.figure(figsize=(11, 9))  # Adjusted aspect ratio slightly higher for stacked text rows

    sns.barplot(
        x='Mean_Beta', 
        y='Plot_Label',  # Use our clean, multi-line wrapped labels
        data=top_features, 
        color='steelblue', 
        xerr=top_features['Std_Beta'],
        capsize=0.2
    )

    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Top 15 Predictors for Method 1: {conspir}, Elastic Net (Mean $\\beta$ ± 1 SD)", fontweight='bold', fontsize=13, pad=15)
    plt.xlabel("Standardized Coefficient ($\\beta$)", fontweight='bold', labelpad=10)
    plt.ylabel("Feature Explanation", fontweight='bold', labelpad=10)

    # Tight layout automatically respects our multi-line wrapped margins
    plt.tight_layout()
    plt.show()

    # 5. Keep the console log perfectly single-lined for cleaner text inspection
    # Ensure the key passed to dict.get is a string to avoid type-checking issues with NA values
    top_features['Console_Label'] = top_features['Feature'].map(lambda x: CODEBOOK_MAP.get(str(x), 'Label not found'))
    print("\nTop 15 Feature Importances:")
    print(top_features[['Feature', 'Console_Label', 'Mean_Beta', 'Std_Beta']].to_string(index=False))


if __name__ == "__main__":
    df = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\300_conspir_c_55dim.csv")
    df = df.set_index('ID')

    y = df['w2_conspir_c']
    y = 6 - y
    X = df.drop(columns=['w2_conspir_a','w2_conspir_b','w2_conspir_c','w2_conspir_d'])
    conspir = "covid"  # Update this string to reflect the specific conspiracy being analyzed

    y = y.astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')

    feature_names = X.columns.tolist()

    run_nested_cv(X, y, feature_names)
    pass
