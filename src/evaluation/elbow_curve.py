import textwrap
import os
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
def run_nested_cv(X: pd.DataFrame, y: pd.Series, feature_names: list, conspir: str, verbose_plot: bool = False):
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

    classes = np.sort(np.unique(y))
    aggregated_cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    global_y_true, global_y_pred_cont, global_y_pred_disc = [], [], []
    fold_coefs_list = []
    
    # NEW: Track per-fold Pearson correlation for the elbow curve variance
    fold_pearson_corrs = []

    outer_splits = list(outer_cv.split(X, y))
    
    with tqdm(total=len(outer_splits), desc="Outer CV Progress", unit="fold", leave=False) as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0 
            )
            
            n_candidates = len(param_grid['enet__alpha']) * len(param_grid['enet__l1_ratio'])
            total_fits = n_candidates * inner_cv.get_n_splits()
            
            with tqdm_joblib(tqdm(desc=f"Tuning Fold {fold_idx}/10", total=total_fits, leave=False, unit="fit")):
                grid.fit(X_train, y_train)

            best_pipe = grid.best_estimator_

            surviving_features = best_pipe.named_steps['scaler'].get_feature_names_out(
                best_pipe.named_steps['imputer'].get_feature_names_out(feature_names)
            )
            
            current_coefs = pd.Series(
                best_pipe.named_steps['enet'].coef_, 
                index=surviving_features
            )
            fold_coefs_list.append(current_coefs)

            preds_train_cont = grid.predict(X_train)
            class_proportions = y_train.value_counts(normalize=True).sort_index()
            cum_probs = class_proportions.cumsum().values
            thresholds = np.percentile(preds_train_cont, cum_probs[:-1] * 100)
            
            preds_test_cont = grid.predict(X_test)
            bin_indices = np.digitize(preds_test_cont, bins=thresholds)
            preds_test_disc = classes[bin_indices]

            fold_cm = confusion_matrix(y_test, preds_test_disc, labels=classes)
            aggregated_cm += fold_cm
            
            global_y_true.extend(y_test.values)
            global_y_pred_cont.extend(preds_test_cont)
            global_y_pred_disc.extend(preds_test_disc)
            
            # Calculate and store fold-level Pearson correlation
            fold_r, _ = pearsonr(y_test, preds_test_cont)
            fold_pearson_corrs.append(fold_r)
            
            tqdm.write(
                f"Fold {fold_idx:02d} | Alpha={grid.best_params_['enet__alpha']:.4f}, "
                f"L1={grid.best_params_['enet__l1_ratio']:.2f} | Fold r={fold_r:.3f}"
            )
            pbar.update(1)

    # Calculate aggregations required for the return dictionary
    mean_fold_r = np.mean(fold_pearson_corrs)
    std_fold_r = np.std(fold_pearson_corrs)
    
    global_y_true = np.array(global_y_true)
    global_y_pred_cont = np.array(global_y_pred_cont)
    global_y_pred_disc = np.array(global_y_pred_disc)

    global_accuracy = accuracy_score(global_y_true, global_y_pred_disc)
    pearson_corr_global, _ = pearsonr(global_y_true, global_y_pred_cont)

    # Retained your specific plotting logic inside the toggle block
    if verbose_plot:
        _generate_plots(aggregated_cm, classes, conspir, global_accuracy, pearson_corr_global, fold_coefs_list)

    return {
        'mean_r': mean_fold_r,
        'std_r': std_fold_r,
        'global_r': pearson_corr_global,
        'sample_size': len(y)
    }

def _generate_plots(aggregated_cm, classes, conspir, global_accuracy, pearson_corr, fold_coefs_list):
    """Helper function to offload your existing plotting logic to keep the main function clean."""
    plt.figure(figsize=(8, 6))
    string_classes = [str(c) for c in classes]
    sns.heatmap(aggregated_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=string_classes, yticklabels=string_classes)
    plt.xlabel('Elastic Net Predicted Value', fontweight='bold')
    plt.ylabel('True Value', fontweight='bold')
    plt.title(f'Accuracy: {global_accuracy:.3f} | Pearson r: {pearson_corr:.3f}')
    plt.show()

# ==========================================
# Elbow Curve Implementation
# ==========================================
def plot_elbow_curve(threshold_results, title="Validation of Minimal Character Count"):
    thresholds = sorted(threshold_results.keys())
    means = np.array([threshold_results[t]['mean_r'] for t in thresholds])
    stds = np.array([threshold_results[t]['std_r'] for t in thresholds])
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, means, marker='o', linestyle='-', color='indigo', label='Mean Fold Pearson $r$')
    plt.fill_between(thresholds, means - stds, means + stds, color='indigo', alpha=0.15, label='± 1 Std Dev')
    
    # Annotate sample size at each point
    for t in thresholds:
        n = threshold_results[t]['sample_size']
        plt.annotate(f"n={n}", (t, threshold_results[t]['mean_r']), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.title(title, fontweight='bold')
    plt.xlabel('Minimal Character Count Threshold', fontweight='bold')
    plt.ylabel('Out-of-Sample Pearson Correlation', fontweight='bold')
    plt.xticks(thresholds)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define your setup
    base_dir = r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\elbow_test\\"
    thresholds = [100, 200, 300, 350, 400, 500]
    conspir = "NATO War Provocation"  # Update this string to reflect the specific conspiracy being analyzed
    
    results = {}
    
    print("="*60)
    print("STARTING ELBOW CURVE VALIDATION PIPELINE")
    print("="*60)
    
    for t in thresholds:
        file_path = os.path.join(base_dir, f"{t}_conspir_d_55dim.csv")
        print(f"\nEvaluating Threshold: {t} characters...")
        
        try:
            df = pd.read_csv(file_path)
            df = df.set_index('ID')

            y = df['w2_conspir_d']
            y = 6 - y
            X = df.drop(columns=['w2_conspir_a','w2_conspir_b','w2_conspir_c','w2_conspir_d'])
            
            y = y.astype(int)
            X = X.apply(pd.to_numeric, errors='coerce')
            feature_names = X.columns.tolist()

            # Run CV silently (no plots) to collect metrics
            metrics = run_nested_cv(X, y, feature_names, conspir, verbose_plot=False)
            results[t] = metrics
            
            print(f"--> Threshold {t} Complete: Mean r = {metrics['mean_r']:.3f} ± {metrics['std_r']:.3f} (n={metrics['sample_size']})")
            
        except FileNotFoundError:
            print(f"File not found for threshold {t}: {file_path}. Skipping.")
    
    # Generate the final Elbow Curve
    if results:
        plot_elbow_curve(results, title=f"Elbow Curve for M1, Elastic Net: {conspir}")