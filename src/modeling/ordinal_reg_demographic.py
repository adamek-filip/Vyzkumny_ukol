import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import contextlib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import textwrap

# Suppress L-BFGS-B warnings for cleaner output during heavy CV loops
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Custom context manager to integrate tqdm with joblib for inner loop progress tracking
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into a tqdm progress bar.
    This intercepts the parallel processing engine Scikit-Learn uses under the hood.
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
# 1. The Core Model (From previous logic)
# ==========================================
class ElasticNetOrdinalRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if K <= 2:
            raise ValueError("Response must have 3 or more levels.")
            
        y_encoded = LabelEncoder().fit_transform(y)
        N, P = X.shape
        
        cum_props = np.cumsum(np.bincount(y_encoded)) / N
        cum_props = np.clip(cum_props, 1e-6, 1 - 1e-6)
        initial_zetas = logit(cum_props[:-1])
        
        zeta_1 = initial_zetas[0]
        delta_zetas = np.diff(initial_zetas)
        delta_zetas[delta_zetas <= 0] = 1e-2 

        w0 = np.concatenate([np.zeros(P), np.zeros(P), [zeta_1], delta_zetas])
        bounds = [(0, None)] * (2 * P) + [(None, None)] + [(1e-5, None)] * (K - 2)

        res = minimize(
            fun=self._objective, x0=w0, args=(X, y_encoded, K, N, P),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        w_opt = res.x
        self.coef_ = w_opt[0:P] - w_opt[P:2*P]
        self.zetas_ = np.zeros(K - 1)
        self.zetas_[0] = w_opt[2*P]
        for k in range(1, K - 1):
            self.zetas_[k] = self.zetas_[k-1] + w_opt[2*P+1+k-1]
            
        return self

    def _objective(self, w, X, y_encoded, K, N, P):
        beta_plus = w[0:P]
        beta_minus = w[P:2*P]
        beta = beta_plus - beta_minus
        
        zetas = np.zeros(K - 1)
        zetas[0] = w[2*P]
        for k in range(1, K - 1):
            zetas[k] = zetas[k-1] + w[2*P+1+k-1]
            
        zetas_padded = np.concatenate([[-np.inf], zetas, [np.inf]])
        eta = np.dot(X, beta)
        
        upper_zeta = zetas_padded[y_encoded + 1]
        lower_zeta = zetas_padded[y_encoded]
        
        probs = np.clip(expit(upper_zeta - eta) - expit(lower_zeta - eta), 1e-15, 1.0)
        nll = -np.sum(np.log(probs)) / N
        
        l1_penalty = self.alpha * self.l1_ratio * np.sum(beta_plus + beta_minus)
        l2_penalty = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(beta**2)
        
        return nll + l1_penalty + l2_penalty

    def predict_proba(self, X):
        X = np.asarray(X)
        eta = np.dot(X, self.coef_)
        zetas_padded = np.concatenate([[-np.inf], self.zetas_, [np.inf]])
        K = len(self.classes_)
        probs = np.zeros((X.shape[0], K))
        for k in range(K):
            probs[:, k] = expit(zetas_padded[k+1] - eta) - expit(zetas_padded[k] - eta)
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# ==========================================
# 3. Nested 10x10 Cross-Validation Setup
# ==========================================
def run_nested_cv(X, y, feature_names):
    print("Starting 10x10 Stratified Nested Cross-Validation...")
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    classes = np.sort(np.unique(y))

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    param_grid = {
        'model__alpha': np.logspace(-4, 2, 10),
        'model__l1_ratio': np.linspace(0.01, 1.0, 10) 
    }
    
    global_y_true = []
    global_y_pred_disc = []
    fold_coefs = []
    overall_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
    
    # We convert the generator to a list so tqdm knows exactly how many total iterations there are for the ETA
    outer_splits = list(outer_cv.split(X, y))
    
    # Initialize the tqdm progress bar for the outer loop
    with tqdm(total=len(outer_splits), desc="Outer CV Progress", unit="fold") as pbar:
        
        for fold, (train_ix, test_ix) in enumerate(outer_splits, 1):
            X_train, X_test = X[train_ix], X[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]
            
            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            
            # Create the pipeline
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # Handles the np.nan values
                ('scaler', StandardScaler()),                # Centers and scales the features
                ('model', ElasticNetOrdinalRegression())     # Your custom model
            ])
            
            # --- THE INNER LOOP TRACKING ---
            search = GridSearchCV(
                pipeline,  
                param_grid, 
                cv=inner_cv, 
                scoring='neg_log_loss', 
                n_jobs=-1, 
                verbose=0  
            )

            # 2. Calculate the exact number of fits for the inner progress bar
            # (Updated to use the new dictionary keys)
            n_candidates = len(param_grid['model__alpha']) * len(param_grid['model__l1_ratio'])
            total_fits = n_candidates * inner_cv.get_n_splits()
            
            # 3. Execute the fit inside our custom context manager
            # We use leave=False so the inner bar deletes itself upon completion, 
            # keeping the terminal clean.
            with tqdm_joblib(tqdm(desc=f"Running fits for Outer {fold}/10", total=total_fits, leave=False, unit="fit")):
                search.fit(X_train, y_train)

            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            fold_coefs.append(best_model.named_steps['model'].coef_)

            
            global_y_true.extend(y_test)
            global_y_pred_disc.extend(y_pred)
            
            cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
            overall_cm += cm

            fold_acc = accuracy_score(y_test, y_pred)
            
            # --- DYNAMIC TERMINAL UPDATES ---
            # Instead of a standard print statement, we use tqdm.write so it doesn't break the progress bar visual
            tqdm.write(f"\nCompleted Outer Fold {fold}/10 | Best Params: {search.best_params_} | Acc: {fold_acc:.4f}\n")
            
        
    # Calculate global metrics strictly once outside the loop
    global_y_true = np.array(global_y_true)
    global_y_pred_disc = np.array(global_y_pred_disc)
    
    global_accuracy = accuracy_score(global_y_true, global_y_pred_disc)
    global_qwk = cohen_kappa_score(global_y_true, global_y_pred_disc, labels=classes, weights='quadratic')
    global_pearson, _ = pearsonr(global_y_true, global_y_pred_disc)
    global_spearman, _ = spearmanr(global_y_true, global_y_pred_disc)

    results = {
        "Accuracy": global_accuracy,
        "Quadratic Weighted Kappa": global_qwk,
        "Pearson r": global_pearson,
        "Spearman rho": global_spearman
    }
    
    # Calculate the mean and std of coefficients across all 10 outer folds
    fold_coefs = np.array(fold_coefs)  # Shape: (10, P)
    mean_coefs = np.mean(fold_coefs, axis=0)
    std_coefs = np.std(fold_coefs, axis=0)
    
    # Create a DataFrame and sort by the absolute magnitude of the mean coefficient
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Beta': mean_coefs,
        'Std_Beta': std_coefs,
        'Abs_Mean_Beta': np.abs(mean_coefs)
    })
    
    # Sort descending by absolute effect size
    coef_df = coef_df.sort_values(by='Abs_Mean_Beta', ascending=False).reset_index(drop=True)
    
    return results, overall_cm, classes, coef_df 

# ==========================================
# 4. Execution & Visualization
# ==========================================
if __name__ == "__main__":  
    df = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\df_design_d_demog.csv")
    df = df.set_index('ID')

    y = df['w2_conspir_d']
    y = 6 - y
    X = df.drop(columns=['w2_conspir_d'])
    X['w3_resil_2'] = 6 - X['w3_resil_2']  # Reverse code this specific resilience item
    X['w3_resil_4'] = 6 - X['w3_resil_4']  # Reverse code this specific resilience item
    X['w3_resil_6'] = 6 - X['w3_resil_6']  # Reverse code this specific resilience item
    
    conspir = "NATO War Provocation"  # Update this string to reflect the specific conspiracy being analyzed

    y = y.astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')

    # SAVE FEATURE NAMES HERE
    feature_names = X.columns.tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    results, overall_cm, classes, coef_df = run_nested_cv(X, y, feature_names)
    
    # 2. Print Aggregated Results
    print("\n" + "="*40)
    print("FINAL AGGREGATED RESULTS")
    print("="*40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    string_classes = [str(c) for c in classes]
    # 3. Plot Overall Confusion Matrix
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        overall_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=string_classes, 
        yticklabels=string_classes
    )
    plt.xlabel('Predicted Value', fontweight='bold')
    plt.ylabel('True Value', fontweight='bold')
    plt.suptitle(conspir, fontsize=16, fontweight='bold')

    header_text = (
        f"Accuracy: {results['Accuracy']:.3f}  |  "
        f"QWK: {results['Quadratic Weighted Kappa']:.3f}  |  "
        f"Spearman \u03c1: {results['Spearman rho']:.3f}  |  "
        f"Pearson r: {results['Pearson r']:.3f}"
    )

    plt.title(header_text, pad=10, fontsize=10)
    plt.tight_layout()
    plt.show()


# 1. Bring in our codebook mapping dictionary
    CODEBOOK_MAP = {
    'w3_resil_1': 'Resilience: I usually bounce back quickly after going through tough times.',
    'w3_resil_2': 'Resilience: It is difficult for me to go through difficult situations.',
    'w3_resil_3': "Resilience: It doesn't take me long to recover from a stressful event.",
    'w3_resil_4': 'Resilience: When something bad happens, I have a hard time getting over it.',
    'w3_resil_5': 'Resilience: I usually get through difficult times without much difficulty.',
    'w3_resil_6': 'Resilience: It takes me longer to get over setbacks in my life',
    'w3_sat_dem': 'Satisfaction with the functioning of democracy',
    'w3_ref_att_1': 'Attitudes towards Ukrainian refugees: values in the Czech Republic',
    'w3_feltsec': 'Feeling of safety',
    'w3_sub_inc': 'Subjective evaluation of household income',
    'w3_mig_att_1': 'Attitudes towards migrants: Immigrants are generally a benefit to the Czech economy',
    'w3_pol_trust_1': 'Trust in political institutions: President of the Republic',
    'w3_pol_interest': 'Interest in politics',
    'w3_pol_trust_2': 'Trust in political institutions: Government',
    'w3_pol_trust_5': 'Trust in political institutions: Constitutional Court',
    'w3_pol_trust_8': 'Trust in political institutions: Public media (e.g. Czech Television, Czech Radio)',
    'w3_pol_trust_9': 'Trust in political institutions: Scientists',
    'w3_issue_sal_1': 'Importance of the topic: same-sex marriage',
    'w3_issue_sal_2': "Importance of the topic: Czech Republic's membership in the EU",
    'w3_issue_sal_3': 'Importance of the topic: the level of tax rates for people with higher incomes',
    'w3_issue_sal_4': 'Importance of the topic: permanent residence for UA refugees',
    'w3_issue_sal_6': 'Importance of the topic: mandatory vaccination',
    'w3_anx_1': 'Depression and anxiety: Feeling nervous, anxious, or tense',
    'w3_anx_2': 'Depression and Anxiety: Inability to stop worrying or get worries under control',
    'w3_anx_3': 'Depression and anxiety: Feeling down, depressed, or hopeless',
    'w3_anx_4': 'Depression and Anxiety: Little interest or pleasure in the things I do',
    'w3_fin_hshld': "Egotropic retrospective: Over the past six months, the financial situation of the respondent's household...",
    'w3_econ_ret': 'Sociotropic retrospective: Over the past six months, the state of the economy in the Czech Republic...',
    'w3_threat_1': 'Sources of threat: Natural disaster',
    'w3_threat_2': 'Sources of threat: Military attack on the Czech Republic',
    'w3_threat_3': 'Sources of threat: Disease pandemic',
    'w3_threat_4': 'Sources of threat: The arrival of large numbers of immigrants',
    'w3_threat_5': 'Sources of threat: Restrictions on rights and freedoms',
    'w3_threat_6': 'Sources of threat: Manifestations of extremism in the Czech Republic',
    'w3_gen_trust': 'Social trust',
    'w3_anger_1': 'Anger over the overall situation in the Czech Republic',
    'w3_left_right': 'Ideology - self-classification on the left-right scale',
    'w3_fear_1': 'Concerns about the overall situation in the Czech Republic',
    'w3_fear_3': 'Concerns over economic conditions',
    'w1_IDE_5b_Ubytování a stravování.': 'Sector of current employment - Accommodation and food services',
    'w1_IDE_5b_Zemědělství, myslivost, lesní hospodářství.': 'Sector of current employment - Agriculture, hunting, forestry',
    'age_group_65+': 'Age group: 65+',
    'w3_pol_trust_8_is_nevi': ' No answer: Trust in political institutions: Public media (e.g. Czech Television, Czech Radio)',
    'age_group_50-64': 'Age group: 50-64',
    'w3_sat_dem_is_nevi': ' No answer: Satisfaction with the functioning of democracy',
    'w1_IDE_8_1.0': 'Gender: Female',
    'w1_IDE_19_4.0': 'Medium-sized town'
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
plt.title(f"Top 15 Predictors {conspir} (Mean $\\beta$ ± 1 SD), Ordinal regression", fontweight='bold', fontsize=13, pad=15)
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