import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
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
        ('imputer', SimpleImputer(strategy='median')), # Required!
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


    df_results = pd.DataFrame({
    'y_true': global_y_true,
    'y_pred_cont': global_y_pred_cont,
    'y_pred_disc': global_y_pred_disc
    })
    
    df_results.to_csv(conspir +"_M0.csv", index=False)

    global_accuracy = accuracy_score(global_y_true, global_y_pred_disc)
    global_mse = mean_squared_error(global_y_true, global_y_pred_cont)
    global_r2 = r2_score(global_y_true, global_y_pred_cont)
    pearson_corr, _ = pearsonr(global_y_true, global_y_pred_cont)
    spearman_corr, _ = spearmanr(global_y_true, global_y_pred_cont)
    global_qwk = cohen_kappa_score(global_y_true, global_y_pred_disc, labels=classes, weights='quadratic')

    # Aggregate Coefficients

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

    print("\n" + "="*55)
    print("TOP 15 MOST PROMINENT FEATURES")
    print("="*55)
    print(coef_df[['Feature', 'Mean_Beta', 'Std_Beta']].head(15).to_string(index=False))

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
    plt.suptitle(f"Method 0: {conspir}", fontsize=16, fontweight='bold')

    header_text = (
        f"Accuracy: {global_accuracy:.3f}  |  "
        f"QWK: {global_qwk:.3f}  |  "
        f"Spearman \u03c1: {spearman_corr:.3f}  |  "
        f"Pearson r: {pearson_corr:.3f}"
    )

    plt.title(header_text, pad=10, fontsize=10)
    plt.tight_layout()
    plt.show()


    # 1. Bring in our codebook mapping dictionary
    '''

    CODEBOOK_MAP = {
    'w3_resil_1': 'Resilience: Po prožití těžkých časů se obvykle rychle vzpamatuji',
    'w3_resil_2': 'Resilience: Je pro mě těžké procházet obtížnými situacemi',
    'w3_resil_3': 'Resilience: Netrvá mi dlouho se vzpamatovat po stresující události',
    'w3_resil_4': 'Resilience: Když se stane něco špatného, těžko se z toho dostávám',
    'w3_resil_5': 'Resilience: Obvykle procházím náročnými obdobími bez větších těžkostí',
    'w3_resil_6': 'Resilience: Trvá mi spíš delší dobu, než se dostanu přes nezdary ve svém životě',
    'w3_sat_dem': 'Spokojenost s fungováním demokracie',
    'w3_feltsec': 'Pocit bezpečí',
    'w3_sub_inc': 'Subjektivní hodnocení příjmu domácnosti',
    'w3_mig_att_1': 'Postoje k migrantům: Přistěhovalci jsou obecně pro českou ekonomiku přínosem',
    'w3_pol_trust_1': 'Důvěra pol. institucím: Prezident republiky',
    'w3_pol_interest': 'Zájem o politiku',
    'w3_pol_trust_2': 'Důvěra pol. institucím: Vláda',
    'w3_pol_trust_5': 'Důvěra pol. institucím: Ústavní soud',
    'w3_pol_trust_8': 'Důvěra pol. institucím: Veřejnoprávní média (např. ČT, ČRo)',
    'w3_pol_trust_9': 'Důvěra pol. institucím: Vědci a vědkyně',
    'w3_issue_sal_1': 'Důležitost tématu: stejnopohlavní manželství',
    'w3_issue_sal_2': 'Důležitost tématu: členství ČR v EU',
    'w3_issue_sal_3': 'Důležitost tématu: výše daňové sazby pro lidi s vyššími přijmy',
    'w3_issue_sal_4': 'Důležitost tématu: trvalý pobyt pro UA uprchlíky',
    'w3_issue_sal_6': 'Důležitost tématu: povinné očkování',
    'w3_anx_1': 'Deprese a úzkost: Pocit nervozity, úzkosti nebo napětí',
    'w3_anx_2': 'Deprese a úzkost: Nemožnost přestat se obávat nebo dostat obavy pod kontrolu',
    'w3_anx_3': 'Deprese a úzkost: Pocit, že jsem na dně, pocit deprese nebo beznaděje',
    'w3_anx_4': 'Deprese a úzkost: Malý zájem nebo potěšení z věcí, které dělám',
    'w3_fin_hshld': 'Egotropic retrospective: Během posledních šesti měsíců se finanční situace domácnosti respondenta...',
    'w3_econ_ret': 'Sociotropic retrospective: Během posledních šesti měsíců se stav ekonomiky v České republice...',
    'w3_threat_1': 'Zdroje ohrožení: přírodní katastrofa',
    'w3_threat_2': 'Zdroje ohrožení: vojenské napadení ČR jinou zemí',
    'w3_threat_3': 'Zdroje ohrožení: pandemie onemocnění',
    'w3_threat_4': 'Zdroje ohrožení: příchod velkého množství přistěhovalců do ČR',
    'w3_threat_5': 'Zdroje ohrožení: omezení práv a svobod',
    'w3_threat_6': 'Zdroje ohrožení: projevy levicového nebo pravicového extrémismu v ČR',
    'w3_gen_trust': 'Sociální důvěra',
    'w3_anger_1': 'Vztek nad: celkovou situací v ČR',
    'w3_left_right': 'Ideologie - sebezařazení na škále levice-pravice',
    'w3_fear_1': 'Obavy nad: celkovou situací v ČR',
    'w3_fear_3': 'Obavy nad: ekonomickými podmínkami v ČR',
    'w1_IDE_5b_Ubytování a stravování.': 'Odvětví aktuálního zaměstnání - Ubytování a stravování',
    'age_group_65+': 'Věková skupina: 65+',
    'w3_pol_trust_8_is_nevi': ' No answer: Trust in political institutions: Public media (e.g. Czech Television, Czech Radio)',
    'age_group_50-64': 'Věková skupina: 50-64',
    'w3_sat_dem_is_nevi': ' No answer: Satisfaction with the functioning of democracy',
    'w1_IDE_8_1.0': 'Gender: Female',
    'w1_IDE_19_4.0': 'Medium-sized town'
    }
    '''

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
    plt.title(f"Top 15 Predictors of Method 0: {conspir}, Elastic Net (Mean $\\beta$ ± 1 SD)", fontweight='bold', fontsize=13, pad=15)
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
    df = pd.read_csv(r"C:\filip_a_jeho_veci\Programovani\VYZKUMAK\data\df_design_d_demog.csv")
    df = df.set_index('ID')

    y = df['w2_conspir_d']
    X = df.drop(columns=['w2_conspir_d'])

    conspir = "nato"  # Update this string to reflect the specific conspiracy being analyzed


    y = y.astype(int)
    X = X.apply(pd.to_numeric, errors='coerce')

    feature_names = X.columns.tolist()

    run_nested_cv(X, y, feature_names)
    pass


