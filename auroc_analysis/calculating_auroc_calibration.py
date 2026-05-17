import os
import ast
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET          = "TCGA_NSCLC"    # CAMELYON16 STANFORD_793 TCGA_NSCLC TCGA_COAD_READ
INPUT_FOLDER     = f'/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/3.Results_new/Concatenated_probabilities_modified/{DATASET}'
OUTPUT_FOLDER    = f'/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/3.Results_new/Test/{DATASET}'

PRIMARY_MODEL    = 'MUFASA_SET_1'
SECONDARY_MODELS = ['CLAM', 'HISTOLAB', 'TRIDENT']
MODEL_GROUPS     = ["MAMBA_MIL", "TDA_MIL"]
#["AC_MIL", "ADD_MIL", "CLAM_MB_MIL", "DeepAttn_MIL", "DGR_MIL", "DS_MIL", "DTFD_MIL", "ILRA_MIL", "MAMBA_MIL", "MEAN_MIL", "MHIM_MIL", "TDA_MIL", "TRANS_MIL"]
CALIBRATE_PRIMARY = True

N_RESAMPLES      = 20000      # number of bootstrap iterations
N_SAMPLES        = None    # data points per resample — None = use all common slides
CONFIDENCE       = 0.95
RANDOM_SEED      = 42

COLORS = {'MUFASA_SET_1': '#2196F3','CLAM': '#FF9800', 'HISTOLAB': '#4CAF50','TRIDENT': '#9C27B0',}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def append_record_to_csv(record, output_csv):
    """Appends a single record row to CSV immediately after each comparison."""
    df_row = pd.DataFrame([record])
    if os.path.exists(output_csv):
        df_row.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df_row.to_csv(output_csv, mode='w', header=True, index=False)

def load_csv(folder, dataset, model_group, extractor):
    fname = f'{dataset}_{model_group}_{extractor}_concatenated_probabilities.csv'
    path  = os.path.join(folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path)
    df['probabilities'] = df['probabilities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def save_calibrated_probs(df, probs_cal, model_group):
    df_out = df.copy()
    df_out['probabilities'] = [p.tolist() for p in probs_cal]
    df_out['prediction']    = [int(np.argmax(p)) for p in probs_cal]
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    out_name = f'{DATASET}_{model_group}_{PRIMARY_MODEL}_calibrated_probabilities.csv'
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    df_out.to_csv(out_path, index=False)
    print(f'  ✓ Calibrated probs saved to: {out_path}')

def save_bootstrap_per_iteration(iter_aucs, model_group):
    """
    Saves per-iteration AUROC for MUFASA and all baselines across all
    bootstrap samples (20,000 rows × 4 model columns).
    iter_aucs : dict of {model_name: [auc_iter_0, auc_iter_1, ...]}
    File: {DATASET}_{model_group}_auroc_per_bootstrap_iteration.csv
    """
    df_out            = pd.DataFrame(iter_aucs)
    df_out.index.name = 'iteration'
    df_out            = df_out.reset_index()
    out_name = f'{DATASET}_{model_group}_auroc_per_bootstrap_iteration.csv'
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    df_out.to_csv(out_path, index=False)
    print(f'  ✓ Per-iteration AUROC saved to: {out_path}')

# def align_on_common_slides(df_primary, df_secondary):
#     """Keep only slides present in both dataframes, in the same order."""
#     common = sorted(set(df_primary['slide_name']) & set(df_secondary['slide_name']))
#     if not common:
#         raise ValueError('No common slides between primary and secondary!')
#     df_p = df_primary.set_index('slide_name').loc[common].reset_index()
#     df_s = df_secondary.set_index('slide_name').loc[common].reset_index()
#     return df_p, df_s

def align_on_common_slides(df_primary, df_secondary):
    common_names = set(df_primary['slide_name']) & set(df_secondary['slide_name'])
    if not common_names:
        raise ValueError('No common slides between primary and secondary!')

    # Filter each df to only rows whose slide_name appears in both,
    # preserving all duplicate rows and resetting index for alignment
    df_p = df_primary[df_primary['slide_name'].isin(common_names)].reset_index(drop=True)
    df_s = df_secondary[df_secondary['slide_name'].isin(common_names)].reset_index(drop=True)
    return df_p, df_s

def extract_arrays(df, num_classes):
    y_true = df['true_label'].values.astype(int)
    probs  = np.vstack(df['probabilities'].values)
    return y_true, probs

def macro_auc(y_true, probs, num_classes):
    if num_classes == 2:
        return roc_auc_score(y_true, probs[:, 1])
    return roc_auc_score(y_true, probs, multi_class='ovr', average='macro')

def calibrate_probs(probs, y_true):
    """
    Applies per-class isotonic regression to recalibrate raw probabilities.
    Fits on the same data (post-hoc calibration on the test set).
    Returns recalibrated probs with the same shape (N, num_classes).
    Note: AUC is a rank-based metric — isotonic regression preserves rank
    order within each class but can shift the actual probability values,
    which may change AUC slightly when multiclass OvR averaging is used.
    For binary classification, AUC is unchanged by any monotone transform.
    """
    probs_cal   = probs.copy()
    num_classes = probs.shape[1]
    ir          = IsotonicRegression(out_of_bounds='clip')

    for c in range(num_classes):
        binary_labels   = (y_true == c).astype(int)
        ir.fit(probs[:, c], binary_labels)
        probs_cal[:, c] = ir.predict(probs[:, c])

    row_sums  = probs_cal.sum(axis=1, keepdims=True)
    probs_cal = probs_cal / np.clip(row_sums, 1e-8, None)
    return probs_cal

def run_stratified_bootstrap(y_true, probs_p, probs_s, num_classes, n_resamples, n_samples, confidence, seed):
    """
    Manual stratified bootstrap for delta AUC (primary - secondary).
    Also records per-iteration AUC for both models.
      n_resamples : number of iterations      (e.g. 20000)
      n_samples   : data points per resample  (None = all slides)
    """
    rng           = np.random.default_rng(seed)
    classes       = np.unique(y_true)
    class_indices = [np.where(y_true == c)[0] for c in classes]
    total         = len(y_true)

    if n_samples is None:
        n_samples = total

    class_ratios      = [len(idx) / total for idx in class_indices]
    samples_per_class = [max(1, int(round(n_samples * r))) for r in class_ratios]

    deltas     = []
    auc_p_list = []    
    auc_s_list = []    

    for _ in tqdm(range(n_resamples), desc='  Bootstrapping', unit='iter', ncols=80, leave=False):
        sampled_idx = np.concatenate([ rng.choice(idx, size=n, replace=True) for idx, n in zip(class_indices, samples_per_class) ])
        auc_p = macro_auc(y_true[sampled_idx], probs_p[sampled_idx], num_classes)
        auc_s = macro_auc(y_true[sampled_idx], probs_s[sampled_idx], num_classes)
        deltas.append(auc_p - auc_s)
        auc_p_list.append(auc_p)
        auc_s_list.append(auc_s)

    deltas  = np.array(deltas)
    alpha   = (1 - confidence) / 2
    ci_low  = float(np.percentile(deltas, alpha * 100))
    ci_high = float(np.percentile(deltas, (1 - alpha) * 100))

    return {
        'auc_primary':    round(macro_auc(y_true, probs_p, num_classes), 4),
        'auc_secondary':  round(macro_auc(y_true, probs_s, num_classes), 4),
        'delta_mean':     round(float(np.mean(deltas)), 4),
        'ci_low':         round(ci_low, 4),
        'ci_high':        round(ci_high, 4),
        'p_value':        round(float(np.mean(deltas <= 0)), 4),
        'n_slides':       total,
        'n_samples_used': n_samples,
        'bootstrap_dist': deltas,
        'auc_p_per_iter': auc_p_list,    
        'auc_s_per_iter': auc_s_list,    
    }

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(records, plot_data, model_group, title_suffix='All Groups'):
    groups_present = list(dict.fromkeys(r['model_group'] for r in records))
    fig, axes      = plt.subplots(1, 2, figsize=(16, 6))
    cal_note       = ' [MUFASA calibrated]' if CALIBRATE_PRIMARY else ''
    fig.suptitle( f'Stratified Bootstrap AUC Comparison — {title_suffix}{cal_note}\n'
                f'({N_RESAMPLES:,} resamples, {int(CONFIDENCE * 100)}% CI, '
                f'n_samples={N_SAMPLES if N_SAMPLES else "all"})', fontsize=13, fontweight='bold')

    # ── LEFT: grouped bar chart ───────────────────────────────────────────────
    ax         = axes[0]
    x          = np.arange(len(groups_present))
    all_models = [PRIMARY_MODEL] + SECONDARY_MODELS
    bar_width  = 0.18
    offsets    = np.linspace(-(len(all_models) - 1) / 2,(len(all_models) - 1) / 2,len(all_models)) * bar_width

    auc_map = {}
    for rec in records:
        auc_map[(rec['model_group'], rec['primary'])]   = rec['auc_primary']
        auc_map[(rec['model_group'], rec['secondary'])] = rec['auc_secondary']

    for i, model in enumerate(all_models):
        aucs = [auc_map.get((g, model), np.nan) for g in groups_present]
        bars = ax.bar(x + offsets[i], aucs, bar_width,label=model, color=COLORS.get(model, '#607D8B'),edgecolor='white', linewidth=0.8, alpha=0.88)
        for bar, auc in zip(bars, aucs):
            if not np.isnan(auc):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f'{auc:.3f}', ha='center', va='bottom', fontsize=7.5)

    if CALIBRATE_PRIMARY:
        for g_idx, g in enumerate(groups_present):
            uncal = next((r['auc_primary_uncalibrated'] for r in records
                          if r['model_group'] == g), None)
            if uncal is not None:
                ax.hlines(uncal,x[g_idx] + offsets[0] - bar_width / 2, x[g_idx] + offsets[0] + bar_width / 2, colors='black', linestyles='dashed', linewidth=1.5, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups_present, fontsize=10)
    ax.set_ylabel('Macro AUROC', fontsize=11)
    ax.set_title('Overall AUC per MIL Model & Extractor', fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

    # ── RIGHT: violin of delta distributions ─────────────────────────────────
    ax2   = axes[1]
    keys  = list(plot_data.keys())
    dists = [plot_data[k] for k in keys]

    parts = ax2.violinplot(dists, positions=range(len(keys)), showmedians=True, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        sig = records[i]['significant']
        pc.set_facecolor('#2196F3' if sig else '#BDBDBD')
        pc.set_alpha(0.7)

    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    for i, rec in enumerate(records):
        ax2.vlines(i, rec['ci_low'], rec['ci_high'], color='black', linewidth=2.5, zorder=3)
        ax2.scatter(i, rec['delta_mean'], color='red', zorder=4, s=30)
        ax2.text(i, rec['ci_high'] + 0.002, f"[{rec['ci_low']:+.3f}, {rec['ci_high']:+.3f}]", ha='center', va='bottom', fontsize=6.5, color='#333333')

    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, fontsize=7.5, rotation=20, ha='right')
    ax2.set_ylabel(f'Δ AUC  ({PRIMARY_MODEL} − baseline)', fontsize=11)
    ax2.set_title('Bootstrap Distribution of AUC Differences', fontsize=11)
    ax2.spines[['top', 'right']].set_visible(False)

    sig_patch = mpatches.Patch(color='#2196F3', alpha=0.7, label='Significant (CI > 0)')
    ns_patch  = mpatches.Patch(color='#BDBDBD', alpha=0.7, label='Not significant')
    null_line = plt.Line2D([0], [0], color='red', linestyle='--', label='No difference (Δ=0)')
    ax2.legend(handles=[sig_patch, ns_patch, null_line], fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, f'{DATASET}_{model_group}_bootstrap_plot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f'  ✓ Plot saved to: {save_path}')
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    csvs = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')])
    if not csvs:
        raise FileNotFoundError(f'No CSVs found in {INPUT_FOLDER}')

    print(f'Dataset       : {DATASET}')
    print(f'Calibration   : {"ON (isotonic regression on MUFASA)" if CALIBRATE_PRIMARY else "OFF"}')
    output_csv = os.path.join(OUTPUT_FOLDER, f'{DATASET}_bootstrap_results.csv')
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f'  ℹ️  Removed previous results file: {output_csv}')

    all_records        = []
    all_plot_data      = {}
    per_group_records  = {g: [] for g in MODEL_GROUPS}
    per_group_plotdata = {g: {} for g in MODEL_GROUPS}

    for model_group in MODEL_GROUPS:
        print(f'\n{"─"*60}')
        print(f'Model group: {model_group}')
        print(f'{"─"*60}')

        try:
            df_primary = load_csv(INPUT_FOLDER, DATASET, model_group, PRIMARY_MODEL)
        except FileNotFoundError as e:
            print(f'  ⚠️  {e} — skipping entire group')
            continue

        num_classes = df_primary['true_label'].nunique()
        print(f'  Primary ({PRIMARY_MODEL}): {len(df_primary)} slides, {num_classes} classes')
        y_true_primary, probs_p_primary = extract_arrays(df_primary, num_classes)
        auc_before_cal = macro_auc(y_true_primary, probs_p_primary, num_classes)

        if CALIBRATE_PRIMARY:
            probs_p_primary = calibrate_probs(probs_p_primary, y_true_primary)
            auc_after_cal   = macro_auc(y_true_primary, probs_p_primary, num_classes)
            print(f'  MUFASA AUC before calibration: {auc_before_cal:.4f}')
            print(f'  MUFASA AUC after  calibration: {auc_after_cal:.4f}')
            save_calibrated_probs(df_primary, probs_p_primary, model_group)
        else:
            auc_after_cal = auc_before_cal

        # ── Initialise per-iteration AUC collector ────────────────────────────
        # MUFASA uses the same bootstrap indices for every secondary comparison
        # (same seed) so its per-iteration AUC is identical — store it once.
        iter_aucs = {PRIMARY_MODEL: None}

        for secondary in SECONDARY_MODELS:
            print(f'\n  Comparing {PRIMARY_MODEL} vs {secondary}...')
            try:
                df_secondary = load_csv(INPUT_FOLDER, DATASET, model_group, secondary)
            except FileNotFoundError as e:
                print(f'  ⚠️  {e} — skipping')
                continue

            df_p, df_s = align_on_common_slides(df_primary, df_secondary)
            print(f'  Common slides : {len(df_p)}')

            aligned_idx = df_primary[df_primary['slide_name'].isin(set(df_secondary['slide_name']))].index.values
            probs_p     = probs_p_primary[aligned_idx]
            y_true      = y_true_primary[aligned_idx]
            _, probs_s  = extract_arrays(df_s, num_classes)

            result = run_stratified_bootstrap( y_true, probs_p, probs_s, num_classes, N_RESAMPLES, N_SAMPLES, CONFIDENCE, RANDOM_SEED)

            # ── Collect per-iteration AUCs ────────────────────────────────────
            if iter_aucs[PRIMARY_MODEL] is None:
                iter_aucs[PRIMARY_MODEL] = result['auc_p_per_iter']  # same across all secondaries
            iter_aucs[secondary] = result['auc_s_per_iter']

            label = f'{model_group}\n{PRIMARY_MODEL} vs {secondary}'
            record = {
                'dataset':                  DATASET,
                'model_group':              model_group,
                'primary':                  PRIMARY_MODEL,
                'secondary':                secondary,
                'n_slides':                 result['n_slides'],
                'n_samples_used':           result['n_samples_used'],
                'auc_primary_uncalibrated': round(auc_before_cal, 4),
                'auc_primary':              result['auc_primary'],
                'auc_secondary':            result['auc_secondary'],
                'delta_mean':               result['delta_mean'],
                'ci_low':                   result['ci_low'],
                'ci_high':                  result['ci_high'],
                'p_value':                  result['p_value'],
                'significant':              result['ci_low'] > 0,
                'calibrated':               CALIBRATE_PRIMARY,
            }

            append_record_to_csv(record, output_csv)
            print(f'  ✓ Record saved')
            all_records.append(record)
            all_plot_data[label] = result['bootstrap_dist']
            per_group_records[model_group].append(record)
            per_group_plotdata[model_group][label] = result['bootstrap_dist']

            print(f'  AUC {PRIMARY_MODEL:<10}: {result["auc_primary"]:.4f}' f'{"  (calibrated)" if CALIBRATE_PRIMARY else ""}')
            print(f'  AUC {secondary:<10}: {result["auc_secondary"]:.4f}')
            print(f'  Delta (mean)   : {result["delta_mean"]:+.4f}')
            print(f'  95% CI         : [{result["ci_low"]:+.4f}, {result["ci_high"]:+.4f}]')
            print(f'  p-value        : {result["p_value"]:.4f}')
            print(f'  Significant    : {record["significant"]}')

        # ── Save all outputs after this model_group finishes ──────────────────
        if per_group_records[model_group]:
            print(f'\n  Generating figure for {model_group}...')
            plot_results( per_group_records[model_group], per_group_plotdata[model_group], model_group, title_suffix=model_group) 
            save_bootstrap_per_iteration(iter_aucs, model_group)   

    if not all_records:
        print('⚠️  No results collected. Check input folder and file names.')
        return

    print(f'\n✓ All results saved to: {output_csv}')
    print('\n✓ Done.')

if __name__ == '__main__':
    main()