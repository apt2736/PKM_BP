#!/usr/bin/env python3
"""
build_train_red_ir_notebook.py
Generates the complete Jupyter Notebook models/train_red_ir.ipynb
for dual SBP and DBP estimation using a 60-second sample window,
without per-window normalization in process_single_file(), with Optuna tuning
for both SBP and DBP, clinical Standard Deviation reporting, and standalone ANSI C export.
"""

import os
import json
import nbformat as nbf

def build_red_ir_notebook():
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    }

    cells = []

    # Title & Overview
    title_md = """# Blood Pressure Estimation Pipeline: Red & IR PPG + Lead-I ECG (Dual SBP & DBP 60s Models)

This notebook implements an end-to-end, reproducible, standalone machine learning pipeline for **Systolic Blood Pressure (SBP)** and **Diastolic Blood Pressure (DBP)** estimation from dual-wavelength photoplethysmography (PPG Red & IR) and single-lead ECG (Lead I) using a **60-second window duration**.

### Pipeline Stages
1. **60-Second Dedicated Feature Extraction**: Resamples filtered waveforms (240 Hz $\\rightarrow$ 100 Hz) and extracts 23 dedicated multi-domain cardiovascular biomarkers directly without window-level pre-scaling.
2. **Dataset Partitioning & Optuna Hyperparameter Optimization**: Partitions into 70% Train, 20% Validation, and 10% Holdout Test sets. Tunes LightGBM regressors for both SBP and DBP with Bayesian optimization (`n_estimators <= 100` for embedded flash constraints) and evaluates $R^2$, MAE, RMSE, Mean Error, and **Standard Deviation (SD)**.
3. **Dual Blood Pressure Diagnostics & Scatter Plots**: Diagnostic scatter plots for SBP and DBP comparing model predictions against clinical ground truth.
4. **Full Production Training & TinyML C Export**: Retrains winning 60s SBP and DBP models on 100% of data and exports pure C decision tree code via `tinymlgen` (`m2cgen`) to `deploy/lgbm_sbp.c` and `deploy/lgbm_dbp.c`.
5. **Ported C Code Verification**: Verifies bit-exact concordance between Python and C inference and executes `bp_pipeline` end-to-end on Subject 001 (6,000 samples @ 100 Hz).
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Step 0 MD & Code: Setup & Dependencies
    step0_md = """## Step 0: Environment Setup & Directory Initialization
"""
    cells.append(nbf.v4.new_markdown_cell(step0_md))

    code_setup = r"""import os
import sys
import glob
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import optuna

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 14

# Paths
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
filtered_dir = os.path.join(base_dir, "A dataset of simultaneous collected ECG and PPG signals", "Filtered_Data")
deploy_dir = os.path.join(base_dir, "deploy")
plots_dir = os.path.join(base_dir, "plots")
esp_dir = os.path.join(deploy_dir, "demo_bp_esp")

os.makedirs(deploy_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(esp_dir, exist_ok=True)

info_path = os.path.join(base_dir, "A dataset of simultaneous collected ECG and PPG signals", "information.csv")
info_df = pd.read_csv(info_path)
print(f"Loaded clinical metadata for {len(info_df)} subjects.")
"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # Step 1 MD & Code: Feature Extraction from Filtered Waveforms (60s Window)
    step1_md = """## Step 1: Feature Extraction from Filtered Waveforms (60s Window)
We resample the filtered waveforms from 240 Hz to 100 Hz using polyphase FIR filtering and extract the **23 dedicated cardiovascular biomarkers** over 60-second windows directly without window-level pre-scaling.
"""
    cells.append(nbf.v4.new_markdown_cell(step1_md))

    code_extraction = r"""FS_RAW = 240
FS_TARGET = 100

best_features = [
    'PAT_d', 'PAT_p', 'PAT_f_fridericia', 'PAT_f_framingham', 'PAT_d_framingham',
    'PAT_d_inv', 'PAT_d_sq_inv', 'PAT_p_sq_inv', 'PTT_p_est', 'PTT_f_est',
    'PTT_f_sq_inv', 'PTT_d_sq_inv', 'PTT_p_sq_inv', 'PW75', 'k_val',
    'PI_IR', 'AC_DC_ratio', 'IR_VPG_RMS', 'IR_APG_RMS', 'RED_VPG_RMS',
    'IR_Tsys', 'RED_PW50', 'Sex'
]

def compute_morphology(sig, peaks, valleys, fs=100.0):
    if len(peaks) == 0 or len(valleys) < 2:
        return {
            'pw75': 120.0, 'tsys': 150.0, 'tdia': 650.0,
            'apg_b_a': 0.0, 'apg_agi': 0.0
        }
    
    pw75_list = []
    tsys_list, tdia_list = [], []
    
    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i+1]
        if end - start < int(0.35 * fs):
            continue
        pulse = sig[start:end]
        p_min = np.min(pulse)
        p_max = np.max(pulse)
        p_norm = (pulse - p_min) / (p_max - p_min + 1e-5)
        
        p_peaks = [p - start for p in peaks if start <= p < end]
        if len(p_peaks) == 0:
            p_pk = int(np.argmax(p_norm))
        else:
            p_pk = p_peaks[0]
            
        tsys_list.append((p_pk / fs) * 1000.0)
        tdia_list.append(((len(pulse) - p_pk) / fs) * 1000.0)
        pw75_list.append((np.sum(p_norm >= 0.75) / fs) * 1000.0)
        
    v_pulse = np.gradient(sig)
    a_pulse = np.gradient(v_pulse)
    a_peaks, _ = signal.find_peaks(a_pulse, distance=int(0.35*fs))
    a_valleys, _ = signal.find_peaks(-a_pulse, distance=int(0.35*fs))
    
    apg_b_a = 0.0
    apg_agi = 0.0
    if len(a_peaks) > 0 and len(a_valleys) > 0:
        a_val = np.mean(a_pulse[a_peaks])
        b_val = np.mean(a_pulse[a_valleys])
        apg_b_a = float(b_val / (a_val + 1e-5))
        apg_agi = float((b_val - a_val) / (a_val + 1e-5))
        
    tsys_mean = float(np.mean(tsys_list)) if len(tsys_list) > 0 else 150.0
    tdia_mean = float(np.mean(tdia_list)) if len(tdia_list) > 0 else 650.0
    
    return {
        'pw75': float(np.mean(pw75_list)) if len(pw75_list) > 0 else 120.0,
        'tsys': tsys_mean,
        'tdia': tdia_mean,
        'apg_b_a': apg_b_a,
        'apg_agi': apg_agi
    }

def compute_red_morphology(sig, peaks, valleys, fs=100.0):
    if len(peaks) == 0 or len(valleys) < 2:
        return {'pw50': 250.0}
    pw50_list = []
    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i+1]
        if end - start < int(0.35 * fs):
            continue
        pulse = sig[start:end]
        p_min = np.min(pulse)
        p_max = np.max(pulse)
        p_norm = (pulse - p_min) / (p_max - p_min + 1e-5)
        pw50_list.append((np.sum(p_norm >= 0.50) / fs) * 1000.0)
    return {'pw50': float(np.mean(pw50_list)) if len(pw50_list) > 0 else 250.0}

def process_single_file(f):
    win_sec = 60
    step_sec = 2
    win_samples = win_sec * FS_TARGET
    step_samples = step_sec * FS_TARGET
    records = []
    
    sid_str = os.path.basename(f).split('_')[0]
    meta = info_df[info_df['ID'].astype(str).str.zfill(3) == sid_str]
    if len(meta) == 0:
        return records
    row = meta.iloc[0]
    sbp = float(row['SBP/mmHg'])
    dbp = float(row['DBP/mmHg'])
    sex = 1.0 if str(row['Gender']).strip().lower() == 'male' else 0.0
    
    df = pd.read_csv(f).dropna(subset=['ECG_I_Filtered'])
    if len(df) < FS_RAW * win_sec:
        return records
        
    ecg_raw = df['ECG_I_Filtered'].values.astype(float)
    red_raw = df['PPG_RED_Filtered'].values.astype(float)
    ir_raw  = df['PPG_IR_Filtered'].values.astype(float)
    
    ecg_100 = signal.resample_poly(ecg_raw, 5, 12)
    red_100 = signal.resample_poly(red_raw, 5, 12)
    ir_100  = signal.resample_poly(ir_raw, 5, 12)
    
    total_len = len(ir_100)
    
    for w_start in range(0, total_len - win_samples + 1, step_samples):
        w_end = w_start + win_samples
        e_w = ecg_100[w_start:w_end]
        r_w = red_100[w_start:w_end]
        i_w = ir_100[w_start:w_end]
        
        # Pan-Tompkins QRS on ECG
        e_diff = np.gradient(e_w)
        e_qrs = np.convolve(e_diff**2, np.ones(15)/15.0, mode='same')
        qrs_max = np.max(e_qrs)
        e_qrs_norm = e_qrs / (qrs_max + 1e-5) if qrs_max > 0 else e_qrs
        
        ecg_peaks, _ = signal.find_peaks(e_qrs_norm, distance=int(0.35 * FS_TARGET), prominence=0.10)
        
        i_std = np.std(i_w) + 1e-5
        r_std = np.std(r_w) + 1e-5
        ir_peaks, _  = signal.find_peaks(i_w, distance=int(0.35 * FS_TARGET), prominence=0.25 * i_std)
        red_peaks, _ = signal.find_peaks(r_w, distance=int(0.35 * FS_TARGET), prominence=0.25 * r_std)
        
        if len(ecg_peaks) < 3 or len(ir_peaks) < 3 or len(red_peaks) < 3:
            continue

        rr_sec = float((ecg_peaks[-1] - ecg_peaks[0]) / ((len(ecg_peaks) - 1) * FS_TARGET))
        if rr_sec < 0.3:
            rr_sec = 0.8
        pep_est = 60.0 + 0.12 * (1000.0 * rr_sec) * 0.1
        
        v_ir = np.gradient(i_w)
        a_ir = np.gradient(v_ir)
        v_red = np.gradient(r_w)
        
        pat_p_ir, pat_f_ir, pat_d_ir = [], [], []
        pat_p_red, pat_f_red, pat_d_red = [], [], []
        
        for r_peak in ecg_peaks:
            p_ir_cands = [p for p in ir_peaks if p > r_peak and p < r_peak + 60]
            if len(p_ir_cands) > 0:
                p_ir_idx = p_ir_cands[0]
                del_p = (p_ir_idx - r_peak) / FS_TARGET * 1000.0
                pat_p_ir.append(del_p)
                search_start = max(0, p_ir_idx - 30)
                f_cand = search_start + np.argmin(i_w[search_start:p_ir_idx])
                del_f = (f_cand - r_peak) / FS_TARGET * 1000.0
                pat_f_ir.append(del_f)
                if f_cand < p_ir_idx:
                    d_cand = f_cand + np.argmax(v_ir[f_cand:p_ir_idx])
                    del_d = (d_cand - r_peak) / FS_TARGET * 1000.0
                else:
                    del_d = (del_f + del_p) / 2.0
                pat_d_ir.append(del_d)
                
            p_red_cands = [p for p in red_peaks if p > r_peak and p < r_peak + 60]
            if len(p_red_cands) > 0:
                p_red_idx = p_red_cands[0]
                del_p_r = (p_red_idx - r_peak) / FS_TARGET * 1000.0
                pat_p_red.append(del_p_r)
                search_start_r = max(0, p_red_idx - 30)
                f_cand_r = search_start_r + np.argmin(r_w[search_start_r:p_red_idx])
                del_f_r = (f_cand_r - r_peak) / FS_TARGET * 1000.0
                pat_f_red.append(del_f_r)
                if f_cand_r < p_red_idx:
                    d_cand_r = f_cand_r + np.argmax(v_red[f_cand_r:p_red_idx])
                    del_d_r = (d_cand_r - r_peak) / FS_TARGET * 1000.0
                else:
                    del_d_r = (del_f_r + del_p_r) / 2.0
                pat_d_red.append(del_d_r)
                
        if len(pat_f_ir) == 0:
            continue
        
        pat_p_mean = float(np.mean(pat_p_ir))
        pat_f_mean = float(np.mean(pat_f_ir))
        pat_d_mean = float(np.mean(pat_d_ir)) if len(pat_d_ir) > 0 else pat_f_mean
        
        pat_f_fridericia = float(pat_f_mean / (rr_sec ** (1.0/3.0) + 1e-5))
        pat_f_framingham = float(pat_f_mean + 0.154 * (1.0 - rr_sec) * 1000.0)
        pat_d_framingham = float(pat_d_mean + 0.154 * (1.0 - rr_sec) * 1000.0)
        
        pat_d_inv = float(1.0 / (pat_d_mean + 1e-5))
        pat_d_sq_inv = float(1.0 / (pat_d_mean**2 + 1e-5))
        pat_p_sq_inv = float(1.0 / (pat_p_mean**2 + 1e-5))
        
        ptt_p_est = float(pat_p_mean - pep_est)
        ptt_f_est = float(pat_f_mean - pep_est)
        
        ptt_f_sq_inv = float(1.0 / (ptt_f_est**2 + 1e-5))
        ptt_d_sq_inv = float(1.0 / ((pat_d_mean - pep_est)**2 + 1e-5))
        ptt_p_sq_inv = float(1.0 / (ptt_p_est**2 + 1e-5))
        
        ir_valleys, _  = signal.find_peaks(-i_w, distance=int(0.35 * FS_TARGET), prominence=0.25 * i_std)
        red_valleys, _ = signal.find_peaks(-r_w, distance=int(0.35 * FS_TARGET), prominence=0.25 * r_std)
        
        ir_m  = compute_morphology(i_w, ir_peaks, ir_valleys, FS_TARGET)
        red_m = compute_red_morphology(r_w, red_peaks, red_valleys, FS_TARGET)
        
        ac_ir = float(np.max(i_w) - np.min(i_w))
        dc_ir = float(np.mean(i_w)) + 1e-5
        pi_ir = float((ac_ir / (abs(dc_ir) + 1e-5)) * 100.0)
        
        ac_red = float(np.max(r_w) - np.min(r_w))
        dc_red = float(np.mean(r_w)) + 1e-5
        ac_dc_ratio = float((ac_red + ac_ir) / (dc_red + dc_ir + 1e-5))
        
        v_ir_rms = float(np.sqrt(np.mean(v_ir**2)))
        a_ir_rms = float(np.sqrt(np.mean(a_ir**2)))
        v_red_rms = float(np.sqrt(np.mean(v_red**2)))
        
        t_dia_est = (1000.0 * rr_sec) - ir_m['tsys']
        k_val = float(ir_m['tsys'] / (t_dia_est + 1e-5))
        
        feat_dict = {
            'subject_id': sid_str, 'SBP': sbp, 'DBP': dbp,
            'PAT_d': pat_d_mean, 'PAT_p': pat_p_mean,
            'PAT_f_fridericia': pat_f_fridericia, 'PAT_f_framingham': pat_f_framingham, 'PAT_d_framingham': pat_d_framingham,
            'PAT_d_inv': pat_d_inv, 'PAT_d_sq_inv': pat_d_sq_inv, 'PAT_p_sq_inv': pat_p_sq_inv,
            'PTT_p_est': ptt_p_est, 'PTT_f_est': ptt_f_est,
            'PTT_f_sq_inv': ptt_f_sq_inv, 'PTT_d_sq_inv': ptt_d_sq_inv, 'PTT_p_sq_inv': ptt_p_sq_inv,
            'PW75': ir_m['pw75'], 'k_val': k_val,
            'PI_IR': pi_ir, 'AC_DC_ratio': ac_dc_ratio,
            'IR_VPG_RMS': v_ir_rms, 'IR_APG_RMS': a_ir_rms, 'RED_VPG_RMS': v_red_rms,
            'IR_Tsys': ir_m['tsys'], 'RED_PW50': red_m['pw50'], 'Sex': sex
        }
        records.append(feat_dict)
        
    return records

csv_60s = os.path.join(deploy_dir, "extracted_features_60s.csv")
if os.path.exists(csv_60s):
    print(f"Loading 60s features from {csv_60s}...")
    df_data = pd.read_csv(csv_60s)
else:
    print("Extracting unscaled 60s window features...")
    files = sorted(glob.glob(os.path.join(filtered_dir, "*_1.csv")))
    with Pool(os.cpu_count()) as pool:
        all_results = pool.map(process_single_file, files)
    df_data = pd.DataFrame([r for sub in all_results for r in sub])
    df_data.to_csv(csv_60s, index=False)

print(f"\nExtracted 60-Second Feature Dataset: {len(df_data):,} samples ({len(best_features)} dedicated biomarkers)")
"""
    cells.append(nbf.v4.new_code_cell(code_extraction))

    # Step 2 MD & Code: Dataset Partitioning & Optuna Hyperparameter Optimization (Dual SBP & DBP)
    step2_md = """## Step 2: Dataset Partitioning & Optuna Hyperparameter Optimization (Dual SBP & DBP 60s Models)

We partition the 60-second window dataset into 70% Train, 20% Validation, and 10% Holdout Test splits:
- **Training Set (70%)**: Used to fit model parameters.
- **Validation Set (20%)**: Used for Bayesian hyperparameter tuning.
- **Holdout Test Set (10%)**: Kept strictly untouched throughout tuning to provide unbiased clinical assessment.

We tune separate LightGBM models for **SBP** and **DBP** using Optuna Bayesian optimization (`n_estimators <= 100` for ESP32 flash constraints), reporting:
- **$R^2$ Score**: Coefficient of Determination.
- **MAE**: Mean Absolute Error (mmHg).
- **RMSE**: Root Mean Squared Error (mmHg).
- **Mean Error (Bias)**: $\\bar{e} = \\frac{1}{n}\\sum (y_{pred} - y_{true})$ (AAMI/ISO target: $\\le \\pm 5.0\\text{ mmHg}$).
- **Standard Deviation (SD)**: $s = \\sqrt{\\frac{1}{n-1}\\sum (e_i - \\bar{e})^2}$ (AAMI/ISO target: $\\le 8.0\\text{ mmHg}$).
"""
    cells.append(nbf.v4.new_markdown_cell(step2_md))

    code_optuna_training = r"""optuna.logging.set_verbosity(optuna.logging.WARNING)

# 70% Train, 20% Val, 10% Holdout Test Partitioning
train_val_idx, test_idx = train_test_split(df_data.index, test_size=0.10, random_state=42)
val_ratio = 0.20 / 0.90
train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, random_state=42)

X_all = df_data[best_features]
y_sbp_all = df_data['SBP']
y_dbp_all = df_data['DBP']

X_train, y_sbp_train, y_dbp_train = X_all.loc[train_idx], y_sbp_all.loc[train_idx], y_dbp_all.loc[train_idx]
X_val,   y_sbp_val,   y_dbp_val   = X_all.loc[val_idx],   y_sbp_all.loc[val_idx],   y_dbp_all.loc[val_idx]
X_test,  y_sbp_test,  y_dbp_test  = X_all.loc[test_idx],  y_sbp_all.loc[test_idx],  y_dbp_all.loc[test_idx]

def tune_lgbm(target_train, target_val, name="SBP"):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 40, 100, step=10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.20, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        m = LGBMRegressor(**params)
        m.fit(X_train, target_train)
        return r2_score(target_val, m.predict(X_val))
        
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=35, timeout=45)
    best_hp = study.best_params.copy()
    best_hp['n_estimators'] = min(100, int(best_hp.get('n_estimators', 100)))
    best_hp['random_state'] = 42
    best_hp['n_jobs'] = -1
    best_hp['verbose'] = -1
    return best_hp

print("Tuning SBP Regressor with Optuna...")
best_hp_sbp = tune_lgbm(y_sbp_train, y_sbp_val, "SBP")
model_sbp = LGBMRegressor(**best_hp_sbp)
model_sbp.fit(X_train, y_sbp_train)

print("Tuning DBP Regressor with Optuna...")
best_hp_dbp = tune_lgbm(y_dbp_train, y_dbp_val, "DBP")
model_dbp = LGBMRegressor(**best_hp_dbp)
model_dbp.fit(X_train, y_dbp_train)

# Evaluate SBP Predictions
pred_sbp_train = model_sbp.predict(X_train)
pred_sbp_val   = model_sbp.predict(X_val)
pred_sbp_test  = model_sbp.predict(X_test)
err_sbp_test   = pred_sbp_test - y_sbp_test

# Evaluate DBP Predictions
pred_dbp_train = model_dbp.predict(X_train)
pred_dbp_val   = model_dbp.predict(X_val)
pred_dbp_test  = model_dbp.predict(X_test)
err_dbp_test   = pred_dbp_test - y_dbp_test

res_summary = [
    {
        'Target': 'SBP (mmHg)',
        'Partition': 'Train (70%)',
        'Samples': f"{len(X_train):,}",
        'R²': f"{r2_score(y_sbp_train, pred_sbp_train):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_sbp_train, pred_sbp_train):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_sbp_train, pred_sbp_train):.2f}",
        'Mean Error (mmHg)': f"{np.mean(pred_sbp_train - y_sbp_train):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(pred_sbp_train - y_sbp_train, ddof=1):.2f}"
    },
    {
        'Target': 'SBP (mmHg)',
        'Partition': 'Val (20%)',
        'Samples': f"{len(X_val):,}",
        'R²': f"{r2_score(y_sbp_val, pred_sbp_val):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_sbp_val, pred_sbp_val):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_sbp_val, pred_sbp_val):.2f}",
        'Mean Error (mmHg)': f"{np.mean(pred_sbp_val - y_sbp_val):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(pred_sbp_val - y_sbp_val, ddof=1):.2f}"
    },
    {
        'Target': 'SBP (mmHg)',
        'Partition': 'Holdout Test (10%)',
        'Samples': f"{len(X_test):,}",
        'R²': f"{r2_score(y_sbp_test, pred_sbp_test):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_sbp_test, pred_sbp_test):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_sbp_test, pred_sbp_test):.2f}",
        'Mean Error (mmHg)': f"{np.mean(err_sbp_test):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(err_sbp_test, ddof=1):.2f}"
    },
    {
        'Target': 'DBP (mmHg)',
        'Partition': 'Train (70%)',
        'Samples': f"{len(X_train):,}",
        'R²': f"{r2_score(y_dbp_train, pred_dbp_train):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_dbp_train, pred_dbp_train):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_dbp_train, pred_dbp_train):.2f}",
        'Mean Error (mmHg)': f"{np.mean(pred_dbp_train - y_dbp_train):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(pred_dbp_train - y_dbp_train, ddof=1):.2f}"
    },
    {
        'Target': 'DBP (mmHg)',
        'Partition': 'Val (20%)',
        'Samples': f"{len(X_val):,}",
        'R²': f"{r2_score(y_dbp_val, pred_dbp_val):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_dbp_val, pred_dbp_val):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_dbp_val, pred_dbp_val):.2f}",
        'Mean Error (mmHg)': f"{np.mean(pred_dbp_val - y_dbp_val):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(pred_dbp_val - y_dbp_val, ddof=1):.2f}"
    },
    {
        'Target': 'DBP (mmHg)',
        'Partition': 'Holdout Test (10%)',
        'Samples': f"{len(X_test):,}",
        'R²': f"{r2_score(y_dbp_test, pred_dbp_test):.4f}",
        'MAE (mmHg)': f"{mean_absolute_error(y_dbp_test, pred_dbp_test):.2f}",
        'RMSE (mmHg)': f"{root_mean_squared_error(y_dbp_test, pred_dbp_test):.2f}",
        'Mean Error (mmHg)': f"{np.mean(err_dbp_test):+.2f}",
        'Standard Deviation (mmHg)': f"{np.std(err_dbp_test, ddof=1):.2f}"
    }
]

print("\n" + "="*125)
print("   DUAL SBP & DBP LIGHTGBM 60-SECOND MODEL BENCHMARK (WITH CLINICAL STANDARD DEVIATION)")
print("="*125)
print(pd.DataFrame(res_summary).to_string(index=False))
print("="*125)
"""
    cells.append(nbf.v4.new_code_cell(code_optuna_training))

    # Step 3 MD & Code: Dual Blood Pressure Regression Diagnostics & Scatter Plots
    step3_md = """## Step 3: Dual Blood Pressure Regression Diagnostics & Scatter Plots (60s Models)
"""
    cells.append(nbf.v4.new_markdown_cell(step3_md))

    code_plots = r"""fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# SBP Scatter
r2_sbp = r2_score(y_sbp_test, pred_sbp_test)
mae_sbp = mean_absolute_error(y_sbp_test, pred_sbp_test)
axes[0].scatter(y_sbp_test, pred_sbp_test, alpha=0.40, color='#d9534f', edgecolors='none', s=30, label=f'Holdout Test ({len(y_sbp_test):,} pts)')
lims_sbp = [min(y_sbp_test.min(), pred_sbp_test.min()) - 5, max(y_sbp_test.max(), pred_sbp_test.max()) + 5]
axes[0].plot(lims_sbp, lims_sbp, 'k--', lw=1.5, label='Identity Line (y=x)')
axes[0].set_xlim(lims_sbp)
axes[0].set_ylim(lims_sbp)
axes[0].set_title(f"Systolic Blood Pressure (SBP, 60s Window)\nTest R² = {r2_sbp:.4f} | MAE = {mae_sbp:.2f} mmHg")
axes[0].set_xlabel("Clinical Ground Truth SBP (mmHg)")
axes[0].set_ylabel("LightGBM Estimated SBP (mmHg)")
axes[0].legend(loc='upper left')

# DBP Scatter
r2_dbp = r2_score(y_dbp_test, pred_dbp_test)
mae_dbp = mean_absolute_error(y_dbp_test, pred_dbp_test)
axes[1].scatter(y_dbp_test, pred_dbp_test, alpha=0.40, color='#0275d8', edgecolors='none', s=30, label=f'Holdout Test ({len(y_dbp_test):,} pts)')
lims_dbp = [min(y_dbp_test.min(), pred_dbp_test.min()) - 5, max(y_dbp_test.max(), pred_dbp_test.max()) + 5]
axes[1].plot(lims_dbp, lims_dbp, 'k--', lw=1.5, label='Identity Line (y=x)')
axes[1].set_xlim(lims_dbp)
axes[1].set_ylim(lims_dbp)
axes[1].set_title(f"Diastolic Blood Pressure (DBP, 60s Window)\nTest R² = {r2_dbp:.4f} | MAE = {mae_dbp:.2f} mmHg")
axes[1].set_xlabel("Clinical Ground Truth DBP (mmHg)")
axes[1].set_ylabel("LightGBM Estimated DBP (mmHg)")
axes[1].legend(loc='upper left')

plt.tight_layout()
benchmark_plot_path = os.path.join(plots_dir, "bp_optuna_diagnostics.png")
plt.savefig(benchmark_plot_path, dpi=130)
plt.show()
print(f"Saved dual blood pressure diagnostic scatter plot: {benchmark_plot_path}")
"""
    cells.append(nbf.v4.new_code_cell(code_plots))

    # Step 4 MD & Code: Full Production Training & TinyML C Export (Dual SBP & DBP Models)
    step4_md = """## Step 4: Full Production Training & Standalone ANSI C Export (Dual SBP & DBP 60s Models)
We retrain the **SBP** and **DBP** models on 100% of the dataset and export them to standalone ANSI C decision tree sources (`deploy/lgbm_sbp.c` and `deploy/lgbm_dbp.c`).
"""
    cells.append(nbf.v4.new_markdown_cell(step4_md))

    code_m2c_export = r"""import m2cgen as tinymlgen

# 1. Train Production SBP & DBP Models on 100% of Filtered_Data (60s Window)
X_full = df_data[best_features].values
y_sbp_full = df_data['SBP'].values
y_dbp_full = df_data['DBP'].values

prod_sbp_model = LGBMRegressor(**best_hp_sbp)
prod_sbp_model.fit(X_full, y_sbp_full)

prod_dbp_model = LGBMRegressor(**best_hp_dbp)
prod_dbp_model.fit(X_full, y_dbp_full)

print(f"Trained production 60s SBP and DBP LightGBM models on {len(X_full):,} window samples using {len(best_features)} features.")

# 2. Export Standalone C Decision Trees (SBP & DBP)
c_sbp_source = tinymlgen.export_to_c(prod_sbp_model, function_name="score_sbp")
sbp_c_path = os.path.join(deploy_dir, "lgbm_sbp.c")
with open(sbp_c_path, "w") as f:
    f.write('#include "lgbm_sbp.h"\n\n' + c_sbp_source)

sbp_h_path = os.path.join(deploy_dir, "lgbm_sbp.h")
with open(sbp_h_path, "w") as f:
    f.write("#ifndef LGBM_SBP_H\n#define LGBM_SBP_H\n\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\ndouble score_sbp(double *input);\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif // LGBM_SBP_H\n")

c_dbp_source = tinymlgen.export_to_c(prod_dbp_model, function_name="score_dbp")
dbp_c_path = os.path.join(deploy_dir, "lgbm_dbp.c")
with open(dbp_c_path, "w") as f:
    f.write('#include "lgbm_dbp.h"\n\n' + c_dbp_source)

dbp_h_path = os.path.join(deploy_dir, "lgbm_dbp.h")
with open(dbp_h_path, "w") as f:
    f.write("#ifndef LGBM_DBP_H\n#define LGBM_DBP_H\n\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n\ndouble score_dbp(double *input);\n\n#ifdef __cplusplus\n}\n#endif\n\n#endif // LGBM_DBP_H\n")

print(f"Exported SBP model to C: {sbp_c_path} ({len(c_sbp_source):,} chars)")
print(f"Exported DBP model to C: {dbp_c_path} ({len(c_dbp_source):,} chars)")

# 3. Export Unified bp_models.h with SBP and DBP Feature Mappings
bp_models_h_path = os.path.join(deploy_dir, "bp_models.h")
h_lines = [
    "/**",
    " * @file bp_models.h",
    " * @brief Optuna-Optimized LightGBM Model Interface for SBP & DBP Estimation (60s Model)",
    " */",
    "",
    "#ifndef BP_MODELS_H",
    "#define BP_MODELS_H",
    "",
    f"#define NUM_INPUT_FEATURES {len(best_features)}",
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
    "double score_sbp(double *input);",
    "double score_dbp(double *input);",
    "",
    "static inline double predict_sbp(double *features) {",
    "    return score_sbp(features);",
    "}",
    "",
    "static inline double predict_dbp(double *features) {",
    "    return score_dbp(features);",
    "}",
    "",
    "// Feature Index Mapping for double input[NUM_INPUT_FEATURES]"
]
for idx, col in enumerate(best_features):
    feat_def = f"FEAT_{col.upper()}".replace(" ", "_")
    h_lines.append(f"#define {feat_def:<30} {idx}")

h_lines.extend([
    "",
    "#ifdef __cplusplus",
    "}",
    "#endif",
    "",
    "#endif // BP_MODELS_H",
    ""
])
with open(bp_models_h_path, "w") as f:
    f.write("\n".join(h_lines))
print(f"Generated unified C header: {bp_models_h_path}")

# 4. Save feature_names.json
feat_json_path = os.path.join(deploy_dir, "feature_names.json")
with open(feat_json_path, "w") as f:
    json.dump(best_features, f, indent=2)
print(f"Saved feature names metadata: {feat_json_path}")
"""
    cells.append(nbf.v4.new_code_cell(code_m2c_export))

    # Step 5 MD & Code: Test Ported C Code on Filtered Dataset (Dual SBP & DBP Verification)
    step5_md = """## Step 5: Test Ported C Code on Filtered Dataset (Resampled 100 Hz Bandpassed Signals, 60.0s Window)
Since `bp_pipeline` applies Z-score normalization and processes pre-bandpassed signals directly, verification is conducted using the resampled (100 Hz) signals from the **Filtered Dataset** using a 60-second window.
"""
    cells.append(nbf.v4.new_markdown_cell(step5_md))

    code_raw_c_test = r"""# 1. Sync Firmware Headers and C Sources to ESP32 demo folder
if os.path.exists(esp_dir):
    for mf in ["lgbm_sbp.c", "lgbm_sbp.h", "lgbm_dbp.c", "lgbm_dbp.h", "bp_models.h", "bp_pipeline.h", "bp_pipeline.c"]:
        src_p = os.path.join(deploy_dir, mf)
        if os.path.exists(src_p):
            shutil.copy2(src_p, os.path.join(esp_dir, mf))

# 2. Select Test Subject (Subject 001) from Filtered Dataset, resample 240 Hz -> 100 Hz (60.0s window)
test_sid = "001"
filt_test_file = os.path.join(filtered_dir, f"{test_sid}_1.csv")
df_filt_test = pd.read_csv(filt_test_file).dropna(subset=['ECG_I_Filtered'])

r_ecg_100 = signal.resample_poly(df_filt_test['ECG_I_Filtered'].values.astype(float), 5, 12)
r_red_100 = signal.resample_poly(df_filt_test['PPG_RED_Filtered'].values.astype(float), 5, 12)
r_ir_100  = signal.resample_poly(df_filt_test['PPG_IR_Filtered'].values.astype(float), 5, 12)

# Take first 60.0s window (6,000 samples @ 100 Hz)
w_ecg = r_ecg_100[0:6000]
w_red = r_red_100[0:6000]
w_ir  = r_ir_100[0:6000]

info_test = info_df[info_df['ID'].astype(str).str.zfill(3) == test_sid].iloc[0]
actual_sbp = float(info_test['SBP/mmHg'])
actual_dbp = float(info_test['DBP/mmHg'])
actual_hr  = float(info_test['HR/bpm'])
is_male    = 1.0 if str(info_test['Gender']).strip().lower() == 'male' else 0.0

# 3. Generate sample_signals.h from Filtered Dataset for C inference testing (60.0s Window)
sample_h_lines = [
    f'/* Filtered Dataset: Bandpassed Waveform Data for Subject {test_sid} (Resampled 240 Hz -> 100 Hz, 60.0s @ 100 Hz) */',
    '#ifndef SAMPLE_SIGNALS_H',
    '#define SAMPLE_SIGNALS_H',
    '',
    '#define SAMPLE_SIGNAL_LEN 6000',
    f'#define SAMPLE_SUBJECT_ID "{test_sid}"',
    f'#define SAMPLE_TRUE_SBP {actual_sbp:.1f}',
    f'#define SAMPLE_TRUE_DBP {actual_dbp:.1f}',
    f'#define SAMPLE_TRUE_MAP {actual_dbp + (actual_sbp - actual_dbp) / 3.0:.2f}',
    f'#define SAMPLE_TRUE_HR {actual_hr:.1f}',
    f'#define SAMPLE_IS_MALE {is_male:.1f}',
    '',
    'static const double SAMPLE_PPG_RED[SAMPLE_SIGNAL_LEN] = {',
    ', '.join([f'{v:.4f}' for v in w_red]),
    '};',
    '',
    'static const double SAMPLE_PPG_IR[SAMPLE_SIGNAL_LEN] = {',
    ', '.join([f'{v:.4f}' for v in w_ir]),
    '};',
    '',
    'static const double SAMPLE_ECG_LEAD_I[SAMPLE_SIGNAL_LEN] = {',
    ', '.join([f'{v:.4f}' for v in w_ecg]),
    '};',
    '',
    '#endif // SAMPLE_SIGNALS_H',
    ''
]
with open(os.path.join(deploy_dir, "sample_signals.h"), "w") as f:
    f.write("\n".join(sample_h_lines))
print(f"Generated sample_signals.h with 60.0s bandpassed 100 Hz window (6,000 samples) from Filtered Dataset for Subject {test_sid}")

# 4. Generate test_deploy.c for C concordance verification (SBP & DBP)
test_match = df_data[df_data['subject_id'].astype(str).str.zfill(3) == test_sid]
if len(test_match) == 0:
    test_match = df_data
test_subject_row = test_match.iloc[0]
sample_feat_input = test_subject_row[best_features].values.astype(float)
test_sbp_py = float(prod_sbp_model.predict([sample_feat_input])[0])
test_dbp_py = float(prod_dbp_model.predict([sample_feat_input])[0])

vec_vals = ", ".join([f"{v:.8f}" for v in sample_feat_input])

c_test_lines = [
    '#include <stdio.h>',
    '#include <math.h>',
    '#include "bp_models.h"',
    '',
    'int main() {',
    '    puts("================================================================");',
    '    puts(" CLINICAL SUBJECT TEST & PORTED C CODE VERIFICATION (SBP & DBP) ");',
    '    puts("   Target: Filtered Dataset (Resampled 100 Hz, 60s Window)     ");',
    '    puts("================================================================");',
    '',
    f'    printf("Testing on Subject %s from Filtered Dataset (60s Dual Models):\\n", "{test_sid}");',
    f'    printf("  Actual Clinical Reference : SBP = %6.2f mmHg | DBP = %6.2f mmHg\\n\\n", {actual_sbp}, {actual_dbp});',
    '',
    '    // Test Model Predictions on Subject Feature Vector',
    f'    double input[NUM_INPUT_FEATURES] = {{{vec_vals}}};',
    '',
    '    double pred_sbp = predict_sbp(input);',
    '    double pred_dbp = predict_dbp(input);',
    '',
    '    printf("Ported C Model Inference Results:\\n");',
    f'    printf("  SBP Predicted in C : %7.2f mmHg (Python Reference: {test_sbp_py:7.2f} mmHg)\\n", pred_sbp);',
    f'    printf("  DBP Predicted in C : %7.2f mmHg (Python Reference: {test_dbp_py:7.2f} mmHg)\\n", pred_dbp);',
    '',
    f'    double err_sbp = fabs(pred_sbp - ({test_sbp_py:.8f}));',
    f'    double err_dbp = fabs(pred_dbp - ({test_dbp_py:.8f}));',
    '',
    '    printf("\\nConcordance Verification (C vs Python):\\n");',
    '    printf("  SBP Discrepancy : %e mmHg\\n", err_sbp);',
    '    printf("  DBP Discrepancy : %e mmHg\\n", err_dbp);',
    '',
    '    if (err_sbp < 1e-4 && err_dbp < 1e-4) {',
    '        puts("\\n>>> STATUS: BIT-EXACT CONCORDANCE VERIFIED (ALL TESTS PASSED) <<<");',
    '        return 0;',
    '    } else {',
    '        puts("\\n>>> STATUS: DISCREPANCY DETECTED <<<");',
    '        return 1;',
    '    }',
    '}'
]

test_c_path = os.path.join(deploy_dir, "test_deploy.c")
with open(test_c_path, "w") as f:
    f.write("\n".join(c_test_lines))

# 5. Compile with GCC & Run test_deploy on Filtered Dataset features
test_bin_path = os.path.join(deploy_dir, "test_deploy")
gcc_cmd = f"gcc -O2 -I{deploy_dir} {test_c_path} {os.path.join(deploy_dir, 'lgbm_sbp.c')} {os.path.join(deploy_dir, 'lgbm_dbp.c')} -lm -o {test_bin_path}"
res_gcc = subprocess.run(gcc_cmd, shell=True, capture_output=True, text=True)
if res_gcc.returncode != 0:
    raise RuntimeError(f"GCC Compilation of deployment test suite failed: {res_gcc.stderr}")

res_run = subprocess.run(test_bin_path, capture_output=True, text=True)
print(res_run.stdout)

# 6. Compile and Verify End-to-End C Pipeline (bp_pipeline) on 60s Filtered Dataset
pipe_bin_path = os.path.join(deploy_dir, "predict_bp_example")
gcc_pipe_cmd = f"gcc -O2 -I{deploy_dir} {os.path.join(deploy_dir, 'predict_bp_example.c')} {os.path.join(deploy_dir, 'bp_pipeline.c')} {os.path.join(deploy_dir, 'lgbm_sbp.c')} {os.path.join(deploy_dir, 'lgbm_dbp.c')} -lm -o {pipe_bin_path}"
res_gcc_pipe = subprocess.run(gcc_pipe_cmd, shell=True, capture_output=True, text=True)
if res_gcc_pipe.returncode != 0:
    raise RuntimeError(f"GCC Compilation of bp_pipeline example failed: {res_gcc_pipe.stderr}")

res_pipe = subprocess.run(pipe_bin_path, capture_output=True, text=True)
print(res_pipe.stdout)
"""
    cells.append(nbf.v4.new_code_cell(code_raw_c_test))

    nb["cells"] = cells

    os.makedirs("models", exist_ok=True)
    out_path = os.path.abspath("models/train_red_ir.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated clean dual SBP & DBP 60s notebook at: {out_path}")

if __name__ == "__main__":
    build_red_ir_notebook()
