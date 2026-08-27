import numpy as np
import scipy.signal as signal
import pandas as pd
import json
import subprocess

with open('deploy/feature_names.json') as f:
    feature_names = json.load(f)

FS_TARGET = 100.0

def c_find_peaks(sig, min_dist, min_prom, max_peaks=128):
    length = len(sig)
    candidates = []
    for i in range(1, length - 1):
        if sig[i] > sig[i-1] and sig[i] >= sig[i+1]:
            left_min = sig[i]
            for k in range(i-1, -1, -1):
                if sig[k] > sig[i]: break
                if sig[k] < left_min: left_min = sig[k]
            right_min = sig[i]
            for k in range(i+1, length):
                if sig[k] > sig[i]: break
                if sig[k] < right_min: right_min = sig[k]
            higher_valley = max(left_min, right_min)
            prom = sig[i] - higher_valley
            if prom >= min_prom:
                candidates.append((i, sig[i]))
    if not candidates: return []
    candidates.sort(key=lambda x: x[1], reverse=True)
    kept = []
    for idx, h in candidates:
        if all(abs(idx - k) >= min_dist for k in kept):
            kept.append(idx)
        if len(kept) >= max_peaks: break
    kept.sort()
    return kept

def compute_single_pulse(pulse):
    pulse_len = len(pulse)
    if pulse_len < 20:
        return {'pw25':0.0, 'pw50':0.0, 'pw75':0.0, 'area_a1':0.0, 'area_a2':0.0,
                'area_ratio':0.0, 'ipa_ratio':0.0, 'decay_slope':0.0, 'tsys':0.0, 'tdia':0.0,
                'aix':0.0, 'apg_b_a':0.0, 'apg_agi':0.0}
    
    p_peak = int(np.argmax(pulse))
    tsys = float(p_peak / FS_TARGET * 1000.0)
    tdia = float((pulse_len - 1 - p_peak) / FS_TARGET * 1000.0)
    
    c25 = int(np.sum(pulse >= 0.25))
    c50 = int(np.sum(pulse >= 0.50))
    c75 = int(np.sum(pulse >= 0.75))
    
    pw25 = float(c25 / FS_TARGET * 1000.0)
    pw50 = float(c50 / FS_TARGET * 1000.0)
    pw75 = float(c75 / FS_TARGET * 1000.0)
    
    sum_a1 = float(np.sum(pulse[:p_peak+1]))
    sum_a2 = float(np.sum(pulse[p_peak+1:])) if p_peak < pulse_len - 1 else 0.0
    area_ratio = float(sum_a1 / (sum_a2 + 1e-5))
    ipa_ratio = float(sum_a2 / (sum_a1 + sum_a2 + 1e-5))
    
    decay_len = pulse_len - 1 - p_peak
    decay_slope = float((pulse[-1] - pulse[p_peak]) / (decay_len / FS_TARGET + 1e-5)) if decay_len > 0 else 0.0
    
    v = np.zeros(pulse_len)
    v[0] = pulse[1] - pulse[0]
    for i in range(1, pulse_len - 1):
        v[i] = (pulse[i+1] - pulse[i-1]) / 2.0
    v[-1] = pulse[-1] - pulse[-2]
    
    a = np.zeros(pulse_len)
    a[0] = v[1] - v[0]
    for i in range(1, pulse_len - 1):
        a[i] = (v[i+1] - v[i-1]) / 2.0
    a[-1] = v[-1] - v[-2]
    
    a_peaks = c_find_peaks(a, 3, 0.001, 32)
    a_val = a[a_peaks[0]] if len(a_peaks) > 0 else a[0]
    if abs(a_val) < 1e-5: a_val = 1.0
    
    b_val = float(np.min(a[:p_peak+1])) if p_peak > 0 else a[0]
    apg_b_a = float(b_val / (abs(a_val) + 1e-5))
    apg_agi = apg_b_a
    
    neg_v = -v
    v_valleys = c_find_peaks(neg_v, 3, 0.001, 32)
    notch_idx = -1
    for k in v_valleys:
        if k > p_peak:
            notch_idx = k
            break
    if notch_idx >= 0 and notch_idx < pulse_len - 1:
        dia_peak = notch_idx + int(np.argmax(pulse[notch_idx:]))
        aix = float((pulse[dia_peak] - pulse[p_peak]) / (pulse[p_peak] + 1e-5))
    else:
        aix = 0.0
        
    return {
        'pw25': pw25, 'pw50': pw50, 'pw75': pw75,
        'area_a1': sum_a1, 'area_a2': sum_a2,
        'area_ratio': area_ratio, 'ipa_ratio': ipa_ratio,
        'decay_slope': decay_slope, 'tsys': tsys, 'tdia': tdia,
        'aix': aix, 'apg_b_a': apg_b_a, 'apg_agi': apg_agi
    }

def compute_morphology(sig):
    length = len(sig)
    neg_sig = -sig
    feet = c_find_peaks(neg_sig, int(0.35 * FS_TARGET), 0.02, 64)
    if len(feet) < 2:
        return compute_single_pulse(sig)
    
    metrics = []
    for i in range(len(feet) - 1):
        f_start = feet[i]
        f_end = feet[i+1]
        plen = f_end - f_start
        if plen >= 35:
            single = compute_single_pulse(sig[f_start:f_end])
            metrics.append(single)
            
    if not metrics:
        return compute_single_pulse(sig)
        
    avg = {}
    for k in metrics[0].keys():
        avg[k] = float(np.mean([m[k] for m in metrics]))
    return avg

def py_extract_pipeline(raw_ppg_red, raw_ppg_ir, raw_ecg, is_male=1.0):
    num_samples = len(raw_ecg)
    
    ppg_sos = signal.cheby2(4, 30, [0.2, 10.0], btype='bandpass', fs=100.0, output='sos')
    ecg_sos = signal.butter(3, [0.5, 35.0], btype='bandpass', fs=100.0, output='sos')
    
    ir_filt = signal.sosfiltfilt(ppg_sos, raw_ppg_ir - np.mean(raw_ppg_ir))
    red_filt = signal.sosfiltfilt(ppg_sos, raw_ppg_red - np.mean(raw_ppg_red))
    ecg_filt = signal.sosfiltfilt(ecg_sos, raw_ecg - np.mean(raw_ecg))
    
    ir_norm = (ir_filt - np.min(ir_filt)) / (np.max(ir_filt) - np.min(ir_filt) + 1e-5)
    red_norm = (red_filt - np.min(red_filt)) / (np.max(red_filt) - np.min(red_filt) + 1e-5)
    ecg_norm = (ecg_filt - np.min(ecg_filt)) / (np.max(ecg_filt) - np.min(ecg_filt) + 1e-5)
    
    ecg_diff = np.zeros(num_samples)
    ecg_diff[0] = ecg_norm[1] - ecg_norm[0]
    for i in range(1, num_samples - 1):
        ecg_diff[i] = (ecg_norm[i+1] - ecg_norm[i-1]) / 2.0
    ecg_diff[-1] = ecg_norm[-1] - ecg_norm[-2]
    
    ecg_qrs = np.zeros(num_samples)
    for i in range(num_samples):
        sum_sq = 0.0
        cnt = 0
        for k in range(-7, 8):
            idx = i + k
            if 0 <= idx < num_samples:
                sum_sq += ecg_diff[idx]**2
                cnt += 1
        ecg_qrs[i] = sum_sq / cnt if cnt > 0 else ecg_diff[i]**2
    ecg_qrs_norm = (ecg_qrs - np.min(ecg_qrs)) / (np.max(ecg_qrs) - np.min(ecg_qrs) + 1e-5)
    
    ecg_peaks = c_find_peaks(ecg_qrs_norm, int(0.35 * FS_TARGET), 0.10, 128)
    ir_peaks  = c_find_peaks(ir_norm,      int(0.35 * FS_TARGET), 0.05, 128)
    red_peaks = c_find_peaks(red_norm,     int(0.35 * FS_TARGET), 0.05, 128)
    
    if len(ecg_peaks) < 2 or len(ir_peaks) < 2 or len(red_peaks) < 2:
        return None
        
    v_ir = np.zeros(num_samples)
    v_ir[0] = ir_norm[1] - ir_norm[0]
    for i in range(1, num_samples - 1): v_ir[i] = (ir_norm[i+1] - ir_norm[i-1]) / 2.0
    v_ir[-1] = ir_norm[-1] - ir_norm[-2]
    
    a_ir = np.zeros(num_samples)
    a_ir[0] = v_ir[1] - v_ir[0]
    for i in range(1, num_samples - 1): a_ir[i] = (v_ir[i+1] - v_ir[i-1]) / 2.0
    a_ir[-1] = v_ir[-1] - v_ir[-2]
    
    v_red = np.zeros(num_samples)
    v_red[0] = red_norm[1] - red_norm[0]
    for i in range(1, num_samples - 1): v_red[i] = (red_norm[i+1] - red_norm[i-1]) / 2.0
    v_red[-1] = red_norm[-1] - red_norm[-2]
    
    a_red = np.zeros(num_samples)
    a_red[0] = v_red[1] - v_red[0]
    for i in range(1, num_samples - 1): a_red[i] = (v_red[i+1] - v_red[i-1]) / 2.0
    a_red[-1] = v_red[-1] - v_red[-2]
    
    pat_p_ir, pat_f_ir, pat_d_ir = [], [], []
    pat_p_red, pat_f_red, pat_d_red = [], [], []
    ptt_inter_p, ptt_inter_f = [], []
    
    for r_i in ecg_peaks:
        p_ir_idx, p_red_idx = -1, -1
        f_ir_idx, f_red_idx = -1, -1
        
        for p in ir_peaks:
            if r_i < p < r_i + int(0.60 * FS_TARGET):
                p_ir_idx = p
                break
        for p in red_peaks:
            if r_i < p < r_i + int(0.60 * FS_TARGET):
                p_red_idx = p
                break
                
        if p_ir_idx >= 0:
            del_p = (p_ir_idx - r_i) / FS_TARGET * 1000.0
            pat_p_ir.append(del_p)
            s_ir = max(0, p_ir_idx - int(0.30 * FS_TARGET))
            f_ir_idx = s_ir + int(np.argmin(ir_norm[s_ir:p_ir_idx]))
            del_f = (f_ir_idx - r_i) / FS_TARGET * 1000.0
            pat_f_ir.append(del_f)
            if f_ir_idx < p_ir_idx:
                d_i = f_ir_idx + int(np.argmax(v_ir[f_ir_idx:p_ir_idx]))
                del_d = (d_i - r_i) / FS_TARGET * 1000.0
            else:
                del_d = (del_f + del_p) / 2.0
            pat_d_ir.append(del_d)
            
        if p_red_idx >= 0:
            del_p_r = (p_red_idx - r_i) / FS_TARGET * 1000.0
            pat_p_red.append(del_p_r)
            s_red = max(0, p_red_idx - int(0.30 * FS_TARGET))
            f_red_idx = s_red + int(np.argmin(red_norm[s_red:p_red_idx]))
            del_f_r = (f_red_idx - r_i) / FS_TARGET * 1000.0
            pat_f_red.append(del_f_r)
            if f_red_idx < p_red_idx:
                d_i_r = f_red_idx + int(np.argmax(v_red[f_red_idx:p_red_idx]))
                del_d_r = (d_i_r - r_i) / FS_TARGET * 1000.0
            else:
                del_d_r = (del_f_r + del_p_r) / 2.0
            pat_d_red.append(del_d_r)
            
        if p_ir_idx >= 0 and p_red_idx >= 0:
            ptt_inter_p.append((p_red_idx - p_ir_idx) / FS_TARGET * 1000.0)
            if f_ir_idx >= 0 and f_red_idx >= 0:
                ptt_inter_f.append((f_red_idx - f_ir_idx) / FS_TARGET * 1000.0)
            else:
                ptt_inter_f.append((p_red_idx - p_ir_idx) / FS_TARGET * 1000.0)
                
    if not pat_p_ir:
        return None
        
    pat_p_mean = float(np.mean(pat_p_ir))
    pat_f_mean = float(np.mean(pat_f_ir))
    pat_d_mean = float(np.mean(pat_d_ir))
    
    ptt_inter_peak = float(np.mean(ptt_inter_p)) if ptt_inter_p else 0.0
    ptt_inter_foot = float(np.mean(ptt_inter_f)) if ptt_inter_f else 0.0
    
    pat_p_red_mean = float(np.mean(pat_p_red)) if pat_p_red else pat_p_mean
    pat_f_red_mean = float(np.mean(pat_f_red)) if pat_f_red else pat_f_mean
    pat_d_red_mean = float(np.mean(pat_d_red)) if pat_d_red else pat_d_mean
    
    delta_pat_peak_red_ir = float(pat_p_red_mean - pat_p_mean)
    delta_pat_foot_red_ir = float(pat_f_red_mean - pat_f_mean)
    delta_pat_deriv_red_ir = float(pat_d_red_mean - pat_d_mean)
    
    rr_sec = 0.8
    if len(ecg_peaks) >= 2:
        rr_sec = float(np.mean(np.diff(ecg_peaks)) / FS_TARGET)
    if rr_sec < 0.3: rr_sec = 0.3
    if rr_sec > 2.0: rr_sec = 2.0
    
    pep_est = 60.0 + 0.12 * (1000.0 * rr_sec) * 0.1
    ptt_p_est = pat_p_mean - pep_est
    ptt_f_est = pat_f_mean - pep_est
    ptt_d_est = pat_d_mean - pep_est
    
    pat_f_bazett = pat_f_mean / np.sqrt(rr_sec + 1e-5)
    pat_d_bazett = pat_d_mean / np.sqrt(rr_sec + 1e-5)
    pat_p_bazett = pat_p_mean / np.sqrt(rr_sec + 1e-5)
    
    pat_f_fridericia = pat_f_mean / (rr_sec**(1.0/3.0) + 1e-5)
    pat_d_fridericia = pat_d_mean / (rr_sec**(1.0/3.0) + 1e-5)
    pat_p_fridericia = pat_p_mean / (rr_sec**(1.0/3.0) + 1e-5)
    
    pat_f_framingham = pat_f_mean + 0.154 * (1.0 - rr_sec) * 1000.0
    pat_d_framingham = pat_d_mean + 0.154 * (1.0 - rr_sec) * 1000.0
    pat_p_framingham = pat_p_mean + 0.154 * (1.0 - rr_sec) * 1000.0
    
    pat_f_inv = 1.0 / (pat_f_mean + 1e-5)
    pat_f_sq_inv = 1.0 / (pat_f_mean**2 + 1e-5)
    pat_d_inv = 1.0 / (pat_d_mean + 1e-5)
    pat_d_sq_inv = 1.0 / (pat_d_mean**2 + 1e-5)
    pat_p_inv = 1.0 / (pat_p_mean + 1e-5)
    pat_p_sq_inv = 1.0 / (pat_p_mean**2 + 1e-5)
    
    ptt_f_inv = 1.0 / (ptt_f_est + 1e-5)
    ptt_d_inv = 1.0 / (ptt_d_est + 1e-5)
    ptt_p_inv = 1.0 / (ptt_p_est + 1e-5)
    ptt_f_sq_inv = 1.0 / (ptt_f_est**2 + 1e-5)
    ptt_d_sq_inv = 1.0 / (ptt_d_est**2 + 1e-5)
    ptt_p_sq_inv = 1.0 / (ptt_p_est**2 + 1e-5)
    
    t_sys_dia = pat_p_mean - pat_f_mean
    t_sys_deriv = pat_d_mean - pat_f_mean
    t_deriv_dia = pat_p_mean - pat_d_mean
    
    ac_ir = float(np.max(raw_ppg_ir) - np.min(raw_ppg_ir))
    dc_ir = float(np.mean(raw_ppg_ir) + 1e-5)
    ac_red = float(np.max(raw_ppg_red) - np.min(raw_ppg_red))
    dc_red = float(np.mean(raw_ppg_red) + 1e-5)
    
    pi_ir = ac_ir / dc_ir
    pi_red = ac_red / dc_red
    r_optical = pi_red / (pi_ir + 1e-5)
    ac_dc_ratio = pi_ir
    
    ir_vpg_rms = float(np.sqrt(np.mean(v_ir**2)))
    ir_apg_rms = float(np.sqrt(np.mean(a_ir**2)))
    red_vpg_rms = float(np.sqrt(np.mean(v_red**2)))
    red_apg_rms = float(np.sqrt(np.mean(a_red**2)))
    
    ir_shr = float(ir_vpg_rms / (ir_apg_rms + 1e-5))
    red_shr = float(red_vpg_rms / (red_apg_rms + 1e-5))
    
    m_ir = compute_morphology(ir_norm)
    m_red = compute_morphology(red_norm)
    
    k_val = float(m_ir['tsys'] / (t_sys_dia + 1e-5))
    
    feat_dict = {
        'PAT_f': pat_f_mean, 'PAT_d': pat_d_mean, 'PAT_p': pat_p_mean,
        'PAT_f_bazett': pat_f_bazett, 'PAT_d_bazett': pat_d_bazett, 'PAT_p_bazett': pat_p_bazett,
        'PAT_f_fridericia': pat_f_fridericia, 'PAT_d_fridericia': pat_d_fridericia, 'PAT_p_fridericia': pat_p_fridericia,
        'PAT_f_framingham': pat_f_framingham, 'PAT_d_framingham': pat_d_framingham, 'PAT_p_framingham': pat_p_framingham,
        'PAT_f_inv': pat_f_inv, 'PAT_f_sq_inv': pat_f_sq_inv,
        'PAT_d_inv': pat_d_inv, 'PAT_d_sq_inv': pat_d_sq_inv,
        'PAT_p_inv': pat_p_inv, 'PAT_p_sq_inv': pat_p_sq_inv,
        'PTT_p_est': ptt_p_est, 'PTT_f_est': ptt_f_est, 'PTT_d_est': ptt_d_est,
        'PTT_f_inv': ptt_f_inv, 'PTT_d_inv': ptt_d_inv, 'PTT_p_inv': ptt_p_inv,
        'PTT_f_sq_inv': ptt_f_sq_inv, 'PTT_d_sq_inv': ptt_d_sq_inv, 'PTT_p_sq_inv': ptt_p_sq_inv,
        'PTT_inter_peak': ptt_inter_peak, 'PTT_inter_foot': ptt_inter_foot,
        'delta_PAT_peak_red_ir': delta_pat_peak_red_ir, 'delta_PAT_foot_red_ir': delta_pat_foot_red_ir, 'delta_PAT_deriv_red_ir': delta_pat_deriv_red_ir,
        'T_sys_dia': t_sys_dia, 'T_sys_deriv': t_sys_deriv, 'T_deriv_dia': t_deriv_dia,
        'PW25': m_ir['pw25'], 'PW50': m_ir['pw50'], 'PW75': m_ir['pw75'],
        'k_val': k_val, 'area_ratio': m_ir['area_ratio'], 'AIx': m_ir['aix'], 'AIx_red': m_red['aix'],
        'PI_IR': pi_ir, 'PI_RED': pi_red, 'R_optical_ratio': r_optical, 'AC_DC_ratio': ac_dc_ratio,
        'IR_VPG_RMS': ir_vpg_rms, 'IR_APG_RMS': ir_apg_rms, 'RED_VPG_RMS': red_vpg_rms, 'RED_APG_RMS': red_apg_rms,
        'IR_SHR': ir_shr, 'RED_SHR': red_shr,
        'IR_Tsys': m_ir['tsys'], 'IR_decay_slope': m_ir['decay_slope'], 'IR_area_A1': m_ir['area_a1'], 'IR_area_A2': m_ir['area_a2'],
        'IR_IPA_ratio': m_ir['ipa_ratio'], 'IR_APG_b_a': m_ir['apg_b_a'], 'IR_APG_AGI': m_ir['apg_agi'],
        'RED_Tsys': m_red['tsys'], 'RED_decay_slope': m_red['decay_slope'], 'RED_PW25': m_red['pw25'], 'RED_PW50': m_red['pw50'], 'RED_PW75': m_red['pw75'],
        'RED_area_A1': m_red['area_a1'], 'RED_area_A2': m_red['area_a2'], 'RED_IPA_ratio': m_red['ipa_ratio'], 'RED_APG_b_a': m_red['apg_b_a'], 'RED_APG_AGI': m_red['apg_agi'],
        'Sex': float(is_male)
    }
    return [feat_dict[fn] for fn in feature_names]

if __name__ == '__main__':
    df_raw = pd.read_csv('A dataset of simultaneous collected ECG and PPG signals/Raw_data/001_1.csv')
    e_r = signal.resample_poly(df_raw['ECG_I'].dropna().values.astype(float), 5, 12)[1000:2000]
    r_r = signal.resample_poly(df_raw['PPG_RED'].dropna().values.astype(float), 5, 12)[1000:2000]
    i_r = signal.resample_poly(df_raw['PPG_IR'].dropna().values.astype(float), 5, 12)[1000:2000]

    py_f = py_extract_pipeline(r_r, i_r, e_r, 1.0)
    
    res = subprocess.run('./test_w1', capture_output=True, text=True)
    lines = res.stdout.strip().split('\n')
    c_feats = [float(x) for x in lines[1:]]

    print(f'Feature Comparison (Python vs C) on Raw Window 1 [1000:2000]:')
    max_diff = 0.0
    for i in range(len(feature_names)):
        diff = abs(py_f[i] - c_feats[i])
        if diff > max_diff: max_diff = diff
        print(f'{i:2d} | {feature_names[i]:<25} | Py: {py_f[i]:12.4f} | C: {c_feats[i]:12.4f} | Diff: {diff:10.4e}')

    print(f'\nMAX DIFFERENCE ACROSS ALL 70 FEATURES: {max_diff:e}')
