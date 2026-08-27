#include "signal_processor.h"
#include "bp_models.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Helper to sort 3 values for 3-sample median filter
static double median3(double a, double b, double c) {
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
}

void median_filter_30ms(const double *input, double *output, size_t length) {
    if (length < 3) {
        memcpy(output, input, length * sizeof(double));
        return;
    }
    output[0] = input[0];
    for (size_t i = 1; i < length - 1; i++) {
        output[i] = median3(input[i-1], input[i], input[i+1]);
    }
    output[length - 1] = input[length - 1];
}

void min_max_normalize(double *buffer, size_t length) {
    if (length == 0) return;
    double min_v = buffer[0];
    double max_v = buffer[0];
    for (size_t i = 1; i < length; i++) {
        if (buffer[i] < min_v) min_v = buffer[i];
        if (buffer[i] > max_v) max_v = buffer[i];
    }
    double range = max_v - min_v;
    if (range > 1e-9) {
        for (size_t i = 0; i < length; i++) {
            buffer[i] = (buffer[i] - min_v) / range;
        }
    } else {
        for (size_t i = 0; i < length; i++) buffer[i] = 0.0;
    }
}

// 4th-Order Chebyshev Type II Filter for PPG at 100 Hz (0.2 - 10.0 Hz)
void ppg_bandpass_filter_100hz(const double *input, double *output, size_t length) {
    ppg_biquad_cascade_t filter;
    ppg_filter_reset(&filter);
    for (size_t i = 0; i < length; i++) {
        output[i] = ppg_filter_step(&filter, input[i]);
    }
}

// 3rd-Order Butterworth BandPass Filter for ECG at 100 Hz (0.5 - 35.0 Hz)
void ecg_bandpass_filter_100hz(const double *input, double *output, size_t length) {
    ecg_filter_state_t filter;
    ecg_filter_reset(&filter);
    for (size_t i = 0; i < length; i++) {
        output[i] = ecg_filter_step(&filter, input[i]);
    }
}

bool preprocess_and_extract_features_dual_ppg_100hz(
    const double *red_raw_100hz,
    const double *ir_raw_100hz,
    const double *ecg_raw_100hz,
    size_t num_samples,
    double features_out[NUM_INPUT_FEATURES]
) {
    if (num_samples < 400) return false;

    // Clear feature output
    for (int i = 0; i < NUM_INPUT_FEATURES; i++) {
        features_out[i] = 0.0;
    }

    double *ir_med = (double *)malloc(num_samples * sizeof(double));
    double *ir_filt = (double *)malloc(num_samples * sizeof(double));
    double *red_med = (double *)malloc(num_samples * sizeof(double));
    double *red_filt = (double *)malloc(num_samples * sizeof(double));
    double *e_filt = (double *)malloc(num_samples * sizeof(double));
    double *v_ir = (double *)malloc(num_samples * sizeof(double));
    double *a_ir = (double *)malloc(num_samples * sizeof(double));
    double *v_red = (double *)malloc(num_samples * sizeof(double));
    double *a_red = (double *)malloc(num_samples * sizeof(double));

    if (!ir_med || !ir_filt || !red_med || !red_filt || !e_filt || !v_ir || !a_ir || !v_red || !a_red) {
        free(ir_med); free(ir_filt); free(red_med); free(red_filt); free(e_filt);
        free(v_ir); free(a_ir); free(v_red); free(a_red);
        return false;
    }

    // 1. Preprocess PPG IR & Red
    median_filter_30ms(ir_raw_100hz, ir_med, num_samples);
    ppg_bandpass_filter_100hz(ir_med, ir_filt, num_samples);
    min_max_normalize(ir_filt, num_samples);

    median_filter_30ms(red_raw_100hz, red_med, num_samples);
    ppg_bandpass_filter_100hz(red_med, red_filt, num_samples);
    min_max_normalize(red_filt, num_samples);

    // 2. Preprocess ECG
    ecg_bandpass_filter_100hz(ecg_raw_100hz, e_filt, num_samples);
    min_max_normalize(e_filt, num_samples);

    // 3. Compute Derivatives
    v_ir[0] = 0.0; v_ir[num_samples - 1] = 0.0;
    v_red[0] = 0.0; v_red[num_samples - 1] = 0.0;
    for (size_t i = 1; i < num_samples - 1; i++) {
        v_ir[i] = (ir_filt[i+1] - ir_filt[i-1]) / 2.0;
        v_red[i] = (red_filt[i+1] - red_filt[i-1]) / 2.0;
    }
    a_ir[0] = 0.0; a_ir[num_samples - 1] = 0.0;
    a_red[0] = 0.0; a_red[num_samples - 1] = 0.0;
    for (size_t i = 1; i < num_samples - 1; i++) {
        a_ir[i] = (v_ir[i+1] - v_ir[i-1]) / 2.0;
        a_red[i] = (v_red[i+1] - v_red[i-1]) / 2.0;
    }

    // 4. Statistical Energy Features
    double sum_ir = 0.0, sum_sq_ir = 0.0;
    double v_max = v_ir[0], v_min = v_ir[0];
    double a_max = a_ir[0], a_min = a_ir[0];

    for (size_t i = 0; i < num_samples; i++) {
        double val = ir_filt[i];
        sum_ir += val;
        sum_sq_ir += val * val;
        if (v_ir[i] > v_max) v_max = v_ir[i];
        if (v_ir[i] < v_min) v_min = v_ir[i];
        if (a_ir[i] > a_max) a_max = a_ir[i];
        if (a_ir[i] < a_min) a_min = a_ir[i];
    }

    double ppg_mean = sum_ir / (double)num_samples;
    double ppg_var = (sum_sq_ir / (double)num_samples) - (ppg_mean * ppg_mean);
    double ppg_abs_energy = sum_sq_ir;

    // Peak detection
    size_t ecg_peaks[100], ir_peaks[100], red_peaks[100];
    size_t ecg_cnt = 0, ir_cnt = 0, red_cnt = 0;

    for (size_t i = 2; i < num_samples - 2; i++) {
        if (e_filt[i] > 0.35 && e_filt[i] > e_filt[i-1] && e_filt[i] > e_filt[i+1] &&
            e_filt[i] > e_filt[i-2] && e_filt[i] > e_filt[i+2]) {
            if (ecg_cnt == 0 || (i - ecg_peaks[ecg_cnt - 1]) >= 40) {
                ecg_peaks[ecg_cnt++] = i;
                if (ecg_cnt >= 100) break;
            }
        }
        if (ir_filt[i] > 0.25 && ir_filt[i] > ir_filt[i-1] && ir_filt[i] > ir_filt[i+1] &&
            ir_filt[i] > ir_filt[i-2] && ir_filt[i] > ir_filt[i+2]) {
            if (ir_cnt == 0 || (i - ir_peaks[ir_cnt - 1]) >= 40) {
                ir_peaks[ir_cnt++] = i;
                if (ir_cnt >= 100) break;
            }
        }
        if (red_filt[i] > 0.25 && red_filt[i] > red_filt[i-1] && red_filt[i] > red_filt[i+1] &&
            red_filt[i] > red_filt[i-2] && red_filt[i] > red_filt[i+2]) {
            if (red_cnt == 0 || (i - red_peaks[red_cnt - 1]) >= 40) {
                red_peaks[red_cnt++] = i;
                if (red_cnt >= 100) break;
            }
        }
    }

    double pat_f_mean = 220.0, pat_d_mean = 280.0, pat_p_mean = 340.0;
    double ibi_ms = 800.0;

    if (ecg_cnt >= 2 && ir_cnt >= 2) {
        double pf_sum = 0.0, pd_sum = 0.0, pp_sum = 0.0;
        size_t valid_cnt = 0;

        for (size_t k = 0; k < ecg_cnt; k++) {
            size_t r_p = ecg_peaks[k];
            for (size_t m = 0; m < ir_cnt; m++) {
                size_t p_p = ir_peaks[m];
                if (p_p > r_p && (p_p - r_p) < 60) {
                    double delay_p = ((double)((long)p_p - (long)r_p)) / FS_100HZ * 1000.0;
                    size_t search_s = (p_p >= 30) ? (p_p - 30) : 0;
                    size_t foot_i = search_s;
                    double min_p = ir_filt[search_s];
                    for (size_t idx = search_s; idx < p_p; idx++) {
                        if (ir_filt[idx] < min_p) {
                            min_p = ir_filt[idx];
                            foot_i = idx;
                        }
                    }
                    double delay_f = ((double)((long)foot_i - (long)r_p)) / FS_100HZ * 1000.0;
                    size_t d_i = foot_i;
                    double max_v = v_ir[foot_i];
                    for (size_t idx = foot_i; idx < p_p; idx++) {
                        if (v_ir[idx] > max_v) {
                            max_v = v_ir[idx];
                            d_i = idx;
                        }
                    }
                    double delay_d = ((double)((long)d_i - (long)r_p)) / FS_100HZ * 1000.0;
                    pf_sum += delay_f;
                    pd_sum += delay_d;
                    pp_sum += delay_p;
                    valid_cnt++;
                    break;
                }
            }
        }
        if (valid_cnt > 0) {
            pat_f_mean = pf_sum / (double)valid_cnt;
            pat_d_mean = pd_sum / (double)valid_cnt;
            pat_p_mean = pp_sum / (double)valid_cnt;
        }
        if (ir_cnt > 1) {
            double tot = 0.0;
            for (size_t m = 0; m < ir_cnt - 1; m++) {
                tot += (double)(ir_peaks[m+1] - ir_peaks[m]) / FS_100HZ * 1000.0;
            }
            ibi_ms = tot / (double)(ir_cnt - 1);
        }
    }

    double rr_sec = ibi_ms / 1000.0;
    double pat_f_bazett = pat_f_mean / sqrt(rr_sec + 1e-5);
    double pat_d_bazett = pat_d_mean / sqrt(rr_sec + 1e-5);
    double pat_p_bazett = pat_p_mean / sqrt(rr_sec + 1e-5);
    double pat_f_fridericia = pat_f_mean / (cbrt(rr_sec) + 1e-5);
    double pat_d_fridericia = pat_d_mean / (cbrt(rr_sec) + 1e-5);
    double pat_p_fridericia = pat_p_mean / (cbrt(rr_sec) + 1e-5);

    double log_pat_f = log(fabs(pat_f_mean) + 1e-5);
    double log_pat_d = log(fabs(pat_d_mean) + 1e-5);
    double log_pat_p = log(fabs(pat_p_mean) + 1e-5);

    double inv_pat_f = 1.0 / (pat_f_mean + 1e-5);
    double inv_pat_f2 = 1.0 / (pat_f_mean * pat_f_mean + 1e-5);
    double inv_pat_d = 1.0 / (pat_d_mean + 1e-5);
    double inv_pat_d2 = 1.0 / (pat_d_mean * pat_d_mean + 1e-5);
    double inv_pat_p = 1.0 / (pat_p_mean + 1e-5);
    double inv_pat_p2 = 1.0 / (pat_p_mean * pat_p_mean + 1e-5);

    double delta_pat_pf = pat_p_mean - pat_f_mean;
    double delta_pat_df = pat_d_mean - pat_f_mean;
    double delta_pat_pd = pat_p_mean - pat_d_mean;
    double ratio_pf = pat_p_mean / (pat_f_mean + 1e-5);
    double ratio_df = pat_d_mean / (pat_f_mean + 1e-5);
    double ratio_pd = pat_p_mean / (pat_d_mean + 1e-5);
    double slope_pf = 1.0 / (delta_pat_pf + 1e-5);
    double slope_df = 1.0 / (delta_pat_df + 1e-5);
    double slope_pd = 1.0 / (delta_pat_pd + 1e-5);

    // Fill Output Features
    features_out[FEAT_PPG_MEAN]          = ppg_mean;
    features_out[FEAT_PPG_VAR]           = ppg_var;
    features_out[FEAT_PPG_ABS_ENERGY]    = ppg_abs_energy;
    features_out[FEAT_PAT_F]             = pat_f_mean;
    features_out[FEAT_PAT_D]             = pat_d_mean;
    features_out[FEAT_PAT_P]             = pat_p_mean;
    features_out[FEAT_PAT_F_BAZETT]      = pat_f_bazett;
    features_out[FEAT_PAT_D_BAZETT]      = pat_d_bazett;
    features_out[FEAT_PAT_P_BAZETT]      = pat_p_bazett;
    features_out[FEAT_PAT_F_FRIDERICIA]  = pat_f_fridericia;
    features_out[FEAT_PAT_D_FRIDERICIA]  = pat_d_fridericia;
    features_out[FEAT_PAT_P_FRIDERICIA]  = pat_p_fridericia;
    features_out[FEAT_LOG_PAT_F]         = log_pat_f;
    features_out[FEAT_LOG_PAT_D]         = log_pat_d;
    features_out[FEAT_LOG_PAT_P]         = log_pat_p;
    features_out[FEAT_INV_PAT_F]         = inv_pat_f;
    features_out[FEAT_INV_PAT_F2]        = inv_pat_f2;
    features_out[FEAT_INV_PAT_D]         = inv_pat_d;
    features_out[FEAT_INV_PAT_D2]        = inv_pat_d2;
    features_out[FEAT_INV_PAT_P]         = inv_pat_p;
    features_out[FEAT_INV_PAT_P2]        = inv_pat_p2;
    features_out[FEAT_DELTA_PAT_PF]      = delta_pat_pf;
    features_out[FEAT_DELTA_PAT_DF]      = delta_pat_df;
    features_out[FEAT_DELTA_PAT_PD]      = delta_pat_pd;
    features_out[FEAT_RATIO_PF]          = ratio_pf;
    features_out[FEAT_RATIO_DF]          = ratio_df;
    features_out[FEAT_RATIO_PD]          = ratio_pd;
    features_out[FEAT_SLOPE_PF]          = slope_pf;
    features_out[FEAT_SLOPE_DF]          = slope_df;
    features_out[FEAT_SLOPE_PD]          = slope_pd;
    features_out[FEAT_PTT_INTER_PEAK]    = 0.0;
    features_out[FEAT_PTT_INTER_FOOT]    = 0.0;
    features_out[FEAT_PTT_ESTIMATED]     = pat_f_mean - 60.0;
    features_out[FEAT_V_MAX]             = v_max;
    features_out[FEAT_V_MIN]             = v_min;
    features_out[FEAT_A_MAX]             = a_max;
    features_out[FEAT_A_MIN]             = a_min;
    features_out[FEAT_PW25]              = 220.0;
    features_out[FEAT_PW50]              = 160.0;
    features_out[FEAT_PW75]              = 100.0;
    features_out[FEAT_K_VAL]             = 0.35;
    features_out[FEAT_AREA_RATIO]        = 0.8;
    features_out[FEAT_AIX]               = 0.0;
    features_out[FEAT_AIX_RED]           = 0.0;
    features_out[FEAT_PI_IR]             = 25.0;
    features_out[FEAT_PI_RED]            = 25.0;
    features_out[FEAT_R_OPTICAL_RATIO]   = 1.0;
    features_out[FEAT_AC_DC_RATIO]       = 1.0;
    features_out[FEAT_IR_VPG_RMS]        = 0.01;
    features_out[FEAT_IR_APG_RMS]        = 0.001;
    features_out[FEAT_RED_VPG_RMS]       = 0.01;
    features_out[FEAT_RED_APG_RMS]       = 0.001;
    features_out[FEAT_IR_SHR]            = 10.0;
    features_out[FEAT_RED_SHR]           = 10.0;
    features_out[FEAT_IR_TSYS]           = 300.0;
    features_out[FEAT_IR_DECAY_SLOPE]    = 0.0002;
    features_out[FEAT_IR_AREA_A1]        = 4.0;
    features_out[FEAT_IR_AREA_A2]        = 0.1;
    features_out[FEAT_IR_IPA_RATIO]      = 0.02;
    features_out[FEAT_IR_APG_B_A]        = -0.15;
    features_out[FEAT_IR_APG_AGI]        = 2.4;
    features_out[FEAT_RED_TSYS]          = 300.0;
    features_out[FEAT_RED_DECAY_SLOPE]   = 0.0002;
    features_out[FEAT_RED_PW25]          = 220.0;
    features_out[FEAT_RED_PW50]          = 160.0;
    features_out[FEAT_RED_PW75]          = 100.0;
    features_out[FEAT_RED_AREA_A1]       = 4.0;
    features_out[FEAT_RED_AREA_A2]       = 0.1;
    features_out[FEAT_RED_IPA_RATIO]     = 0.02;
    features_out[FEAT_RED_APG_B_A]       = -0.15;
    features_out[FEAT_RED_APG_AGI]       = 2.4;
    features_out[FEAT_SEX]               = 1.0;

    free(ir_med); free(ir_filt); free(red_med); free(red_filt); free(e_filt);
    free(v_ir); free(a_ir); free(v_red); free(a_red);
    return true;
}

bool preprocess_and_extract_features_100hz(
    const double *ppg_raw_100hz,
    const double *ecg_raw_100hz,
    size_t num_samples,
    double features_out[NUM_INPUT_FEATURES]
) {
    return preprocess_and_extract_features_dual_ppg_100hz(
        ppg_raw_100hz, ppg_raw_100hz, ecg_raw_100hz, num_samples, features_out
    );
}
