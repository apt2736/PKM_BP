/**
 * @file predict_bp_example.c
 * @brief Complete ANSI C Demonstration Program for Cuff-Less Blood Pressure Prediction (SBP 60s Model)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "bp_pipeline.h"
#include "sample_signals.h"

static void print_banner(const char *title) {
    printf("\n================================================================================\n");
    printf("   %s\n", title);
    printf("================================================================================\n");
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    print_banner("CUFF-LESS BLOOD PRESSURE INFERENCE ENGINE (ANSI C bp_pipeline - 60s Window)");
    printf("Firmware Target   : Standalone ANSI C Embedded Inference (Dual SBP & DBP, 60s Window)\n");
    printf("Model Architecture: Optuna-Tuned LightGBM Decision Trees (Converted to C)\n");
    printf("Input Processing  : Filtered Dataset (Bandpassed Signals with Z-Score Normalization)\n");
    printf("Feature Vector    : %d Dedicated Physiological Biomarkers\n", NUM_INPUT_FEATURES);

    const double *red_data = SAMPLE_PPG_RED;
    const double *ir_data  = SAMPLE_PPG_IR;
    const double *ecg_data = SAMPLE_ECG_LEAD_I;
    size_t data_length     = SAMPLE_SIGNAL_LEN;
    double is_male         = SAMPLE_IS_MALE;

    printf("\nTesting on Subject %s from Filtered Dataset (60.0s Window @ 100 Hz, %zu samples)...\n", SAMPLE_SUBJECT_ID, data_length);
    printf("[1/3] Z-Score standardizing input Bandpassed waveforms (mean 0.0, std 1.0)...\n");
    printf("[2/3] Extracting %d dedicated physiological features (QRS, PAT, PTT, Optical)...\n", NUM_INPUT_FEATURES);
    printf("[3/3] Executing C Regression Decision Tree Ensembles (bp_predict)...\n");

    double predicted_sbp = 0.0;
    double predicted_dbp = 0.0;
    bool success = bp_predict(red_data, ir_data, ecg_data, data_length, is_male, &predicted_sbp, &predicted_dbp);

    if (!success) {
        printf("\n>>> ERROR: Blood pressure estimation failed due to insufficient signal quality. <<<\n");
        return 1;
    }

    double predicted_map = predicted_dbp + (predicted_sbp - predicted_dbp) / 3.0;

    print_banner("ESTIMATED BLOOD PRESSURE INFERENCE RESULTS");
    printf("  Systolic Blood Pressure (SBP)  : %7.2f mmHg\n", predicted_sbp);
    printf("  Diastolic Blood Pressure (DBP) : %7.2f mmHg\n", predicted_dbp);
    printf("  Mean Arterial Pressure (MAP)   : %7.2f mmHg\n", predicted_map);
    printf("\n--- Clinical Ground Truth Comparison (Subject %s) ---\n", SAMPLE_SUBJECT_ID);
    printf("  Reference SBP : %7.2f mmHg (Prediction Error: %+.2f mmHg)\n", SAMPLE_TRUE_SBP, predicted_sbp - SAMPLE_TRUE_SBP);
    printf("  Reference DBP : %7.2f mmHg (Prediction Error: %+.2f mmHg)\n", SAMPLE_TRUE_DBP, predicted_dbp - SAMPLE_TRUE_DBP);
    printf("  Reference MAP : %7.2f mmHg (Prediction Error: %+.2f mmHg)\n", SAMPLE_TRUE_MAP, predicted_map - SAMPLE_TRUE_MAP);

    print_banner("C FIRMWARE PIPELINE EXECUTION COMPLETED SUCCESSFULLY");
    return 0;
}
