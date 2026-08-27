#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "bp_models.h"
#include "signal_processor.h"

// Example ESP32 Main Task / Demonstration Program
int main(void) {
    printf("=== ESP32 Cuff-Less Blood Pressure Estimation Demo ===\n\n");

    #define TEST_SAMPLES 1000 // 10 seconds at 100 Hz
    double ppg_raw[TEST_SAMPLES];
    double ecg_raw[TEST_SAMPLES];

    // Synthetic 100 Hz PPG and ECG signal generation for test demo
    for (int i = 0; i < TEST_SAMPLES; i++) {
        double t = (double)i / 100.0; // time in seconds
        // 1.1 Hz cardiac rhythm (~66 bpm)
        ppg_raw[i] = 2.0 + sin(2.0 * 3.14159 * 1.1 * t) + 0.3 * sin(4.0 * 3.14159 * 1.1 * t);
        ecg_raw[i] = (fmod(t, 0.9) < 0.05) ? 1.5 : 0.1;
    }

    double features[NUM_INPUT_FEATURES];
    bool ok = preprocess_and_extract_features_100hz(ppg_raw, ecg_raw, TEST_SAMPLES, features);

    if (!ok) {
        printf("Error: Signal preprocessing or feature extraction failed!\n");
        return 1;
    }

    printf("Extracted %d Features at 100 Hz Rate successfully.\n", NUM_INPUT_FEATURES);
    printf("  - PPG Mean       : %.4f\n", features[FEAT_PPG_MEAN]);
    printf("  - PAT_f (Foot)   : %.2f ms\n", features[FEAT_PAT_F]);
    printf("  - PAT_d (Peak dV): %.2f ms\n", features[FEAT_PAT_D]);
    printf("  - PAT_p (Peak)   : %.2f ms\n\n", features[FEAT_PAT_P]);

    // Perform C Model Inference via m2cgen generated C code
    double sbp = predict_sbp(features);
    double dbp = predict_dbp(features);
    double map = predict_map(features);

    printf("=== Estimated Blood Pressure Output (ESP32 C Inference) ===\n");
    printf("  - Systolic BP  (SBP) : %7.2f mmHg\n", sbp);
    printf("  - Diastolic BP (DBP) : %7.2f mmHg\n", dbp);
    printf("  - Mean Art. BP (MAP) : %7.2f mmHg\n", map);
    printf("============================================================\n");

    return 0;
}
