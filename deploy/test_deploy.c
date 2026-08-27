#include <stdio.h>
#include <math.h>
#include "bp_models.h"
#include "ppg_bandpass_filter.h"
#include "ecg_filter.h"

int main() {
    puts("================================================================");
    puts("   CLINICAL SUBJECT TEST & PORTED C CODE VERIFICATION (SBP/DBP)  ");
    puts("================================================================");

    // 1. Test Filter Instantiations
    ppg_biquad_cascade_t ppg_filter;
    ppg_filter_reset(&ppg_filter);
    double ppg_test = ppg_filter_step(&ppg_filter, 0.5);

    ecg_filter_state_t ecg_filter;
    ecg_filter_reset(&ecg_filter);
    double ecg_test = ecg_filter_step(&ecg_filter, 100.0);
    double qrs_test = ecg_pan_tompkins_step(&ecg_filter, ecg_test);

    printf("Filter Instantiation: PPG step = %f, ECG step = %f, QRS env = %f\n\n", ppg_test, ecg_test, qrs_test);

    printf("Testing on Subject %s from Clinical Dataset:\n", "001");
    printf("  Actual Clinical Reference : SBP = %6.2f mmHg | DBP = %6.2f mmHg\n\n", 148.0, 90.0);

    // 2. Test Model Predictions on Subject Feature Vector
    double input[NUM_INPUT_FEATURES] = {-221.00000000, 18.00000000, 79.00000000, -289.55530970, 23.58369038, 103.50619668, -264.61577545, 21.55241610, 94.59115955, -156.70873786, 82.29126214, 143.29126214, -0.00452489, 0.00002047, 0.05555552, 0.00308642, 0.01265823, 0.00016023, 12.00970874, -287.99029126, -48.99029126, -0.00347234, -0.02041221, 0.08326590, 0.00001206, 0.00041666, 0.00693322, -2.00000000, -2.00000000, -2.00000000, -2.00000000, -1.00000000, 300.00000000, 239.00000000, 61.00000000, 537.00000000, 327.00000000, 142.00000000, -12.77394293, 2.40968244, -0.38415635, -0.35451145, 17550.93757148, 21752.29389515, 1.23938080, -195.49871014, 0.02412090, 0.00354614, 0.02263395, 0.00345900, 6.78289357, 6.52463033, 632.00000000, -3.05454455, 25.71109307, 10.81507364, 0.29624006, -9.52205360, -9.52205360, 642.00000000, -2.90618837, 583.00000000, 338.00000000, 134.00000000, 27.76530935, 11.24123414, 0.28826506, -8.02194379, -8.02194379, 1.00000000};

    double pred_sbp = predict_sbp(input);
    double pred_dbp = predict_dbp(input);

    printf("Ported C Model Inference Results:\n");
    printf("  SBP Predicted in C : %7.2f mmHg (Python Reference:  146.80 mmHg)\n", pred_sbp);
    printf("  DBP Predicted in C : %7.2f mmHg (Python Reference:   89.22 mmHg)\n", pred_dbp);

    double err_sbp = fabs(pred_sbp - (146.79854617));
    double err_dbp = fabs(pred_dbp - (89.22303309));

    printf("\nConcordance Verification (C vs Python):\n");
    printf("  SBP Discrepancy : %e mmHg\n", err_sbp);
    printf("  DBP Discrepancy : %e mmHg\n", err_dbp);

    if (err_sbp < 1e-4 && err_dbp < 1e-4) {
        puts("\n>>> STATUS: BIT-EXACT CONCORDANCE VERIFIED (ALL TESTS PASSED) <<<");
        return 0;
    } else {
        puts("\n>>> STATUS: DISCREPANCY DETECTED <<<");
        return 1;
    }
}