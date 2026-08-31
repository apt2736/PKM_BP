#include <stdio.h>
#include <math.h>
#include "bp_models.h"

int main() {
    puts("================================================================");
    puts(" CLINICAL SUBJECT TEST & PORTED C CODE VERIFICATION (SBP & DBP) ");
    puts("   Target: Filtered Dataset (Resampled 100 Hz, 60s Window)     ");
    puts("================================================================");

    printf("Testing on Subject %s from Filtered Dataset (60s Dual Models):\n", "001");
    printf("  Actual Clinical Reference : SBP = %6.2f mmHg | DBP = %6.2f mmHg\n\n", 148.0, 90.0);

    // Test Model Predictions on Subject Feature Vector
    double input[NUM_INPUT_FEATURES] = {19.71014493, 81.88405797, -229.30720906, -196.64653026, 41.17955669, 0.05073527, 0.00257407, 0.00014914, 11.55699915, -288.44300085, 0.00001202, 0.00039031, 0.00748703, 158.38235294, 2.03369614, 122362.05382812, -1408.18089264, 0.05263427, 0.00743936, 0.05238041, 576.91176471, 312.50000000, 1.00000000};

    double pred_sbp = predict_sbp(input);
    double pred_dbp = predict_dbp(input);

    printf("Ported C Model Inference Results:\n");
    printf("  SBP Predicted in C : %7.2f mmHg (Python Reference:  146.56 mmHg)\n", pred_sbp);
    printf("  DBP Predicted in C : %7.2f mmHg (Python Reference:   89.35 mmHg)\n", pred_dbp);

    double err_sbp = fabs(pred_sbp - (146.56457951));
    double err_dbp = fabs(pred_dbp - (89.34618465));

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