#include <stdio.h>
#include "bp_pipeline.h"
#include "sample_signals.h"

int main() {
    double feats[NUM_INPUT_FEATURES];
    bool ok = bp_extract_features(SAMPLE_PPG_RED, SAMPLE_PPG_IR, SAMPLE_ECG_LEAD_I, SAMPLE_SIGNAL_LEN, SAMPLE_IS_MALE, feats);
    if (!ok) {
        printf("ERROR extracting features\n");
        return 1;
    }
    for (int i = 0; i < NUM_INPUT_FEATURES; i++) {
        printf("FEAT[%2d] = %.8f\n", i, feats[i]);
    }
    printf("PRED_SBP = %.4f\n", predict_sbp(feats));
    printf("PRED_DBP = %.4f\n", predict_dbp(feats));
    return 0;
}
