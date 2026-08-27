
#include <stdio.h>
#include "ppg_bandpass_filter.h"
#include "sample_signals.h"

int main() {
    double out[SAMPLE_SIGNAL_LEN];
    ppg_filtfilt(SAMPLE_PPG_IR, out, SAMPLE_SIGNAL_LEN);
    for (int i = 0; i < 10; i++) {
        printf("C_FILT[%d] = %f", i, out[i]);
    }
    return 0;
}
