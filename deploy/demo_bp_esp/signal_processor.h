#ifndef SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSOR_H

#include <stddef.h>
#include <stdbool.h>
#include "bp_models.h"
#include "ppg_bandpass_filter.h"
#include "ecg_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FS_100HZ 100.0
#define MEDIAN_KERNEL_SIZE 3
#define NUM_FEATURES NUM_INPUT_FEATURES

// Preprocessing & Feature Extraction Function for 100 Hz PPG & ECG Buffers
// ppg_raw_100hz: Raw PPG array sampled at 100 Hz (e.g. 1000 samples for 10 seconds)
// ecg_raw_100hz: Raw ECG array sampled at 100 Hz
// num_samples  : Number of samples in buffer (e.g. 1000)
// features_out : Output array of size NUM_INPUT_FEATURES (72)
// Returns true on success, false if peak extraction failed
bool preprocess_and_extract_features_100hz(
    const double *ppg_raw_100hz,
    const double *ecg_raw_100hz,
    size_t num_samples,
    double features_out[NUM_INPUT_FEATURES]
);

// Dual-PPG (Red + IR) + ECG Preprocessing & Feature Extraction
bool preprocess_and_extract_features_dual_ppg_100hz(
    const double *red_raw_100hz,
    const double *ir_raw_100hz,
    const double *ecg_raw_100hz,
    size_t num_samples,
    double features_out[NUM_INPUT_FEATURES]
);

// Individual Filtering Functions
void median_filter_30ms(const double *input, double *output, size_t length);
void min_max_normalize(double *buffer, size_t length);
void ppg_bandpass_filter_100hz(const double *input, double *output, size_t length);
void ecg_bandpass_filter_100hz(const double *input, double *output, size_t length);

#ifdef __cplusplus
}
#endif

#endif // SIGNAL_PROCESSOR_H
