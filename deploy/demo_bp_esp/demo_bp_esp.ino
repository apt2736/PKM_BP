/*
 * ESP32 Cuff-Less Blood Pressure Estimation - Inference & Preprocessing Benchmark
 * Target Platform: ESP32 / ESP32-S3 / ESP32-C3 Microcontrollers (Arduino IDE / PlatformIO)
 * 
 * Features:
 *  - Benchmarks 100 Hz Signal Preprocessing & 29-Feature Extraction
 *  - Benchmarks m2cgen-generated LightGBM Inference for SBP, DBP, and MAP
 *  - Measures Microsecond Timing (micros()), Min/Max/Avg/StdDev over 100 Iterations
 *  - Calculates Inferences Per Second (FPS) & CPU Execution Overhead
 */

#ifdef ARDUINO
#include <Arduino.h>
#else
#include <stdio.h>
#endif

#include <math.h>
#include "bp_models.h"
#include "signal_processor.h"

#define NUM_BENCHMARK_RUNS 100
#define SAMPLE_COUNT 3000 // 30 seconds at 100 Hz

// Static buffers to prevent stack overflow on ESP32
static double ppg_raw_buffer[SAMPLE_COUNT];
static double ecg_raw_buffer[SAMPLE_COUNT];
static double extracted_features[NUM_INPUT_FEATURES];

// Timing storage buffers (microseconds)
static unsigned long prep_times[NUM_BENCHMARK_RUNS];
static unsigned long sbp_times[NUM_BENCHMARK_RUNS];
static unsigned long dbp_times[NUM_BENCHMARK_RUNS];
static unsigned long map_times[NUM_BENCHMARK_RUNS];
static unsigned long total_times[NUM_BENCHMARK_RUNS];

void generate_synthetic_signals() {
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    double t = (double)i / FS_100HZ; // time in seconds
    // Simulated 1.1 Hz PPG cardiac waveform (~66 BPM)
    ppg_raw_buffer[i] = 2.0 + sin(2.0 * M_PI * 1.1 * t) + 0.3 * sin(4.0 * M_PI * 1.1 * t);
    // Simulated ECG R-peak pulses
    ecg_raw_buffer[i] = (fmod(t, 0.9) < 0.05) ? 1.5 : 0.1;
  }
}

void compute_stats(const unsigned long *arr, int n, double &avg, double &min_v, double &max_v, double &std_dev) {
  if (n <= 0) return;
  double sum = 0.0;
  min_v = (double)arr[0];
  max_v = (double)arr[0];

  for (int i = 0; i < n; i++) {
    double val = (double)arr[i];
    sum += val;
    if (val < min_v) min_v = val;
    if (val > max_v) max_v = val;
  }
  avg = sum / n;

  double sq_diff_sum = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = (double)arr[i] - avg;
    sq_diff_sum += diff * diff;
  }
  std_dev = sqrt(sq_diff_sum / n);
}

void run_inference_benchmark() {
  Serial.println("\n=========================================================");
  Serial.println("  ESP32 BP Estimation Benchmark (100 Iterations)  ");
  Serial.println("=========================================================");
  Serial.printf("Target Sampling Rate : %.1f Hz\n", FS_100HZ);
  Serial.printf("Buffer Window Size   : %d samples (%.1f sec)\n", SAMPLE_COUNT, (double)SAMPLE_COUNT / FS_100HZ);
  Serial.printf("Feature Vector Count : %d Features\n", NUM_INPUT_FEATURES);
  Serial.printf("CPU Frequency        : %d MHz\n", getCpuFrequencyMhz());
  Serial.println("---------------------------------------------------------");

  // Warmup run
  preprocess_and_extract_features_100hz(ppg_raw_buffer, ecg_raw_buffer, SAMPLE_COUNT, extracted_features);
  predict_sbp(extracted_features);
  predict_dbp(extracted_features);
  predict_map(extracted_features);

  double last_sbp = 0, last_dbp = 0, last_map = 0;

  // Benchmark loop
  for (int run = 0; run < NUM_BENCHMARK_RUNS; run++) {
    // 1. Benchmark Signal Preprocessing & Feature Extraction
    unsigned long t0 = micros();
    bool ok = preprocess_and_extract_features_100hz(ppg_raw_buffer, ecg_raw_buffer, SAMPLE_COUNT, extracted_features);
    unsigned long t1 = micros();
    prep_times[run] = t1 - t0;

    if (!ok) {
      Serial.println("Error: Feature extraction failed during benchmark!");
      return;
    }

    // 2. Benchmark LightGBM SBP Inference
    unsigned long t2 = micros();
    last_sbp = predict_sbp(extracted_features);
    unsigned long t3 = micros();
    sbp_times[run] = t3 - t2;

    // 3. Benchmark LightGBM DBP Inference
    unsigned long t4 = micros();
    last_dbp = predict_dbp(extracted_features);
    unsigned long t5 = micros();
    dbp_times[run] = t5 - t4;

    // 4. Benchmark LightGBM MAP Inference
    unsigned long t6 = micros();
    last_map = predict_map(extracted_features);
    unsigned long t7 = micros();
    map_times[run] = t7 - t6;

    // 5. Total End-to-End Time
    total_times[run] = t7 - t0;

    delay(2); // Short Yield for ESP32 watchdog reset
  }

  // Calculate Statistics
  double prep_avg, prep_min, prep_max, prep_std;
  double sbp_avg, sbp_min, sbp_max, sbp_std;
  double dbp_avg, dbp_min, dbp_max, dbp_std;
  double map_avg, map_min, map_max, map_std;
  double total_avg, total_min, total_max, total_std;

  compute_stats(prep_times, NUM_BENCHMARK_RUNS, prep_avg, prep_min, prep_max, prep_std);
  compute_stats(sbp_times, NUM_BENCHMARK_RUNS, sbp_avg, sbp_min, sbp_max, sbp_std);
  compute_stats(dbp_times, NUM_BENCHMARK_RUNS, dbp_avg, dbp_min, dbp_max, dbp_std);
  compute_stats(map_times, NUM_BENCHMARK_RUNS, map_avg, map_min, map_max, map_std);
  compute_stats(total_times, NUM_BENCHMARK_RUNS, total_avg, total_min, total_max, total_std);

  double model_infer_avg = sbp_avg + dbp_avg + map_avg;
  double throughput_fps = 1000000.0 / total_avg;

  // Print Results
  Serial.println("\n=== INFERENCE & PREPROCESSING BENCHMARK RESULTS ===");
  Serial.printf("1. Feature Extraction (100Hz 30s) : Avg: %8.2f us (%6.2f ms) | Min: %6.0f us | Max: %6.0f us | StdDev: %5.2f us\n",
                prep_avg, prep_avg / 1000.0, prep_min, prep_max, prep_std);
  Serial.printf("2. LightGBM SBP Inference         : Avg: %8.2f us (%6.3f ms) | Min: %6.0f us | Max: %6.0f us | StdDev: %5.2f us\n",
                sbp_avg, sbp_avg / 1000.0, sbp_min, sbp_max, sbp_std);
  Serial.printf("3. LightGBM DBP Inference         : Avg: %8.2f us (%6.3f ms) | Min: %6.0f us | Max: %6.0f us | StdDev: %5.2f us\n",
                dbp_avg, dbp_avg / 1000.0, dbp_min, dbp_max, dbp_std);
  Serial.printf("4. LightGBM MAP Inference         : Avg: %8.2f us (%6.3f ms) | Min: %6.0f us | Max: %6.0f us | StdDev: %5.2f us\n",
                map_avg, map_avg / 1000.0, map_min, map_max, map_std);
  Serial.println("---------------------------------------------------------------------------------------------------------");
  Serial.printf("Total 3-Model Inference Time      : Avg: %8.2f us (%6.3f ms)\n",
                model_infer_avg, model_infer_avg / 1000.0);
  Serial.printf("TOTAL END-TO-END EXECUTION TIME   : Avg: %8.2f us (%6.2f ms) | Min: %6.0f us | Max: %6.0f us | StdDev: %5.2f us\n",
                total_avg, total_avg / 1000.0, total_min, total_max, total_std);
  Serial.printf("Estimated Execution Throughput    : %.2f Inferences / sec (FPS)\n", throughput_fps);
  Serial.println("---------------------------------------------------------------------------------------------------------");

  Serial.println("\n=== SAMPLE INFERENCE ESTIMATION OUTPUT ===");
  Serial.printf("  - Systolic BP  (SBP) : %.2f mmHg\n", last_sbp);
  Serial.printf("  - Diastolic BP (DBP) : %.2f mmHg\n", last_dbp);
  Serial.printf("  - Mean Art. BP (MAP) : %.2f mmHg\n", last_map);
  Serial.println("=========================================================\n");
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("Initializing Synthetic Signal Buffers...");
  generate_synthetic_signals();

  Serial.println("Starting Benchmark...");
  run_inference_benchmark();
}

void loop() {
  // Re-run benchmark every 10 seconds in loop
  delay(10000);
  run_inference_benchmark();
}
