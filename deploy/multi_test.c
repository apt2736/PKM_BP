/**
 * @file multi_test.c
 * @brief Benchmark test: reads full resampled signal from CSV, runs C pipeline
 *
 * Usage: multi_test <csv_path> <is_male> [num_samples]
 * CSV format: ppg_red,ppg_ir,ecg per line
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bp_pipeline.h"
#include "bp_models.h"

#define MAX_SAMPLES 8000

int main(int argc, char *argv[]) {
    if (argc < 3) return 1;
    FILE *f = fopen(argv[1], "r");
    if (!f) return 1;

    double *red = (double*)malloc(MAX_SAMPLES * sizeof(double));
    double *ir  = (double*)malloc(MAX_SAMPLES * sizeof(double));
    double *ecg = (double*)malloc(MAX_SAMPLES * sizeof(double));
    if (!red || !ir || !ecg) { free(red); free(ir); free(ecg); return 3; }

    int n = 0;
    while (n < MAX_SAMPLES && fscanf(f, "%lf,%lf,%lf", &red[n], &ir[n], &ecg[n]) == 3) {
        n++;
    }
    fclose(f);

    if (n < 400) { free(red); free(ir); free(ecg); printf("FAIL\n"); return 2; }

    double is_male = atof(argv[2]);
    bp_prediction_result_t res;
    if (bp_predict_from_raw(red, ir, ecg, (size_t)n, is_male, &res)) {
        printf("%f,%f\n", res.sbp, res.dbp);
    } else {
        printf("FAIL\n");
    }

    free(red);
    free(ir);
    free(ecg);
    return 0;
}
