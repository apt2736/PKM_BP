CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

SRCS = predict_bp_example.c bp_pipeline.c ppg_filter.c ecg_filter.c lgbm_sbp.c lgbm_dbp.c
TARGET = predict_bp_example

all: predict_bp_example test_deploy

predict_bp_example: $(SRCS)
	$(CC) $(CFLAGS) -I. $(SRCS) $(LDFLAGS) -o predict_bp_example

test_deploy: test_deploy.c lgbm_sbp.c lgbm_dbp.c ppg_filter.c ecg_filter.c
	$(CC) $(CFLAGS) -I. test_deploy.c lgbm_sbp.c lgbm_dbp.c ppg_filter.c ecg_filter.c $(LDFLAGS) -o test_deploy

test: all
	./test_deploy
	./predict_bp_example

clean:
	rm -f predict_bp_example test_deploy

.PHONY: all test clean
