DATASET:

## 1. input_data.mat

- Data type: cell array of size 127 × 1 (MATLAB “cell” type)
- Internal structure of each cell: double matrix, 1000 × 2

### Description of contents
- Each cell corresponds to one subject/recording (127 subjects total).
- Inside each cell:
  - Column 1: infrared PPG signal.
  - Column 2: red PPG signal.

### Technical details
- input_data has MATLAB shape (127×1) because it is a cell array:
  - input_data{1, i} is a double matrix [1000×2] for each i = 1..127.
- Data are already pre-processed:
	linear interpolation for NaN/Inf values removal
	band-pass filtering at [0.5–10 Hz]
	despike Hampel
	standardization z-score
	trimmed to 1000 samples

---

## 2. output_data.mat

- Data type: uint8 matrix
- Dimensions: 127 × 2

### Description of contents
- Column 1: reference values for systolic blood pressure (in mmHg).
- Column 2: reference values for diastolic blood pressure (in mmHg).

---

## 3. physiological_data.mat

- Data type: uint8 array of size 127 × 3

### Description of columns
1. Column 1: subject age (integer, uint8).
2. Column 2: sex (male = 1, female = 0, uint8).
3. Column 3: heart rate (beats per minute, uint8).

### Technical details
- The physiological_data{i,j} contains numeric uint8 (e.g. 30 years, 0, 70 bpm).

---

## General summary

1. input_data.mat
   - Variable: input_data
   - Cell array 127×1:
     - Each cell → double matrix [1000×2] with infrared PPG (column 1) and red PPG (column 2) signals

2. output_data.mat
   - Variable: output_data
   - Matrix 127×2 (uint8):
     - Col. 1 → systolic blood pressure
     - Col. 2 → diastolic blood pressure

3. physiological_data.mat
   - Variable: physiological_data
   - Cell array 127×3:
     - Col. 1 → age (uint8)
     - Col. 2 → sex (1=male, 0=female, uint8)
     - Col. 3 → heart rate (uint8)
---
