import os
import nbformat as nbf

def build_pipeline_gen():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title MD
    title_md = """# 🚀 Complete Blood Pressure Estimation C Pipeline Generator (`bp_pipeline`)

This notebook executes the prerequisite training and filter tuning notebooks:
1. `models/train_filter.ipynb` (PPG 4th-Order Chebyshev II + ECG 3rd-Order Butterworth + Pan-Tompkins @ 100 Hz)
2. `models/train_red_ir.ipynb` (10-second sliding windows, `ECG_I_Filtered` + Dual PPG, 72 features, LightGBM **SBP and DBP** models exported to C via `m2cgen`)

Then, it generates, compiles, and verifies the complete **`bp_pipeline`** in `deploy/`:
- `deploy/bp_pipeline.h`: Unified C pipeline interface (SBP & DBP)
- `deploy/bp_pipeline.c`: Complete C signal processing, landmark detection, 72-feature extraction, and `m2cgen` inference
- `deploy/predict_bp_example.c`: Standalone runnable demonstration program with real clinical 10-second test window
- `deploy/Makefile`: Production build automation
- **Clinical Subject Validation Test**: Validates the ported C pipeline on a subject from the dataset.
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Cell 1: Environment & Path Setup
    code_env = r"""import os
import sys
import json
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
import scipy.signal as signal

# Project root resolution
root_dir = os.path.abspath(os.getcwd())
while not os.path.exists(os.path.join(root_dir, "config", "filter_config.json")):
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        break
    root_dir = parent

config_dir = os.path.join(root_dir, "config")
plots_dir = os.path.join(root_dir, "plots")
deploy_dir = os.path.join(root_dir, "deploy")
models_dir = os.path.join(root_dir, "models")
data_dir = os.path.join(root_dir, "A dataset of simultaneous collected ECG and PPG signals")

os.makedirs(deploy_dir, exist_ok=True)
print(f"Environment initialized. Project root: {root_dir}")
"""
    cells.append(nbf.v4.new_code_cell(code_env))

    # Cell 2: Step 1 MD & Code - Run Prerequisite Notebooks
    step1_md = """## Step 1: Run Prerequisite Notebooks (`train_filter.ipynb` & `train_red_ir.ipynb`)
"""
    cells.append(nbf.v4.new_markdown_cell(step1_md))

    code_run_prereqs = r"""print("================================================================================")
print(" 1. Running models/train_filter.ipynb (PPG & ECG Filter Tuning & SOS Export)")
print("================================================================================")
filter_nb_path = os.path.join(models_dir, "train_filter.ipynb")
res_filter = subprocess.run(
    f"jupyter nbconvert --execute --inplace '{filter_nb_path}'",
    shell=True, capture_output=True, text=True
)
if res_filter.returncode != 0:
    print("Warning during train_filter execution:", res_filter.stderr[-500:])
else:
    print("Successfully executed models/train_filter.ipynb")

print("\n================================================================================")
print(" 2. Running models/train_red_ir.ipynb (Feature Extraction, Training & C Export for SBP/DBP)")
print("================================================================================")
train_nb_path = os.path.join(models_dir, "train_red_ir.ipynb")
res_train = subprocess.run(
    f"jupyter nbconvert --execute --inplace '{train_nb_path}'",
    shell=True, capture_output=True, text=True
)
if res_train.returncode != 0:
    print("Warning during train_red_ir execution:", res_train.stderr[-500:])
else:
    print("Successfully executed models/train_red_ir.ipynb")

# Verify essential files are present
required_files = [
    "lgbm_sbp.c", "lgbm_sbp.h",
    "bp_models.h", "feature_names.json",
    "ppg_bandpass_filter.h", "ppg_filter.c",
    "ecg_filter.h", "ecg_filter.c"
]

for rf in required_files:
    p = os.path.join(deploy_dir, rf)
    if not os.path.exists(p):
        src = os.path.join(config_dir, rf)
        if os.path.exists(src):
            shutil.copy2(src, p)
    print(f"  [OK] {rf:<24} ({os.path.getsize(os.path.join(deploy_dir, rf)):,} bytes)")
"""
    cells.append(nbf.v4.new_code_cell(code_run_prereqs))

    # Cell 3: Step 2 MD & Code - Generate bp_pipeline.h and bp_pipeline.c
    step2_md = """## Step 2: Generate C Blood Pressure Pipeline (`bp_pipeline.h` & `bp_pipeline.c`)
"""
    cells.append(nbf.v4.new_markdown_cell(step2_md))

    with open("deploy/bp_pipeline.h", "r") as f:
        pipeline_h_content = f.read()

    with open("deploy/bp_pipeline.c", "r") as f:
        pipeline_c_content = f.read()

    code_gen_pipeline = (
        '# 1. Write deploy/bp_pipeline.h\n'
        'h_code = ' + repr(pipeline_h_content) + '\n'
        'with open(os.path.join(deploy_dir, "bp_pipeline.h"), "w") as f:\n'
        '    f.write(h_code)\n'
        'print(f"Generated header: {os.path.join(deploy_dir, \'bp_pipeline.h\')}")\n\n'
        '# 2. Write deploy/bp_pipeline.c\n'
        'c_code = ' + repr(pipeline_c_content) + '\n'
        'with open(os.path.join(deploy_dir, "bp_pipeline.c"), "w") as f:\n'
        '    f.write(c_code)\n'
        'print(f"Generated implementation: {os.path.join(deploy_dir, \'bp_pipeline.c\')}")\n'
    )
    cells.append(nbf.v4.new_code_cell(code_gen_pipeline))

    # Cell 4: Step 3 MD & Code - Generate predict_bp_example.c & sample_signals.h & Makefile
    step3_md = """## Step 3: Generate Standalone Executable Example (`predict_bp_example.c` & `Makefile`)
"""
    cells.append(nbf.v4.new_markdown_cell(step3_md))

    with open("deploy/sample_signals.h", "r") as f:
        sample_signals_content = f.read()

    with open("deploy/predict_bp_example.c", "r") as f:
        example_c_content = f.read()

    with open("deploy/Makefile", "r") as f:
        makefile_content = f.read()

    code_gen_example = (
        '# 1. Write deploy/sample_signals.h\n'
        'with open(os.path.join(deploy_dir, "sample_signals.h"), "w") as f:\n'
        '    f.write(' + repr(sample_signals_content) + ')\n'
        'print(f"Generated sample signals header: {os.path.join(deploy_dir, \'sample_signals.h\')}")\n\n'
        '# 2. Write deploy/predict_bp_example.c\n'
        'with open(os.path.join(deploy_dir, "predict_bp_example.c"), "w") as f:\n'
        '    f.write(' + repr(example_c_content) + ')\n'
        'print(f"Generated standalone example: {os.path.join(deploy_dir, \'predict_bp_example.c\')}")\n\n'
        '# 3. Write deploy/Makefile\n'
        'with open(os.path.join(deploy_dir, "Makefile"), "w") as f:\n'
        '    f.write(' + repr(makefile_content) + ')\n'
        'print(f"Generated Makefile: {os.path.join(deploy_dir, \'Makefile\')}")\n'
    )
    cells.append(nbf.v4.new_code_cell(code_gen_example))

    # Cell 5: Step 4 MD & Code - Compile & Run Pipeline Verification on Subject
    step4_md = """## Step 4: Compile with GCC & Test Ported C Code on a Clinical Subject
"""
    cells.append(nbf.v4.new_markdown_cell(step4_md))

    code_compile_verify = r"""# 1. Compile and run test suite with make in deploy/
res_make = subprocess.run(f"make -C '{deploy_dir}' clean && make -C '{deploy_dir}' test", shell=True, capture_output=True, text=True)
print(res_make.stdout)
if res_make.returncode != 0:
    print("Compilation/Test Error Output:\n", res_make.stderr)
    raise RuntimeError(f"Makefile compilation/testing failed with exit code {res_make.returncode}")
"""
    cells.append(nbf.v4.new_code_cell(code_compile_verify))

    nb['cells'] = cells

    os.makedirs("models", exist_ok=True)
    out_path = os.path.abspath("models/pipeline_gen.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated notebook: {out_path}")

if __name__ == "__main__":
    build_pipeline_gen()
