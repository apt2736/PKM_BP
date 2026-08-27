"""
PPG Signal Normalization and Plotting Script
=============================================
This script loads "data_PPG" signals from 26 JSON files (1.json to 26.json) in the
'Cuff-less Non-invasive Blood Pressure Estimation Data Set' directory, performs signal
normalization (Min-Max or Z-Score), and generates informative visualization plots saved in the
plots directory.

Dataset features:
- 26 subjects (1.json to 26.json)
- 'data_PPG': Photoplethysmogram time-series signal (~1000 Hz sampling rate)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def load_ppg_data(data_dir: str, num_subjects: int = 26) -> Dict[int, np.ndarray]:
    """
    Loads 'data_PPG' arrays from dataset JSON files (1.json to num_subjects.json).

    Parameters
    ----------
    data_dir : str
        Directory containing the dataset JSON files.
    num_subjects : int
        Number of subjects/JSON files to load (default: 26).

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary mapping subject ID (1 to 26) to PPG signal as float NumPy array.
    """
    ppg_dict = {}
    missing_files = []

    for sub_id in range(1, num_subjects + 1):
        file_path = os.path.join(data_dir, f"{sub_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "data_PPG" in data:
                    ppg_dict[sub_id] = np.array(data["data_PPG"], dtype=np.float64)
                else:
                    print(f"Warning: 'data_PPG' key missing in {file_path}")
        else:
            missing_files.append(file_path)

    if missing_files:
        print(f"Warning: {len(missing_files)} file(s) not found: {missing_files[:3]}...")
    
    print(f"Successfully loaded PPG signals for {len(ppg_dict)} subjects.")
    return ppg_dict


def normalize_signal(
    signal: np.ndarray,
    method: str = "minmax",
    feature_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Normalizes a 1D PPG signal using Min-Max or Z-Score (Standardization).

    Parameters
    ----------
    signal : np.ndarray
        Raw input signal.
    method : str
        Normalization method: 'minmax' or 'zscore'.
    feature_range : Tuple[float, float]
        Target range for Min-Max normalization (default: (0, 1)).

    Returns
    -------
    np.ndarray
        Normalized PPG signal.
    """
    sig = np.asarray(signal, dtype=np.float64)

    if method.lower() == "minmax":
        sig_min = np.min(sig)
        sig_max = np.max(sig)
        if sig_max == sig_min:
            return np.zeros_like(sig)
        min_val, max_val = feature_range
        scaled = (sig - sig_min) / (sig_max - sig_min)
        return scaled * (max_val - min_val) + min_val

    elif method.lower() == "zscore":
        mean = np.mean(sig)
        std = np.std(sig)
        if std == 0:
            return np.zeros_like(sig)
        return (sig - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'zscore'.")


def plot_ppg_grid(
    ppg_dict: Dict[int, np.ndarray],
    norm_method: str = "minmax",
    num_samples: int = 5000,
    fs: float = 1000.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a 6x5 grid showing normalized PPG pulse waveforms for all 26 subjects.

    Parameters
    ----------
    ppg_dict : Dict[int, np.ndarray]
        Dictionary of subject PPG signals.
    norm_method : str
        Normalization method ('minmax' or 'zscore').
    num_samples : int
        Number of samples to display per subject (default: 5000 samples = 5 seconds at 1000 Hz).
    fs : float
        Sampling frequency in Hz (default: 1000.0).
    save_path : Optional[str]
        File path to save the plot figure.
    """
    subjects = sorted(ppg_dict.keys())
    n_subjects = len(subjects)

    cols = 5
    rows = int(np.ceil(n_subjects / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    time = np.arange(num_samples) / fs

    for idx, sub_id in enumerate(subjects):
        ax = axes[idx]
        raw_sig = ppg_dict[sub_id][:num_samples]
        norm_sig = normalize_signal(raw_sig, method=norm_method)

        ax.plot(time, norm_sig, color="#1f77b4", linewidth=1.2, label=f"Sub {sub_id}")
        ax.set_title(f"Subject {sub_id}", fontsize=11, fontweight="bold", pad=4)
        ax.grid(True, linestyle="--", alpha=0.5)

        if idx % cols == 0:
            ax.set_ylabel("Norm. Amp." if norm_method == "minmax" else "Z-Score", fontsize=9)
        if idx >= (rows - 1) * cols or idx >= n_subjects - cols:
            ax.set_xlabel("Time (s)", fontsize=9)

    # Hide unused subplots
    for idx in range(n_subjects, len(axes)):
        fig.delaxes(axes[idx])

    norm_title = "Min-Max Normalized [0, 1]" if norm_method == "minmax" else "Z-Score Normalized (Mean=0, Std=1)"
    fig.suptitle(f"PPG Signals Across 26 Subjects ({norm_title} - 5s Window)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved grid plot to: {save_path}")

    plt.close(fig)


def plot_raw_vs_normalized(
    ppg_dict: Dict[int, np.ndarray],
    subject_id: int = 1,
    num_samples: int = 3000,
    fs: float = 1000.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plots raw vs Min-Max normalized vs Z-score normalized PPG signal for a representative subject.

    Parameters
    ----------
    ppg_dict : Dict[int, np.ndarray]
        Dictionary of subject PPG signals.
    subject_id : int
        Subject ID to display (default: 1).
    num_samples : int
        Number of samples to display (default: 3000 = 3 seconds at 1000 Hz).
    fs : float
        Sampling frequency in Hz (default: 1000.0).
    save_path : Optional[str]
        File path to save the figure.
    """
    if subject_id not in ppg_dict:
        print(f"Subject {subject_id} not available for raw vs normalized comparison.")
        return

    raw_sig = ppg_dict[subject_id][:num_samples]
    minmax_sig = normalize_signal(raw_sig, method="minmax")
    zscore_sig = normalize_signal(raw_sig, method="zscore")
    time = np.arange(len(raw_sig)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(time, raw_sig, color="#333333", linewidth=1.2)
    axes[0].set_title(f"Subject {subject_id} - Raw PPG Signal", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Raw Amplitude", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(time, minmax_sig, color="#2ca02c", linewidth=1.2)
    axes[1].set_title(f"Subject {subject_id} - Min-Max Normalized [0, 1]", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Normalized [0, 1]", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(time, zscore_sig, color="#d62728", linewidth=1.2)
    axes[2].set_title(f"Subject {subject_id} - Z-Score Normalized (Zero Mean, Unit Std)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Z-Score", fontsize=10)
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"PPG Signal Normalization Comparison (Subject {subject_id})", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")

    plt.close(fig)


def plot_overlay_signals(
    ppg_dict: Dict[int, np.ndarray],
    norm_method: str = "minmax",
    num_samples: int = 3000,
    fs: float = 1000.0,
    save_path: Optional[str] = None
) -> None:
    """
    Overlays normalized PPG signals of all 26 subjects on a single plot.

    Parameters
    ----------
    ppg_dict : Dict[int, np.ndarray]
        Dictionary of subject PPG signals.
    norm_method : str
        Normalization method ('minmax' or 'zscore').
    num_samples : int
        Number of samples to overlay per subject (default: 3000 = 3s).
    fs : float
        Sampling frequency in Hz (default: 1000.0).
    save_path : Optional[str]
        File path to save figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    time = np.arange(num_samples) / fs

    for sub_id, sig in sorted(ppg_dict.items()):
        norm_sig = normalize_signal(sig[:num_samples], method=norm_method)
        ax.plot(time, norm_sig, alpha=0.35, linewidth=1.0, label=f"Sub {sub_id}" if sub_id <= 5 else "")

    norm_title = "Min-Max [0, 1]" if norm_method == "minmax" else "Z-Score"
    ax.set_title(f"Overlay of Normalized PPG Signals (Subjects 1-26, {norm_title})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Normalized Amplitude" if norm_method == "minmax" else "Z-Score", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved overlay plot to: {save_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize and plot data_PPG signals from Cuff-less Non-invasive Blood Pressure Estimation Data Set."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_data_dir = os.path.join(project_root, "Cuff-less Non-invasive Blood Pressure Estimation Data Set")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Path to directory containing dataset JSON files (1.json to 26.json)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=script_dir,
        help="Directory to save output plot images (default: plots/)."
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        choices=["minmax", "zscore"],
        default="minmax",
        help="Normalization method: 'minmax' [0, 1] or 'zscore' (mean 0, std 1)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of signal samples to plot per subject segment (default: 5000)."
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=1000.0,
        help="Sampling frequency of PPG signal in Hz (default: 1000.0)."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from: {args.data_dir}")
    ppg_dict = load_ppg_data(args.data_dir)

    if not ppg_dict:
        print("Error: No PPG data found. Please check data_dir path.")
        sys.exit(1)

    # 1. Grid plot for all 26 subjects (Min-Max normalized)
    grid_minmax_path = os.path.join(args.output_dir, "ppg_signals_grid_minmax.png")
    plot_ppg_grid(
        ppg_dict,
        norm_method="minmax",
        num_samples=args.num_samples,
        fs=args.fs,
        save_path=grid_minmax_path
    )

    # 2. Grid plot for all 26 subjects (Z-Score normalized)
    grid_zscore_path = os.path.join(args.output_dir, "ppg_signals_grid_zscore.png")
    plot_ppg_grid(
        ppg_dict,
        norm_method="zscore",
        num_samples=args.num_samples,
        fs=args.fs,
        save_path=grid_zscore_path
    )

    # 3. Raw vs Normalization comparison plot for Subject 1
    comparison_path = os.path.join(args.output_dir, "ppg_raw_vs_normalized_comparison.png")
    plot_raw_vs_normalized(
        ppg_dict,
        subject_id=1,
        num_samples=3000,
        fs=args.fs,
        save_path=comparison_path
    )

    # 4. Overlaid signal plot across all 26 subjects
    overlay_path = os.path.join(args.output_dir, "ppg_signals_overlay.png")
    plot_overlay_signals(
        ppg_dict,
        norm_method=args.norm_method,
        num_samples=3000,
        fs=args.fs,
        save_path=overlay_path
    )

    print(f"\nAll plots generated successfully in: {args.output_dir}")


if __name__ == "__main__":
    main()
