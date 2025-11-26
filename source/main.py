# file: detect_plateaus_autothreshold.py
import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import label
import matplotlib.pyplot as plt


def auto_derivative_threshold(signal: np.ndarray, k: float = 0.5) -> float:
    """Compute adaptive derivative threshold based on signal variation, with robust fallback."""
    deriv = np.gradient(signal)
    deriv = deriv[~np.isnan(deriv)]  # drop NaNs

    if deriv.size == 0:
        return 1e-6  # nothing to process

    mad = np.median(np.abs(deriv - np.median(deriv)))
    if np.isnan(mad) or mad == 0:
        # fallback: mean absolute derivative or small fraction of signal std
        mean_abs_deriv = np.mean(np.abs(deriv))
        if mean_abs_deriv == 0 or np.isnan(mean_abs_deriv):
            fallback = max(1e-6, 1e-3 * np.std(signal))
        else:
            fallback = max(1e-6, 0.5 * mean_abs_deriv)
        return fallback

    return max(1e-6, k * mad)


def merge_plateaus(plateaus: list[tuple[int, int]], gap_threshold: int) -> list[tuple[int, int]]:
    """Merge plateaus that are separated by a gap smaller than or equal to gap_threshold."""
    if not plateaus:
        return []

    merged = []
    current_start, current_end = plateaus[0]

    for i in range(1, len(plateaus)):
        next_start, next_end = plateaus[i]
        # If the gap between current end and next start is small enough
        if next_start - current_end <= gap_threshold:
            # Extend the current plateau
            current_end = next_end
        else:
            # Gap is too big, save current and start a new one
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))
    return merged


def detect_plateaus(signal: np.ndarray,
                    smooth_window: int = 11,
                    poly_order: int = 3,
                    deriv_threshold: float | None = None,
                    min_plateau_len: int = 5,
                    adaptive_k: float = 0.5,
                    merge_gap: int = 0,
                    filter_baseline: bool = False):
    """Detect plateau regions in a 1D signal with optional adaptive threshold."""
    smoothed = savgol_filter(signal, smooth_window, poly_order)
    deriv = np.gradient(smoothed)

    if deriv_threshold is None:
        deriv_threshold = auto_derivative_threshold(smoothed, k=adaptive_k)
        print(f"[Auto] Using derivative threshold ≈ {deriv_threshold:.5f}")

    is_plateau = np.abs(deriv) < deriv_threshold
    labeled, num_features = label(is_plateau)

    raw_plateaus = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled == i)[0]
        # We initially collect all segments, then merge, then filter by length
        # This allows small fragments to join into a valid large plateau
        if len(indices) > 0:
            start, end = indices[0], indices[-1]
            raw_plateaus.append((start, end))
    
    # Sort just in case (label usually returns sorted, but good to be safe)
    raw_plateaus.sort(key=lambda x: x[0])

    # Merge nearby plateaus
    if merge_gap > 0:
        merged_plateaus = merge_plateaus(raw_plateaus, merge_gap)
    else:
        merged_plateaus = raw_plateaus

    # Calculate baseline if filtering is enabled
    baseline = np.median(signal) if filter_baseline else -np.inf

    # Filter by minimum length AFTER merging AND by baseline level
    final_plateaus = []
    for start, end in merged_plateaus:
        length = end - start + 1
        if length >= min_plateau_len:
            # Check if plateau is above baseline
            if filter_baseline:
                plateau_mean = np.mean(signal[start:end+1])
                if plateau_mean < baseline:
                    continue
            final_plateaus.append((start, end))

    return final_plateaus, smoothed, deriv_threshold


def main():
    parser = argparse.ArgumentParser(description="Detect plateaus in signal (auto threshold).")
    parser.add_argument("--file", required=True, help="Path to CSV file (no headers)")
    parser.add_argument("--minlen", type=int, default=5, help="Minimum plateau length")
    parser.add_argument("--k", type=float, default=0.5, help="Scale factor for adaptive threshold")
    parser.add_argument("--merge-gap", type=int, default=5, help="Max gap to merge consecutive plateaus")
    parser.add_argument("--above-baseline", action="store_true", help="Only keep plateaus above the signal median")
    parser.add_argument("--plot", action="store_true", help="Show plot of detected plateaus")
    parser.add_argument("--times", action="store_true", help="Output start/end times instead of indices")

    args = parser.parse_args()

    # Read headerless CSV → add [time, value]
    df = pd.read_csv(args.file, header=None, names=["time", "value"],sep=" ")

    signal = df["value"].values
    times = df["time"].values

    plateaus, smoothed, threshold = detect_plateaus(signal,
                                                    deriv_threshold=None,
                                                    min_plateau_len=args.minlen,
                                                    adaptive_k=args.k,
                                                    merge_gap=args.merge_gap,
                                                    filter_baseline=args.above_baseline)

    print(f"\nDetected plateaus (auto threshold={threshold:.5f}, merge_gap={args.merge_gap}):")
    """
    for start, end in plateaus:
        if args.times:
            print(f"Start: {times[start]}, End: {times[end]}")
        else:
            print(f"({start}, {end})")
    """

    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.plot(times, signal, label="Original")
        plt.plot(times, smoothed, label="Smoothed", alpha=0.7)
        for (start, end) in plateaus:
            plt.axvspan(times[start], times[end], color="red", alpha=0.3)
        plt.legend()
        plt.title(f"Detected Plateaus (auto threshold={threshold:.5f})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()


if __name__ == "__main__":
    main()
