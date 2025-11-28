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


def detect_plateaus(signal: np.ndarray,
                    smooth_window: int = 11,
                    poly_order: int = 3,
                    deriv_threshold: float | None = None,
                    min_plateau_len: int = 5,
                    adaptive_k: float = 0.5,
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
        if len(indices) > 0:
            start, end = indices[0], indices[-1]
            raw_plateaus.append((start, end))
    
    # Sort just in case (label usually returns sorted, but good to be safe)
    raw_plateaus.sort(key=lambda x: x[0])

    # Calculate baseline if filtering is enabled
    baseline = np.median(signal) if filter_baseline else -np.inf

    # Filter by minimum length and baseline level
    final_plateaus = []
    for start, end in raw_plateaus:
        length = end - start + 1
        if length >= min_plateau_len:
            # Check if plateau is above baseline
            if filter_baseline:
                plateau_mean = np.mean(signal[start:end+1])
                if plateau_mean < baseline:
                    continue
            final_plateaus.append((start, end))

    return final_plateaus, smoothed, deriv_threshold


def detect_plateaus_by_events(signal: np.ndarray,
                              threshold: float,
                              window_width: int = 5,
                              min_plateau_len: int = 5):
    """
    Detect plateaus based on sliding window events.
    Trigger: max(window) - min(window) > threshold
    Direction: window[-1] - window[0] (positive = rise, negative = fall)
    """
    plateaus = []
    in_plateau = False
    start_idx = 0
    
    # Iterate with sliding window
    # We need at least window_width points
    if len(signal) < window_width:
        return []

    for i in range(len(signal) - window_width + 1):
        window = signal[i : i + window_width]
        
        delta = np.max(window) - np.min(window)
        
        if delta > threshold:
            direction = window[-1] - window[0]
            
            if direction > 0: # Rise
                if not in_plateau:
                    # Start plateau after the rise (at the end of this window)
                    in_plateau = True
                    start_idx = i + window_width
            elif direction < 0: # Fall
                if in_plateau:
                    # End plateau before the fall (at the start of this window)
                    in_plateau = False
                    end_idx = i
                    
                    if end_idx - start_idx + 1 >= min_plateau_len:
                        plateaus.append((start_idx, end_idx))
    
    # Handle case where plateau continues to the end
    if in_plateau:
        end_idx = len(signal) - 1
        if end_idx - start_idx + 1 >= min_plateau_len:
            plateaus.append((start_idx, end_idx))
            
    return plateaus


def main():
    parser = argparse.ArgumentParser(description="Detect plateaus in signal.")
    parser.add_argument("--file", required=True, help="Path to CSV file (no headers)")
    parser.add_argument("--minlen", type=int, default=5, help="Minimum plateau length")
    
    # Auto-threshold args
    parser.add_argument("--k", type=float, default=0.5, help="Scale factor for adaptive threshold (auto mode)")
    parser.add_argument("--above-baseline", action="store_true", help="Only keep plateaus above the signal median (auto mode)")
    
    # Event-based args
    parser.add_argument("--delta", type=float, help="Threshold for rise/fall events (event mode)")
    parser.add_argument("--width", type=int, default=5, help="Window width for event detection (event mode)")
    
    parser.add_argument("--plot", action="store_true", help="Show plot of detected plateaus")
    parser.add_argument("--times", action="store_true", help="Output start/end times instead of indices")

    args = parser.parse_args()

    # Read headerless CSV → add [time, value]
    df = pd.read_csv(args.file, header=None, names=["time", "value"],sep=" ")

    signal = df["value"].values
    times = df["time"].values

    # Choose strategy
    if args.delta is not None:
        print(f"Using Event-Based Detection (Delta > {args.delta}, Width={args.width})")
        plateaus = detect_plateaus_by_events(signal, 
                                             threshold=args.delta, 
                                             window_width=args.width,
                                             min_plateau_len=args.minlen)
        threshold_info = f"delta={args.delta}, width={args.width}"
        smoothed = signal 
    else:
        print("Using Auto-Derivative Detection")
        plateaus, smoothed, threshold = detect_plateaus(signal,
                                                        deriv_threshold=None,
                                                        min_plateau_len=args.minlen,
                                                        adaptive_k=args.k,
                                                        filter_baseline=args.above_baseline)
        threshold_info = f"auto threshold={threshold:.5f}"


    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.plot(times, signal, label="Original")
        if args.delta is None: # Only show smoothed if we used it
            plt.plot(times, smoothed, label="Smoothed", alpha=0.7)
            
        for (start, end) in plateaus:
            plt.axvspan(times[start], times[end], color="red", alpha=0.3)
        plt.legend()
        plt.title(f"Detected Plateaus ({threshold_info})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()


if __name__ == "__main__":
    main()
