import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift(old_df, new_df, threshold=0.05):
    drift_report = {}
    drift_detected = False

    for col in old_df.columns:
        if col not in new_df.columns:
            continue

        stat, p_value = ks_2samp(old_df[col], new_df[col])

        drift_report[col] = {
            "p_value": p_value,
            "drift": p_value < threshold
        }

        if p_value < threshold:
            drift_detected = True

    return drift_detected, drift_report
