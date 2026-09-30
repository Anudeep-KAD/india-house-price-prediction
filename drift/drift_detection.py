"""
drift/drift_detection.py
Detects data drift between reference (training) data and production data.
Uses statistical tests: KS-test for numerical features, chi-square for categorical.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "house_prices.csv")

NUMERICAL = [
    "BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
]
CATEGORICAL = [
    "State", "City", "Property_Type", "Furnished_Status",
    "Facing", "Owner_Type", "Availability_Status",
]
P_VALUE_THRESHOLD = 0.05


def load_reference(n=50000):
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=n, random_state=42)


def simulate_production_drift(ref: pd.DataFrame, n=5000) -> pd.DataFrame:
    """
    Simulate production data with realistic drift:
    - Size_in_SqFt shifted upward (larger flats being built)
    - BHK distribution shifted toward higher values
    - City distribution changed (new market entrants)
    - Year_Built skewed to recent
    No drift in: Security, Parking_Space, Amenities, etc.
    """
    prod = ref.sample(n=n, replace=True, random_state=99).copy().reset_index(drop=True)

    # Drift: Size_in_SqFt increased by ~15%
    prod["Size_in_SqFt"] = (prod["Size_in_SqFt"] * 1.15 + np.random.normal(0, 80, n)).clip(300, 8000).astype(int)

    # Drift: BHK shifted toward 3-4 BHK
    prod["BHK"] = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.20, 0.45, 0.25, 0.05])

    # Drift: Year_Built skewed newer
    prod["Year_Built"] = np.random.randint(2010, 2025, n)
    prod["Age_of_Property"] = 2024 - prod["Year_Built"]

    # Drift: Furnished_Status — more furnished properties
    prod["Furnished_Status"] = np.random.choice(
        ["Furnished", "Semi-Furnished", "Unfurnished"], n, p=[0.55, 0.35, 0.10]
    )

    # No drift in: Security, Parking, Nearby_Schools, Amenities, etc.
    return prod


def ks_test(ref_col: pd.Series, prod_col: pd.Series):
    stat, pvalue = stats.ks_2samp(ref_col.dropna(), prod_col.dropna())
    drifted = bool(pvalue < P_VALUE_THRESHOLD)
    return {"statistic": round(float(stat), 4), "p_value": round(float(pvalue), 4), "drifted": drifted}


def chi2_test(ref_col: pd.Series, prod_col: pd.Series):
    all_cats = sorted(set(ref_col.unique()) | set(prod_col.unique()))
    ref_counts = ref_col.value_counts().reindex(all_cats, fill_value=0).astype(float)
    prod_counts = prod_col.value_counts().reindex(all_cats, fill_value=0).astype(float)
    # Scale expected to same total as observed
    total_prod = prod_counts.sum()
    expected = ref_counts / ref_counts.sum() * total_prod
    stat, pvalue = stats.chisquare(prod_counts, f_exp=expected)
    drifted = bool(pvalue < P_VALUE_THRESHOLD)
    return {"statistic": round(float(stat), 4), "p_value": round(float(pvalue), 4), "drifted": drifted}


def detect_drift(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    results = {}

    for col in NUMERICAL:
        if col in reference.columns and col in production.columns:
            results[col] = {"type": "numerical", **ks_test(reference[col], production[col])}

    for col in CATEGORICAL:
        if col in reference.columns and col in production.columns:
            results[col] = {"type": "categorical", **chi2_test(reference[col], production[col])}

    return results


def print_report(results: dict):
    print("\n" + "=" * 65)
    print("DATA DRIFT DETECTION REPORT")
    print("=" * 65)
    print(f"{'Feature':<35} {'Type':<12} {'Stat':>8} {'p-value':>10} {'Drift?':>8}")
    print("-" * 65)

    drift_count = 0
    for feat, info in results.items():
        drifted = info["drifted"]
        if drifted:
            drift_count += 1
        flag = "⚠ YES" if drifted else "  no"
        print(f"{feat:<35} {info['type']:<12} {info['statistic']:>8.4f} {info['p_value']:>10.4f} {flag:>8}")

    print("-" * 65)
    total = len(results)
    print(f"\nSummary: {drift_count}/{total} features show statistically significant drift (p < {P_VALUE_THRESHOLD})")
    print("=" * 65)


def run():
    print("Loading reference data...")
    ref = load_reference(n=50000)
    print(f"  Reference: {len(ref):,} rows")

    print("Simulating production data with drift...")
    prod = simulate_production_drift(ref, n=5000)
    print(f"  Production: {len(prod):,} rows")

    print("Running drift detection...")
    results = detect_drift(ref, prod)
    print_report(results)

    # Save JSON report
    out_path = os.path.join(ROOT, "models", "drift_report.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDrift report saved to {out_path}")
    return results


if __name__ == "__main__":
    run()
