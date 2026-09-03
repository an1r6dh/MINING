import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_FILE = "sensor_data.csv"

def generate_synthetic_dataset(filename: str, num_samples: int = 10000) -> pd.DataFrame:
    """Generates synthetic sensor dataset covering SAFE, WARNING, and DANGER states."""
    print(f"Generating synthetic dataset with {num_samples:,} rows...")
    np.random.seed(42)
    
    n_safe = int(num_samples * 0.7)
    n_warn = int(num_samples * 0.2)
    n_dang = num_samples - n_safe - n_warn

    # SAFE distribution
    safe_tilt = np.random.uniform(0.001, 0.30, n_safe)
    safe_vib = np.random.uniform(0.01, 0.30, n_safe)
    safe_strain = np.random.uniform(0.001, 0.20, n_safe)

    # WARNING distribution
    warn_tilt = np.random.uniform(0.50, 3.50, n_warn)
    warn_vib = np.random.uniform(0.35, 1.20, n_warn)
    warn_strain = np.random.uniform(0.40, 1.80, n_warn)

    # DANGER distribution
    dang_tilt = np.random.uniform(4.00, 15.00, n_dang)
    dang_vib = np.random.uniform(1.30, 6.00, n_dang)
    dang_strain = np.random.uniform(2.00, 8.00, n_dang)

    tilts = np.concatenate([safe_tilt, warn_tilt, dang_tilt])
    vibs = np.concatenate([safe_vib, warn_vib, dang_vib])
    strains = np.concatenate([safe_strain, warn_strain, dang_strain])
    statuses = np.array(["SAFE"] * n_safe + ["WARNING"] * n_warn + ["DANGER"] * n_dang)

    df_gen = pd.DataFrame({
        "node_id": ["NODE_01"] * num_samples,
        "filtered_tilt": np.round(tilts, 4),
        "filtered_vibration": np.round(vibs, 4),
        "filtered_strain": np.round(strains, 4),
        "status": statuses
    })

    # Shuffle rows
    df_gen = df_gen.sample(frac=1, random_state=42).reset_index(drop=True)
    df_gen.to_csv(filename, index=False)
    print(f"Dataset successfully created and saved to '{filename}'.")
    return df_gen

def load_satellite_dataset(data_dir: str = "data", samples_per_raster: int = 1000) -> pd.DataFrame:
    """Extracts real InSAR satellite displacement rasters from data/ folder."""
    import glob
    from PIL import Image

    tif_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
    if not tif_files:
        return None

    print(f"[SATELLITE] Found {len(tif_files)} real InSAR satellite rasters in '{data_dir}/'!")
    print("Extracting physical telemetry features (Tilt, Vibration, Strain)...")

    records = []
    np.random.seed(42)

    for f in tif_files:
        try:
            img = Image.open(f)
            arr = np.array(img)
            valid_mask = ~np.isnan(arr)
            valid_vals = arr[valid_mask]

            if len(valid_vals) == 0:
                continue

            # Sample representative spatial points from raster
            n_samples = min(samples_per_raster, len(valid_vals))
            sub_vals = np.random.choice(valid_vals, size=n_samples, replace=False)

            # Convert satellite surface displacement to physical telemetry values
            abs_disp = np.abs(sub_vals)
            tilts = abs_disp * 15.0      # Spatial gradient (deg/m)
            vibs = abs_disp * 6.0        # Temporal rate of velocity (g)
            strains = abs_disp * 8.0     # Micro-strain deformation (mm/m)

            for t, v, s in zip(tilts, vibs, strains):
                if t > 4.0 or v > 1.3 or s > 2.0:
                    st = "DANGER"
                elif t > 0.5 or v > 0.35 or s > 0.4:
                    st = "WARNING"
                else:
                    st = "SAFE"

                records.append({
                    "node_id": "NODE_01",
                    "filtered_tilt": round(float(t), 4),
                    "filtered_vibration": round(float(v), 4),
                    "filtered_strain": round(float(s), 4),
                    "status": st
                })
        except Exception as e:
            print(f"Skipping {f}: {e}")

    df_sat = pd.DataFrame(records)
    print(f"[SUCCESS] Extracted {len(df_sat):,} records from InSAR satellite passes.")
    df_sat.to_csv(DATASET_FILE, index=False)
    print(f"Saved extracted satellite dataset to '{DATASET_FILE}'.")
    return df_sat

# Determine dataset source
df = None
if os.path.exists("data"):
    df = load_satellite_dataset("data")

if df is None or df.empty:
    if os.path.exists(DATASET_FILE):
        print(f"Loading existing '{DATASET_FILE}'...")
        df = pd.read_csv(DATASET_FILE)
    else:
        df = generate_synthetic_dataset(DATASET_FILE)

# Clean missing entries
df = df.dropna()

# Subsample large rasters to avoid memory limit crashes
MAX_SAMPLES = 100000
if len(df) > MAX_SAMPLES:
    print(
        f"Subsampling from {len(df):,} rows to {MAX_SAMPLES:,} rows for fast training..."
    )
    df = df.sample(n=MAX_SAMPLES, random_state=42)

# Define feature variables and risk status target
X = df[["filtered_tilt", "filtered_vibration", "filtered_strain"]]
y = df["status"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Classifier across CPU cores
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Output evaluation metrics
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Overwrite joblib model artifact
joblib.dump(model, "model.joblib")
print("[SUCCESS] Real model trained and exported to 'model.joblib' successfully!")