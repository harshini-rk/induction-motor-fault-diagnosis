"""
Single-file ML pipeline for induction motor fault diagnosis

Features:
- Load raw .mat files or pre-extracted feature CSV
- Preprocess signals (detrend, bandpass, notch, normalize, windowing)
- Extract time-domain and frequency-domain features
- Train baseline models (Random Forest, SVM, KNN) with cross-validation
- Evaluate on holdout test set and save best model pipeline

Usage examples:
python train_fault_diagnosis.py --mat-dir "../dataset/raw" --out-dir models
python train_fault_diagnosis.py --feature-csv "../dataset/feature_time_48k_2048_load_1.csv" --out-dir models

Requirements: numpy, scipy, pandas, scikit-learn, matplotlib, joblib
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample, detrend as _detrend, iirnotch
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib
# Use non-interactive backend so plotting works on headless servers / missing Tcl/Tk
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings


# -------------------- Data loading --------------------

def _scan_mat_vars(mat: dict):
    """Return candidate variable keys that are array-like in a MAT dict."""
    keys = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if hasattr(v, 'shape') and getattr(v, 'size', 0) > 10:
            keys.append(k)
    return keys


def load_mat_file(path: Path):
    """Load a single .mat file and return a 1D numpy signal and optional fs."""
    mat = loadmat(str(path))
    candidates = _scan_mat_vars(mat)
    if not candidates:
        raise ValueError(f"No signal-like variable in {path}")
    # Heuristic: prefer common names
    for pref in ('signal', 'sig', 'data', 'acc', 'vibration', 'current', 'x'):
        if pref in mat:
            key = pref
            break
    else:
        key = candidates[0]
    arr = np.asarray(mat[key]).squeeze()
    # flatten nested 2D columns
    if arr.ndim > 1:
        arr = arr.flatten()
    fs = None
    for candidate_fs in ('fs', 'Fs', 'fsamp', 'sr'):
        if candidate_fs in mat:
            try:
                fs = int(np.asarray(mat[candidate_fs]).squeeze())
            except Exception:
                fs = None
    return arr.astype(float), fs


def load_mat_folder(folder: Path):
    rows = []
    for p in sorted(folder.glob('*.mat')):
        sig, fs = load_mat_file(p)
        label = infer_label_from_filename(p.name)
        rows.append({'signal': sig, 'fs': fs, 'label': label, 'path': str(p)})
    return pd.DataFrame(rows)


def infer_label_from_filename(fname: str):
    u = fname.upper()
    if 'NORMAL' in u or 'TIME_NORMAL' in u or 'T_' in u:
        return 'healthy'
    if u.startswith('B') or 'B_' in u or 'BEARING' in u:
        return 'bearing'
    if 'IR' in u:
        return 'inner_race'
    if 'OR' in u:
        return 'outer_race'
    if 'RB' in u or 'BROKEN' in u:
        return 'broken_bar'
    # Fallback: try to find keywords
    for token in ('HEALTH', 'H', 'NORMAL'):
        if token in u:
            return 'healthy'
    return 'unknown'


# -------------------- Preprocessing --------------------

def detrend(signal: np.ndarray):
    return _detrend(signal)


def bandpass_filter(signal: np.ndarray, fs: int, low: float = 5.0, high: float = 10000.0, order: int = 4):
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: int, freq: float = 50.0, Q: float = 30.0):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)


def normalize(signal: np.ndarray):
    m = np.mean(signal)
    s = np.std(signal) if np.std(signal) > 0 else 1.0
    return (signal - m) / s


def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int):
    if orig_fs == target_fs:
        return signal
    length = int(len(signal) * target_fs / orig_fs)
    return resample(signal, length)


def window_signal(signal: np.ndarray, window_size: int, overlap: float = 0.5):
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1
    for start in range(0, max(1, len(signal) - window_size + 1), step):
        yield signal[start:start + window_size]


# -------------------- Feature extraction --------------------

def rms(x: np.ndarray):
    return float(np.sqrt(np.mean(np.square(x))))


def peak_to_peak(x: np.ndarray):
    return float(np.max(x) - np.min(x))


def time_features(x: np.ndarray):
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'rms': rms(x),
        'skew': float(skew(x)),
        'kurtosis': float(kurtosis(x)),
        'ptp': peak_to_peak(x)
    }


def fft_features(x: np.ndarray, fs: int, n_bins: int = 10):
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mags = np.abs(X)
    if mags.sum() == 0:
        mags = mags + 1e-12
    dom_idx = int(np.argmax(mags))
    dom_freq = float(freqs[dom_idx])
    centroid = float(np.sum(freqs * mags) / np.sum(mags))
    maxf = freqs.max()
    band_edges = np.linspace(0, maxf, n_bins + 1)
    band_energies = {}
    for i in range(n_bins):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_energies[f'be_{i}'] = float(np.sum(mags[mask] ** 2))
    feats = {'dom_freq': dom_freq, 'centroid': centroid}
    feats.update(band_energies)
    return feats


def extract_features_from_signal(signal: np.ndarray, fs: int, n_bins: int = 10):
    tf = time_features(signal)
    ff = fft_features(signal, fs, n_bins=n_bins)
    feats = {**tf, **ff}
    return feats


# -------------------- ML pipeline --------------------

def build_pipeline(estimator):
    return Pipeline([('scaler', StandardScaler()), ('clf', estimator)])


def train_models(X, y):
    """Train RF, SVM, KNN (with small grids) and a Stacking ensemble built from them."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(build_pipeline(rf), {'clf__n_estimators': [100, 200], 'clf__max_depth': [None, 20]}, cv=skf, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X, y)
    results['rf'] = rf_grid.best_estimator_
    results['rf_cv'] = rf_grid.best_score_

    # SVM
    svc = SVC(probability=True, random_state=42)
    svc_grid = GridSearchCV(build_pipeline(svc), {'clf__C': [0.1, 1.0], 'clf__kernel': ['rbf']}, cv=skf, scoring='accuracy', n_jobs=-1)
    svc_grid.fit(X, y)
    results['svm'] = svc_grid.best_estimator_
    results['svm_cv'] = svc_grid.best_score_

    # KNN
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(build_pipeline(knn), {'clf__n_neighbors': [3, 5, 7]}, cv=skf, scoring='accuracy', n_jobs=-1)
    knn_grid.fit(X, y)
    results['knn'] = knn_grid.best_estimator_
    results['knn_cv'] = knn_grid.best_score_

    # Stacking ensemble using the best estimators (pipelines) as base learners
    # Note: we pass estimators as (name, estimator) tuples; the pipeline objects are compatible
    estimators = [
        ('rf', results['rf']),
        ('svm', results['svm']),
        ('knn', results['knn']),
    ]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1, passthrough=False)
    # Fit stacking on full data (it will internally do internal CV for meta-features)
    stacking.fit(X, y)
    # Evaluate stacking with cross-validation
    try:
        stacking_scores = cross_val_score(stacking, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        stacking_cv = float(stacking_scores.mean())
    except Exception:
        stacking_cv = -1.0

    results['stacking'] = stacking
    results['stacking_cv'] = stacking_cv

    return results


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return acc, report, cm, labels


# -------------------- Utilities --------------------

def build_feature_dataframe_from_mat_folder(mat_folder: Path, window_size: int = 2048, overlap: float = 0.5, default_fs: int = 48000, n_bins: int = 10):
    df = load_mat_folder(mat_folder)
    rows = []
    for _, r in df.iterrows():
        sig = r['signal']
        fs = int(r['fs']) if r['fs'] is not None else default_fs
        sig = detrend(sig)
        try:
            sig = bandpass_filter(sig, fs, low=5.0, high=min(8000, fs//2-1))
        except Exception:
            warnings.warn(f'bandpass failed for {r.get("path")}, skipping')
        sig = normalize(sig)
        for w in window_signal(sig, int(window_size), overlap):
            feats = extract_features_from_signal(w, fs, n_bins=n_bins)
            feats['label'] = r['label']
            rows.append(feats)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_feature_dataframe_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError('CSV must contain a label column named "label"')
    return df


def plot_and_save_confusion(cm, labels, out_path):
    """Save confusion matrix plot to PNG. If plotting fails, save numeric CSV instead."""
    try:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        # Fallback: save matrix values to CSV and warn
        try:
            import numpy as _np
            csv_path = out_path.replace('.png', '.csv')
            _np.savetxt(csv_path, cm, fmt='%d', delimiter=',')
            print(f"Warning: plotting failed ({e}). Saved numeric confusion matrix to {csv_path}")
        except Exception as e2:
            print(f"Error: could not save confusion matrix plot or CSV: {e}; {e2}")


# -------------------- CLI main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-dir', type=str, default=None, help='Folder containing .mat files')
    parser.add_argument('--feature-csv', type=str, default=None, help='Optional pre-extracted feature CSV')
    parser.add_argument('--out-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--window-size', type=int, default=2048)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-bins', type=int, default=10)
    # In notebook environments (e.g. Jupyter/Colab) there may be additional
    # kernel args (like "-f <json>") that argparse doesn't recognize. Use
    # parse_known_args to ignore unknown args when running interactively.
    args, _ = parser.parse_known_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_csv:
        print('Loading features from CSV:', args.feature_csv)
        feat_df = build_feature_dataframe_from_csv(Path(args.feature_csv))
    elif args.mat_dir:
        print('Extracting features from .mat in:', args.mat_dir)
        feat_df = build_feature_dataframe_from_mat_folder(Path(args.mat_dir), window_size=args.window_size, overlap=args.overlap, n_bins=args.n_bins)
    else:
        raise ValueError('Either --mat-dir or --feature-csv must be provided')

    if feat_df.empty:
        raise RuntimeError('No features extracted. Check input data and parameters.')

    # prepare data
    feature_cols = [c for c in feat_df.columns if c != 'label']
    X = feat_df[feature_cols].values
    y = feat_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)

    print('Training models...')
    res = train_models(X_train, y_train)

    # evaluate and save models (including stacking)
    reports = {}
    models_to_eval = {k: res[k] for k in ('rf', 'svm', 'knn')}
    if 'stacking' in res:
        models_to_eval['stacking'] = res['stacking']

    labels = sorted(set(y_test))
    for name, model in models_to_eval.items():
        print(f'Evaluating {name}...')
        try:
            acc, rep, cm, cm_labels = evaluate_model(model, X_test, y_test)
            reports[name] = {'accuracy': acc, 'report': rep}
            # save confusion matrix (plot or fallback CSV)
            out_png = out_dir / f'confusion_{name}.png'
            try:
                plot_and_save_confusion(cm, labels=cm_labels, out_path=str(out_png))
            except Exception as e:
                print(f'Failed to plot confusion for {name}: {e}; saving CSV fallback')
                np.savetxt(str(out_png).replace('.png', '.csv'), cm, delimiter=',', fmt='%d')
            print(name, 'accuracy:', acc)
            print(rep)
        except Exception as e:
            print(f'Error evaluating {name}: {e}')
            continue

        # persist each trained model
        try:
            joblib.dump(model, out_dir / f'{name}_model.joblib')
        except Exception as e:
            print(f'Failed to save {name} model: {e}')

    # pick best by validation cv score stored in res (rf_cv, svm_cv, knn_cv, stacking_cv)
    candidate_scores = {n: res.get(f'{n}_cv', -1) for n in ('rf', 'svm', 'knn')}
    if 'stacking_cv' in res:
        candidate_scores['stacking'] = res.get('stacking_cv', -1)

    best_name = max(candidate_scores.keys(), key=lambda n: candidate_scores[n])
    best_model = res[best_name]
    print('Best model selected:', best_name)

    joblib.dump({'model': best_model, 'reports': reports}, out_dir / 'best_model.joblib')
    print('Saved best model to', out_dir / 'best_model.joblib')


if __name__ == '__main__':
    main()
