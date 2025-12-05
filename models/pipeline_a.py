from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from utils.feature_generation import compute_rdkit_descriptors

def run_rdkit_pipeline(df, train_idx, val_idx, test_idx):
    print("\n=== Pipeline A: RDKit Descriptors + RandomForest ===")
    
    X_all = compute_rdkit_descriptors(df["mol"].tolist())
    y_all = df["y"].values

    # Remove low-variance features
    vt = VarianceThreshold(threshold=0.0)
    X_all = vt.fit_transform(X_all)

    # Standardize the features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Split data into train, validation, and test
    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    # Train a RandomForest model
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    def auc(split_name, X, y):
        proba = clf.predict_proba(X)[:, 1]
        score = roc_auc_score(y, proba)
        print(f"{split_name} ROC-AUC = {score:.3f}")
        return score

    auc("Train", X_train, y_train)
    auc("Valid", X_val, y_val)
    auc("Test ", X_test, y_test)

    return clf
