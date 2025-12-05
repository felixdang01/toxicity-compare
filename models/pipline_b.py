from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from utils.feature_generation import compute_ecfp4_bits

def run_ecfp4_pipeline(df, train_idx, val_idx, test_idx):
    print("\n=== Pipeline B: ECFP4 (1024-bit) + LogisticRegression ===")
    
    X_all = compute_ecfp4_bits(df["mol"].tolist(), n_bits=1024, radius=2)
    y_all = df["y"].values

    # Standardize the features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Split data into train, validation, and test
    X_train, X_val, X_test = X_all[train_idx], X_all[val_idx], X_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)
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
