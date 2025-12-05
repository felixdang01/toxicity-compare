from utils.data_preprocessing import load_and_clean_csv, make_splits
from models.pipeline_a import run_rdkit_pipeline
from models.pipeline_b import run_ecfp4_pipeline

if __name__ == "__main__":
    # Load and clean the dataset
    df = load_and_clean_csv("data.csv", smiles_col="smiles", label_col="y")
    print(f"Loaded {len(df)} molecules after cleaning.")

    # Split the dataset into train, validation, and test
    train_idx, val_idx, test_idx = make_splits(df)

    # Pipeline A: RDKit descriptors + RandomForest
    rdkit_model = run_rdkit_pipeline(df, train_idx, val_idx, test_idx)

    # Pipeline B: ECFP4 fingerprints + Logistic Regression
    ecfp_model = run_ecfp4_pipeline(df, train_idx, val_idx, test_idx)
