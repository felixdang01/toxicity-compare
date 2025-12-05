import numpy as np
import pandas as pd
import deepchem as dc
from sklearn.model_selection import train_test_split
from rdkit import Chem

def load_bace_dataset():
    print("Loading BACE dataset from DeepChem...")
    tasks, datasets, transformers = dc.molnet.load_bace_classification()
    train, val, test = datasets

    def dc_to_df(dc_dataset):
        smiles = dc_dataset.ids
        y = dc_dataset.y.flatten()
        return pd.DataFrame({"smiles": smiles, "y": y})

    df_train = dc_to_df(train)
    df_val = dc_to_df(val)
    df_test = dc_to_df(test)

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print("Dataset loaded:", df.shape)
    return df

def load_and_clean_csv(path="data.csv", smiles_col="smiles", label_col="y"):
    df = load_bace_dataset()
    df = df[[smiles_col, label_col]].dropna()
    df.rename(columns={smiles_col: "smiles", label_col: "y"}, inplace=True)

    mols = []
    valid_idx = []
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)

    df = df.iloc[valid_idx].reset_index(drop=True)
    df["mol"] = mols
    df["y"] = df["y"].astype(int)
    return df

def make_splits(df, test_size=0.2, val_size=0.2, random_state=42):
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=df["y"]
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=random_state, stratify=df["y"].iloc[train_idx]
    )
    return train_idx, val_idx, test_idx
