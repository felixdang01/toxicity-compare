import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def compute_rdkit_descriptors(mols):
    desc_names = [d[0] for d in Descriptors._descList]
    desc_funcs = [d[1] for d in Descriptors._descList]

    data = []
    for mol in mols:
        row = []
        for f in desc_funcs:
            try:
                row.append(f(mol))
            except Exception:
                row.append(np.nan)
        data.append(row)

    df_desc = pd.DataFrame(data, columns=desc_names)
    df_desc = df_desc.replace([np.inf, -np.inf], np.nan)
    df_desc = df_desc.dropna(axis=1)  # drop NaN columns
    return df_desc

def compute_ecfp4_bits(mols, n_bits=1024, radius=2):
    fps = []
    for mol in mols:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr)
    return np.array(fps)
