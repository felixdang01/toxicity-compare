# **Toxicity Compare**

This project compares two machine learning pipelines for toxicity prediction using molecular descriptors and fingerprint representations. The **BACE dataset** from the **DeepChem** library is used to evaluate the performance of the models.

The pipelines used are:
- **Pipeline A**: RDKit Descriptors + RandomForest
- **Pipeline B**: ECFP4 fingerprints + Logistic Regression

---

## **Project Structure**

── models/

  ├── pipeline_a.py # Pipeline A implementation (RDKit Descriptors + RandomForest)

  └── pipeline_b.py # Pipeline B implementation (ECFP4 + Logistic Regression)

 ── utils/

  ├── data_preprocessing.py # Data cleaning and preprocessing functions

  └── feature_generation.py # Feature generation functions (descriptors, fingerprints)

 ── main.py # Main script to run the models and compare results

 ── requirements.txt # Python dependencies for the project

 ── README.md # Project overview

 ── Toxicity.ipynib # Run it on Google Colab

---

## **Dataset**

The **BACE dataset** is loaded from the **DeepChem** library. It contains 1513 molecules with associated toxicity labels. The dataset is preprocessed and cleaned directly using DeepChem, and it includes both molecular features (e.g., RDKit descriptors or ECFP4 fingerprints) and toxicity labels.

---

## **Installation**

To set up the project, you need Python 3.x. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/toxicity-compare.git
    cd toxicity-compare
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install additional dependencies** if necessary:
    ```bash
    pip install wandb deepchem tensorflow scikit-learn
    ```

---

## **Usage**

1. **Log into Weights & Biases (W&B)** (Optional for tracking experiments):
    ```bash
    wandb login
    ```

2. **Run the main script**:
    ```bash
    python main.py
    ```

   This will load the **BACE dataset**, apply different feature generation methods (RDKit descriptors, ECFP4 fingerprints), train models (Random Forest, Logistic Regression), and evaluate them using ROC-AUC on train, validation, and test sets.

---

## **Model Pipelines**

### **Pipeline A: RDKit Descriptors + RandomForest**
This pipeline uses **RDKit descriptors** to represent molecules and trains a **RandomForest** classifier.

- **Train ROC-AUC**: 1.000
- **Validation ROC-AUC**: 0.872
- **Test ROC-AUC**: 0.877

### **Pipeline B: ECFP4 (1024-bit) + LogisticRegression**
This pipeline uses **ECFP4 fingerprints** to represent molecules and trains a **Logistic Regression** classifier.

- **Train ROC-AUC**: 1.000
- **Validation ROC-AUC**: 0.796
- **Test ROC-AUC**: 0.853

---

## **Warnings**

- **W&B not logged in**: If you see the message `"wandb: WARNING W&B installed but not logged in"`, you need to log into Weights & Biases:
    ```bash
    wandb login
    ```

- **TensorFlow Deprecation Warning**: A warning related to TensorFlow is shown because `experimental_relax_shapes` is deprecated. We recommend switching to `reduce_retracing` when TensorFlow updates.

- **DeepChem Deprecation Warning**: Warnings related to the usage of `MorganGenerator`. These warnings suggest upgrading to the newer version of the generator.

---

## **Contributing**

If you would like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request. Contributions are always welcome!

---

## **License**

This project is licensed under the MIT License.
