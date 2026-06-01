# Data Imputation Architectures: Statistical Mean & Median Feature Recovery
[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Engineering-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Strategy-Statistical%20Imputation-orange)](https://scikit-learn.org/stable/modules/impute.html)
[![Dataset](https://img.shields.io/badge/Dataset-Titanic%20Toy%20Slices-green)](./titanic_toy.csv)

## 🏗️ Project Overview
Missing features present a critical bottleneck in production machine learning workflows. While dropping records via listwise deletion is a quick fix, it frequently discards massive chunks of valuable data volume. When data missingness follows a **Missing Completely At Random (MCAR)** pattern, filling empty cells with central tendency metrics—such as the mean or median of the training set—serves as an efficient method to preserve sample size.

This repository explores the configuration and application of **Mean and Median Imputation** utilizing Scikit-Learn's stateful **`SimpleImputer`** module. Working with the **Titanic Toy Dataset** (`titanic_toy.csv`), this project provides a rigorous side-by-side diagnostic benchmark. It evaluates how replacing missing values in continuous variables (like `Age` and `Fare`) shifts the overall feature variance, distorts distribution shapes, and affects downstream machine learning estimators.

---

## 🛠️ Advanced Engineering Mechanics

### 1. The Statistical Assumptions & Variance Shrinkage
* **The MCAR Requirement:** Mean and median imputations assume that the missingness mechanism is entirely random. The fact that a value is missing must have zero correlation with any other observed or hidden characteristic in the dataset.
* **The Problem of Variance Shrinkage:** Because statistical imputation replaces every missing point with an identical, fixed average scalar, it creates an artificial concentration of data right at the center of the distribution. This shrinks the variance ($\sigma^2$), compresses the standard deviation, and distorts the natural covariance between features, which can negatively impact distance-based algorithms like K-Nearest Neighbors (KNN) or Support Vector Machines (SVM).

### 2. The Preprocessing Pipeline Architecture
The workflow isolates the training set to calculate baseline central metrics, serializes these parameters via `SimpleImputer`, and then maps them uniformly across independent data paths to track distribution shifts.


```text
                 ┌─────────────────────────────────────┐
                 │  Sparse Titanic Feature Vector      │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │     Missing Value Imputation        │
                 └─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                                                     │
        ▼                                                     ▼

 Mean Imputation                                     Median Imputation
(SimpleImputer)                                     (SimpleImputer)
        │                                                     │
        ▼                                                     ▼

 Suitable for                                      Suitable for
 Symmetric Data                                    Skewed Data
        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               ▼

                 ┌─────────────────────────────────────┐
                 │     Imputed Feature Matrix          │
                 └─────────────────────────────────────┘
```


---

## 🔬 Implementation Workflows

The source notebook `Mean-Median-Imputation-Using-ML.ipynb` runs an end-to-end data auditing and transformation sequence:

1. **Missing Variance Profiling:** Running detailed baseline summaries using Pandas `.isnull().mean()` to evaluate null rates across numeric attributes (`Age`, `Fare`).
2. **Train-Test Partition Split:** Separating the dataset before any imputation steps occur to prevent data leakage from the test distribution into the training metrics.
3. **Stateful Imputer Instantiation:** Managing parallel transformations using Scikit-Learn objects:
   * `SimpleImputer(strategy='mean')` for normally distributed continuous vectors.
   * `SimpleImputer(strategy='median')` for highly skewed numerical distributions.
4. **Distribution Shift Auditing (Crucial Phase):**
   * Generating overlayed Seaborn Kernel Density Estimate (KDE) and histogram plots to check for artificial peaks.
   * Tracking the exact mathematical change in variance before and after imputation ($\Delta \sigma^2 = \sigma^2_{\text{raw}} - \sigma^2_{\text{imputed}}$).
5. **Covariance & Correlation Checks:** Running Pearson correlation audits to verify that the relationships between features were not distorted by the imputed data points.

---

## 📊 Statistical Imputation Selection Matrix

| Imputation Strategy | Mathematical Target | Sensitivity to Outliers | Optimal Target Distribution | Post-Imputation Effect |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Imputation** | Complete Dataset Average ($\mu$) | Highly Sensitive | Pure Normal / Gaussian | Distorts distributions with heavy tails; reduces variance |
| **Median Imputation**| 50th Percentile Midpoint | Highly Robust | Significantly Skewed (e.g., `Fare`) | Preserves the shape of skewed data better than the mean |

---

## 💻 Tech Stack & Requirements
* **Language Environment:** Python 3.9+
* **Data Processing Foundations:** Pandas, NumPy
* **Pipeline Infrastructure:** Scikit-Learn (`impute.SimpleImputer`, `model_selection.train_test_split`)
* **Statistical Visualization Engines:** Matplotlib, Seaborn
* **Execution Environment:** Jupyter Notebook Runtime

---

## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/Mean-Median-Imputation-Using-ML.git](https://github.com/your-username/Mean-Median-Imputation-Using-ML.git)
    cd Mean-Median-Imputation-Using-ML
    ```
2.  **Install Production Environment Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Launch the Diagnostics Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `Mean-Median-Imputation-Using-ML.ipynb` to step through the comparative evaluations and visualize how central tendency imputations affect baseline modeling metrics.

---
