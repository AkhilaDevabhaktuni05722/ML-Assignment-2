# ML Assignment - 2

## a. Problem Statement

The objective of this assignment is to implement classification machine learning
models on a common classification dataset, evaluate their performance using
multiple evaluation metrics, and demonstrate the trained models through an
interactive Streamlit application.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin Diagnostic (WDBC-compatible)

The dataset contains **569 instances and 30 numerical features**, satisfying
the assignment requirement of at least 500 instances and 12 features.

The target column is `diagnosis`:
- `0` = malignant
- `1` = benign

The dataset is a binary classification problem.

## c. GitHub Repository Link

**Add your GitHub repository link here after uploading the project:**

`https://github.com/<AkhilaDevabhaktuni05722>/<ML-Assignment-2>`

## d. Models Used

The assignment document explicitly lists these five models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest

The document later refers to six models in the comparison table. To address
that inconsistency, **SVM** has been included as the sixth model.

### Evaluation Metrics

The following metrics are calculated:
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

The exact values are generated in `model_results.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| SVM | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |


### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Provides a strong linear baseline and performs well after feature scaling. |
| Decision Tree | Easy to interpret and can model non-linear relationships, although it can overfit if not controlled. |
| KNN | Uses distances between observations and benefits from feature scaling. Its performance depends on the choice of K. |
| Naive Bayes | Fast probabilistic classifier that provides a useful baseline for the dataset. |
| Random Forest | Ensemble of decision trees that generally provides strong and stable classification performance. |
| SVM | Margin-based classifier that can perform well after feature scaling. |

### Overall Winner

Based on the generated test-set results, **Logistic Regression** has the
strongest overall F1/AUC/accuracy combination among the implemented models.

## Streamlit Features

The application provides:
1. CSV test-data upload.
2. Model-selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix.
5. Classification report.
6. Test-data preview.

## Project Structure

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── train_data.csv
├── model_results.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── svm.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Deployment

1. Upload the complete project to GitHub.
2. Open Streamlit Community Cloud.
3. Sign in with GitHub.
4. Create a new app.
5. Select this repository and the `main` branch.
6. Select `app.py`.
7. Deploy the application.

After deployment, add the live Streamlit URL to the final submission PDF.

## BITS Virtual Lab Screenshot

Run the assignment in BITS Virtual Lab and capture the required screenshot.
Insert that screenshot into the final PDF.

## Submission Checklist

- [ ] GitHub repository link works.
- [ ] All source code is uploaded.
- [ ] `requirements.txt` is uploaded.
- [ ] `README.md` is uploaded.
- [ ] `test_data.csv` is uploaded.
- [ ] All saved model files are uploaded.
- [ ] Streamlit app is deployed.
- [ ] Streamlit app opens without errors.
- [ ] CSV upload works.
- [ ] Model dropdown works.
- [ ] Evaluation metrics are displayed.
- [ ] Confusion matrix/classification report is displayed.
- [ ] BITS Virtual Lab screenshot is included.
- [ ] README content is included in the submitted PDF.
