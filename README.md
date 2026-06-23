# Classification Models, Evaluation Metrics & Handling Imbalanced Data

## Project Overview

This project evaluates supervised classification models using standard performance metrics and imbalance-aware modeling techniques. The objective is to train classification models, measure their predictive quality, compare model behavior, and document the evaluation workflow clearly enough for review or submission.

The classification problem solved is binary tumor diagnosis using the Breast Cancer Wisconsin diagnostic dataset from `scikit-learn`. The target labels represent malignant and benign tumor classes, and the features describe computed characteristics of cell nuclei from digitized biopsy images.

Algorithms implemented:

- Logistic Regression
- Logistic Regression with class weighting for imbalance handling
- Decision Tree Classifier for comparison

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Workflow

1. Data Loading
2. Train-Test Split
3. Feature Scaling
4. Logistic Regression
5. Confusion Matrix
6. Classification Report
7. ROC Curve & AUC
8. Class Imbalance Handling
9. Decision Tree Comparison

## Results

The following metrics were extracted from the notebook outputs and reproduced from the same workflow:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9825 | 0.9825 | 0.9825 | 0.9825 | 0.9954 |
| Decision Tree | 0.9123 | 0.9161 | 0.9123 | 0.9130 | Not evaluated |

Logistic Regression confusion matrix:

```text
[[41, 1],
 [ 1, 71]]
```

Visual evaluation outputs:

- `images/confusion_matrix.png`
- `images/roc_curve.png`

## Repository Structure

```text
AI_ML_Task4_Classification/
├── notebook/
│   └── AI_ML_Task4_Classification_Evaluation.ipynb
├── report/
│   └── Task4_Report.pdf
├── images/
│   ├── roc_curve.png
│   └── confusion_matrix.png
├── README.md
├── requirements.txt
└── .gitignore
```

## How To Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open and execute the notebook:

```bash
jupyter notebook notebook/AI_ML_Task4_Classification_Evaluation.ipynb
```

Run the cells from top to bottom to load the dataset, train the models, generate evaluation metrics, and display the confusion matrix and ROC curve.

## Conclusion

Logistic Regression is the selected final model because it achieved stronger overall performance than the Decision Tree baseline. It reached 98.25% accuracy, weighted precision/recall/F1 of 98.25%, and a ROC-AUC score of 0.9954, indicating excellent class separation on the test set.
