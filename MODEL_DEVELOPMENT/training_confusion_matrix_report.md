# Overall Final Model Training Confusion Matrix Report

Exact confusion matrix and diagnostic performance metrics for the final overall model (**M6_FINAL**) evaluated on full development data ($N=2,126$ images / $252$ patients).

## 1. Overall Training Confusion Matrix Table

| Actual Ground Truth | Predicted Malignant (Pos) | Predicted Benign (Neg) | Total Class Count |
| --- | ---: | ---: | ---: |
| **Actual Malignant** | **770 (True Positives)** | **20 (False Negatives)** | 790 |
| **Actual Benign** | **55 (False Positives)** | **1,281 (True Negatives)** | 1,336 |
| **Total** | **825** | **1,301** | **2,126** |

## 2. Derived Clinical Performance Metrics

- **Overall Training Accuracy**: **`96.47%`** (2,051 / 2,126 images)
- **Balanced Accuracy**: **`96.68%`**
- **Sensitivity (Recall)**: **`97.47%`** (770 / 790 malignant images)
- **Specificity**: **`95.88%`** (1,281 / 1,336 benign images)
- **Positive Predictive Value (PPV / Precision)**: **`93.33%`** (770 / 825 positive calls)
- **Negative Predictive Value (NPV)**: **`98.46%`** (1,281 / 1,301 negative calls)
- **F1-Score**: **`0.9536`**
