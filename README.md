# 👥 HR Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a company, using the IBM HR Analytics dataset. The project covers data cleaning, feature engineering, dimensionality reduction, class imbalance handling, and benchmarking of 8 classification models.

---

## 📊 Dataset

- **Source:** [Kaggle — IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees × 35 features
- **Target:** Attrition (Yes / No)
- **Class distribution:** 1,233 stayed (83.9%) / 237 left (16.1%) — imbalanced dataset

---

## 🔍 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Visualized attrition distribution with bar chart and donut pie chart
- Plotly interactive correlation heatmap across all numerical features
- Multi-histogram overlays comparing each feature split by attrition outcome
- Boxplots for outlier detection across numerical columns

### 2. Data Preprocessing
- Dropped constant or irrelevant columns: `EmployeeCount`, `StandardHours`, `EmployeeNumber`, `Over18`
- Removed outliers in `TotalWorkingYears` using IQR method
- Dropped rate columns (`HourlyRate`, `DailyRate`, `MonthlyRate`) after confirming no attrition signal
- Applied encoding strategy by column type:
  - **Label Encoding** → `Gender`, `Attrition`, `OverTime`
  - **Ordinal Encoding** → `BusinessTravel` (Non-Travel < Travel_Rarely < Travel_Frequently)
  - **One-Hot Encoding** → remaining categorical columns

### 3. Feature Engineering
- `RatioMangComp` — ratio of years with current manager to total years at company
- `RatioRoleComp` — ratio of years in current role to total years at company
- `hasPormotionLast5Years` — binary flag for whether employee was promoted in the last 5 years

### 4. Dimensionality Reduction & Visualization
- **PCA** (40 components) to reduce feature space before modeling
- **t-SNE** visualizations at 3 stages: raw features, after PCA, and after SMOTE — to observe class separability

### 5. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to balance the training set before model training

### 6. Model Benchmarking
Trained and evaluated 8 classification models on the same PCA-reduced, SMOTE-balanced training data:

| Model | Notes |
|---|---|
| Logistic Regression | L1 penalty (Lasso), liblinear solver |
| Support Vector Classifier (SVC) | RBF kernel, C=1, gamma=0.003 |
| K-Nearest Neighbors | k=11 |
| Gaussian Naive Bayes | Baseline probabilistic model |
| Decision Tree | max_depth=5 |
| Random Forest | 1,500 estimators, max_depth=5 |
| AdaBoost | 200 estimators on Decision Tree stumps |
| Gradient Boosting | 350 estimators, learning_rate=0.005 |
| XGBoost | 350 estimators, learning_rate=0.01 |

Each model is evaluated with: Accuracy, Recall, Precision, F1 Score, Confusion Matrix, and Classification Report.

---

## 📁 Project Structure

```
hr-attrition-prediction/
│
├── predict_employee_attrition.ipynb        # Main notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Dataset
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/hr-attrition-prediction.git
```

2. Install the required libraries
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost imbalanced-learn
```

3. Place the dataset CSV in the same folder as the notebook, then open it
```bash
jupyter notebook predict_employee_attrition.ipynb
```

4. Run all cells in order

---

## 📦 Libraries Used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Static visualizations |
| `plotly` | Interactive charts and heatmaps |
| `scikit-learn` | Preprocessing, PCA, t-SNE, all models, metrics |
| `xgboost` | XGBoost classifier |
| `imbalanced-learn` | SMOTE for class imbalance |

---

## 💡 Key Findings

- The dataset is significantly imbalanced (only ~16% attrition) — SMOTE was essential to prevent models from simply predicting "No" for every employee
- Employees who work overtime, have lower job satisfaction, or have been in the same role for many years show higher attrition rates
- Ensemble methods (Gradient Boosting, XGBoost, Random Forest) generally outperformed simpler classifiers
- **Recall is the most important metric** in this problem — a false negative (failing to identify an employee who will leave) is more costly than a false positive

---

## 👤 Author

**Khaled Abdulaziz**
Feel free to connect or leave feedback via GitHub Issues.
