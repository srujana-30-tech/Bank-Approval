# 🏦 Bank Loan Approval Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a bank loan application should be **Approved** or **Rejected** using Machine Learning classification algorithms.

The project uses customer financial information such as:

* Credit Score
* Annual Income
* Debt-to-Income Ratio
* Employment Status
* Other available customer attributes

A loan approval status is created based on predefined financial conditions, and multiple Machine Learning models are trained and compared to determine their classification accuracy.

The project also uses an **Ensemble Voting Classifier** that combines predictions from multiple Machine Learning models.

---

## 🎯 Objective

The main objectives of this project are:

* Analyze a bank loan dataset.
* Perform data cleaning and preprocessing.
* Handle missing and infinite values.
* Convert categorical data into numerical form.
* Standardize the input features.
* Create a loan approval target variable.
* Train multiple Machine Learning classification models.
* Evaluate the models using accuracy and classification metrics.
* Compare the performance of different models.
* Build an ensemble model using hard voting.
* Visualize the accuracy comparison between models.

---

## 📂 Dataset

The project reads the dataset from:

```text
Bank dataset 2.csv
```

The dataset contains customer-related financial information used to determine loan approval status.

A `customer_id` column is removed during preprocessing because it is an identifier and is not required for Machine Learning prediction.

---

## 🧠 Loan Approval Logic

The `Loan_Status` target variable is created using the following conditions:

```text
Credit Score >= 650
AND
Annual Income >= 30000
AND
Debt-to-Income Ratio <= 0.4
```

If all three conditions are satisfied:

```text
Loan_Status = Approved
```

Otherwise:

```text
Loan_Status = Rejected
```

### Example

| Credit Score | Annual Income | Debt-to-Income Ratio | Loan Status |
| -----------: | ------------: | -------------------: | ----------- |
|          720 |         50000 |                 0.30 | Approved    |
|          680 |         35000 |                 0.35 | Approved    |
|          620 |         50000 |                 0.30 | Rejected    |
|          700 |         25000 |                 0.30 | Rejected    |
|          700 |         50000 |                 0.50 | Rejected    |

This rule-based target is then converted into numerical values for Machine Learning.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Handle Missing Values
   ↓
Handle Infinite Values
   ↓
Remove Unnecessary Columns
   ↓
Encode Categorical Variables
   ↓
Create Loan Status
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Train Multiple ML Models
   ↓
Evaluate Models
   ↓
Compare Accuracy
   ↓
Voting Ensemble
   ↓
Accuracy Visualization
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib

### Machine Learning Algorithms

* Support Vector Machine (SVM)
* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* Gaussian Naive Bayes
* Voting Ensemble Classifier

---

## 🔧 Data Preprocessing

### 1. Loading the Dataset

Pandas is used to load the CSV dataset.

```python
import pandas as pd

data = pd.read_csv("Bank dataset 2.csv")
```

The dataset is inspected using:

```python
data.head()
data.info()
data.describe()
data.isnull().sum()
```

---

### 2. Creating Loan Status

NumPy is used to create the target variable:

```python
data["Loan_Status"] = np.where(
    (data["credit_score"] >= 650) &
    (data["annual_income"] >= 30000) &
    (data["debt_to_income_ratio"] <= 0.4),
    "Approved",
    "Rejected"
)
```

---

### 3. Removing Customer ID

The customer ID is removed because it is only an identifier.

```python
data.drop("customer_id", axis=1, inplace=True)
```

---

### 4. Removing Unnamed Columns

Unnecessary columns beginning with `Unnamed` are removed.

```python
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
```

---

### 5. Handling Missing Values

Missing numerical values are replaced with their respective mean values.

```python
data.fillna(data.mean(numeric_only=True), inplace=True)
```

---

### 6. Handling Infinite Values

Infinite values are replaced with missing values and then handled using mean imputation.

```python
data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.fillna(data.mean(numeric_only=True), inplace=True)
```

---

### 7. Encoding Categorical Variables

Categorical values such as employment status and loan status are converted into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["employment_status"] = le.fit_transform(data["employment_status"])
data["Loan_Status"] = le.fit_transform(data["Loan_Status"])
```

---

### 8. Separating Features and Target

The input features are stored in `X`, while the loan status is stored in `y`.

```python
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]
```

---

### 9. Feature Scaling

`StandardScaler` is used to standardize the feature values.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
```

Scaling helps Machine Learning algorithms that are sensitive to differences in feature ranges.

---

### 10. Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

The project uses:

* **80%** data for training
* **20%** data for testing

`random_state=42` ensures reproducibility.

---

# 🤖 Machine Learning Models

## 1. Support Vector Machine (SVM)

SVM is used with an RBF kernel.

```python
from sklearn.svm import SVC

svm = SVC(kernel='rbf')

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
```

The model is evaluated using accuracy and a classification report.

```python
accuracy_score(y_test, svm_pred)
classification_report(y_test, svm_pred)
```

---

## 2. Decision Tree

A Decision Tree classifier is trained to classify loan applications.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
```

The accuracy is calculated using:

```python
dt_acc = accuracy_score(y_test, dt_pred)
```

---

## 3. K-Nearest Neighbors (KNN)

KNN classifies an application based on nearby training examples.

The project uses:

```python
K = 3
```

Implementation:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)
```

---

## 4. Random Forest

Random Forest uses multiple decision trees to make predictions.

The project uses:

```python
n_estimators = 100
```

Implementation:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
```

A confusion matrix is also generated:

```python
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, rf_pred))
```

---

## 5. Gaussian Naive Bayes

Gaussian Naive Bayes is another classification algorithm used for predicting loan status.

```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)
```

---

# 🤝 Ensemble Learning

The project also implements a **Voting Classifier**.

The ensemble combines:

* SVM
* Decision Tree
* Random Forest

using hard voting.

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('dt', dt),
        ('rf', rf)
    ],
    voting='hard'
)

ensemble.fit(X_train, y_train)

ensemble_pred = ensemble.predict(X_test)

ensemble_acc = accuracy_score(y_test, ensemble_pred)
```

### Hard Voting

In hard voting, each model provides a class prediction.

The class receiving the majority of votes becomes the final prediction.

For example:

```text
SVM              → Approved
Decision Tree    → Approved
Random Forest    → Rejected
```

Final ensemble prediction:

```text
Approved
```

because it received the majority vote.

---

# 📊 Model Comparison

The accuracy of all models is collected into a DataFrame.

```python
results = pd.DataFrame({
    "Model": [
        "SVM",
        "Decision Tree",
        "KNN",
        "Random Forest",
        "Naive Bayes",
        "Ensemble"
    ],
    "Accuracy": [
        accuracy_score(y_test, svm_pred),
        dt_acc,
        knn_acc,
        rf_acc,
        nb_acc,
        ensemble_acc
    ]
})
```

The results contain the following models:

| Model         |
| ------------- |
| SVM           |
| Decision Tree |
| KNN           |
| Random Forest |
| Naive Bayes   |
| Ensemble      |

The exact accuracy values depend on the dataset used when the notebook is executed.

---

# 📈 Accuracy Visualization

Matplotlib is used to create a bar chart comparing the accuracy of all Machine Learning models.

```python
plt.figure(figsize=(8,5))

plt.bar(results["Model"], results["Accuracy"])

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of ML Models")

plt.xticks(rotation=15)

plt.show()
```

The graph provides a visual comparison of the performance of the different classification models.

---

# 📋 Evaluation Metrics

The project primarily uses **Accuracy** for comparing the models.

### Accuracy

Accuracy measures the proportion of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

The project also generates a classification report for SVM containing classification metrics.

The Random Forest model additionally uses a confusion matrix to examine prediction results.

---

# 📁 Project Structure

A possible GitHub repository structure is:

```text
Bank-Loan-Approval-Prediction/
│
├── Bank.ipynb
├── Bank dataset 2.csv
├── README.md
└── requirements.txt
```

If the dataset contains private, confidential, or personally identifiable information, it should not be uploaded publicly.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

## 2. Open the Project Folder

```bash
cd Bank-Loan-Approval-Prediction
```

## 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

Or install from `requirements.txt` if provided:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

### Using Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Bank.ipynb
```

Then run the notebook cells from top to bottom.

Make sure the dataset:

```text
Bank dataset 2.csv
```

is available in the appropriate project directory.

---

# 🔍 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Data loading
* Data exploration
* Data cleaning
* Missing-value handling
* Infinite-value handling
* Categorical encoding
* Feature selection
* Feature scaling
* Train-test splitting
* Classification
* Model evaluation
* Accuracy comparison
* Confusion matrix
* Ensemble learning
* Data visualization

---

# 🎓 Learning Outcomes

Through this project, the following Machine Learning concepts are demonstrated:

1. Understanding a real-world classification problem.
2. Preparing raw data for Machine Learning.
3. Creating a target variable using defined conditions.
4. Converting categorical data into numerical data.
5. Handling missing and infinite values.
6. Standardizing numerical features.
7. Splitting data into training and testing sets.
8. Training different classification algorithms.
9. Comparing model performance.
10. Understanding ensemble voting.
11. Visualizing model accuracy.

---

# ⚠️ Important Note

This project demonstrates Machine Learning classification using a rule-based `Loan_Status` target created from the dataset's financial attributes.

Therefore, the model's predictions should be treated as a **Machine Learning project demonstration** and not as a real-world banking or financial decision system.

Actual bank loan approval systems may involve additional factors, regulations, risk assessment procedures, credit history, verification processes, and domain-specific requirements.

---

# 🚀 Future Improvements

The project can be extended by adding:

* More comprehensive exploratory data analysis (EDA)
* Additional Machine Learning algorithms
* Hyperparameter tuning
* Cross-validation
* Precision, recall, and F1-score comparison
* ROC-AUC evaluation
* Feature importance analysis
* Confusion matrices for all models
* Interactive prediction interface
* Streamlit web application
* Model saving using Joblib or Pickle
* Real-world loan approval dataset
* More advanced ensemble techniques

---

# 👩‍💻 Project Type

**Machine Learning Classification Project**

### Domain

**Banking / Financial Services**

### Task

**Loan Approval Prediction**

### Models Used

```text
SVM
Decision Tree
KNN
Random Forest
Naive Bayes
Voting Ensemble
```

### Main Evaluation Metric

```text
Accuracy
```

---

## 📌 Conclusion

This project demonstrates a complete Machine Learning workflow for predicting bank loan approval status. The dataset is cleaned and preprocessed, categorical variables are encoded, numerical features are standardized, and the data is divided into training and testing sets.

Multiple classification algorithms are trained and evaluated, including SVM, Decision Tree, KNN, Random Forest, and Naive Bayes. A Voting Ensemble is also implemented by combining SVM, Decision Tree, and Random Forest predictions.

Finally, the model accuracies are collected and visualized to provide a clear comparison of the implemented Machine Learning approaches.
