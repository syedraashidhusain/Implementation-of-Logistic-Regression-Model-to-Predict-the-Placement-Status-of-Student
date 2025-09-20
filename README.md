#Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Start

#### 2.Import required libraries: pandas, numpy, sklearn (for preprocessing, splitting, model, and evaluation).

#### 3.Load dataset: Read Placement_Data.csv using pandas.read_csv().

#### 4.Data preprocessing:
  a.Drop irrelevant columns: sl_no (serial number), salary (since placement is predicted before salary).
  b.Encode categorical variables (gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation, status) into numerical     values using LabelEncoder.

#### 5.Define features and target:
  a.Features X = all columns except status.
  b.Target y = status column (0 = Not Placed, 1 = Placed).

#### 6.Split dataset: Divide into training and testing sets using train_test_split with 80% training and 20% testing.

#### 7.Standardize features: Apply StandardScaler to scale the numeric values for better model convergence.

#### 8.Build Logistic Regression model:
  a.Initialize Logistic Regression with max_iter=200.
  b.Train (fit) the model using training data (X_train, y_train).

#### 9.Predict placement status: Use the trained model to predict on X_test.

#### 10.Evaluate performance:
  a.Compute accuracy using accuracy_score.
  b.Generate confusion matrix using confusion_matrix.
  c.Generate classification report using classification_report.
  d.Convert confusion matrix and report into better tabular format with pandas.DataFrame.

#### 11.End



## Program:
```python

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M syed raashid  husain
RegisterNumber: 25009038

# Logistic Regression for Placement Prediction (using Placement_Data.csv)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = pd.read_csv(r"C:\Users\israv\Downloads\Placement_Data.csv")
print("First 5 rows:\n", data.head())


# Step 2: Preprocessing
# Drop irrelevant columns (like serial number, names if present)
if "sl_no" in data.columns:
    data = data.drop("sl_no", axis=1)
if "salary" in data.columns:
    data = data.drop("salary", axis=1)   # can't use salary to predict placement

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Step 3: Separate features and target
X = data.drop("status", axis=1)   # features
y = data["status"]                # target

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Accuracy
print("\n\n\n✅ Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix as DataFrame
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=["Actual:Not Placed", "Actual:Placed"], 
                     columns=["Pred:Not Placed", "Pred:Placed"])

print("\nConfusion Matrix (better format):")
print(cm_df)

# Classification report as DataFrame
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (better format):")
print(report_df.round(2)) 
```


## Output:
<img width="737" height="680" alt="image" src="https://github.com/user-attachments/assets/fd3066fb-386b-48a5-981c-ea62e9d07184" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

