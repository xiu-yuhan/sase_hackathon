#Coded by Magnus

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#from stroker import returnPatientValueIntoMaine
import stroker
import pandas as pd

# Assume:
# X is a 2D array-like structure of shape (n_samples, 11)
# y is a 1D array-like structure of shape (n_samples,)

# Example shapes:
# X.shape -> (n_samples, 11)
# y.shape -> (n_samples,)

patients = stroker.doWork()


# Assuming you have a list of patient dicts
df = pd.DataFrame([p.__dict__ for p in patients])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['gender', 'work_type', 'residence', 'smoking_status', 'ever_married'])

# Separate features and target
X = df_encoded.drop(columns=['stroke'])
y = df_encoded['stroke']




#Stratified train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)


#--------------------------------------------------------------------------------------
#Challenge 1 
# Initialize baseline model
model = LogisticRegression()

# Fit on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))

#---------------------------------------------------------------------------------------

#Challenge 2
import numpy as np

# If y is a NumPy array
unique, counts = np.unique(y, return_counts=True)
#print("Class distribution:")
#for label, count in zip(unique, counts):
    #print(f"Class {label}: {count} samples")

# OR if y is a pandas Series (preferred for pretty output)
#print("Class distribution:\n", pd.Series(y).value_counts())

from imblearn.over_sampling import SMOTE

# Create SMOTE object
smote = SMOTE(random_state=42)

# Fit and resample only the training set
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution
#print("After SMOTE:", np.bincount(y_train_resampled))

model.fit(X_train_resampled, y_train_resampled)

y_pred2 = model.predict(X_test)

#print("Accuracy:", accuracy_score(y_test, y_pred2))
#print("Classification Report:\n", classification_report(y_test, y_pred2))






y_train = y_train.astype(int)
y_test = y_test.astype(int)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure labels are integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 🔧 Fix type mismatch
    y_pred = y_pred.astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"🔍 Results for {name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("-" * 40)

    return {"Model": name, "Precision": precision, "Recall": recall, "F1-score": f1}

results = []

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
results.append(evaluate_model("Decision Tree", dt, X_train_resampled, y_train_resampled, X_test, y_test))

# Random Forest
rf = RandomForestClassifier(random_state=42)
results.append(evaluate_model("Random Forest", rf, X_train_resampled, y_train_resampled, X_test, y_test))

# XGBoost Block
# ✅ Clean XGBoost input: convert features to numeric (float)
X_train_resampled_xgb = X_train_resampled.astype(float)
X_test_xgb = X_test.astype(float)

# ✅ Make sure labels are ints
y_train_resampled_xgb = y_train_resampled.astype(int)
y_test_xgb = y_test.astype(int)

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
results.append(evaluate_model("XGBoost", xgb, X_train_resampled_xgb, y_train_resampled_xgb, X_test_xgb, y_test_xgb))



#df_results = pd.DataFrame(results)
#print("\n📊 Comparison Table:")
#print(df_results)

import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = df_encoded.corr()

# Plot heatmap
# Focus on how each feature correlates with stroke
#plt.figure(figsize=(10, 6))
#stroke_corr = correlation_matrix['stroke'].sort_values(ascending=False)
#sns.barplot(x=stroke_corr.values, y=stroke_corr.index)
#plt.title("Correlation of Features with Stroke")
#plt.show()


#Plot heatmap improved
plt.figure(figsize=(10, 6))
stroke_corr = df_encoded.corr()['stroke'].drop('stroke').sort_values(ascending=False)

sns.barplot(x=stroke_corr.values, y=stroke_corr.index)
plt.title("Correlation of Features with Stroke", fontsize=14)
plt.xlabel("Correlation")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

