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

# Initialize baseline model
model = LogisticRegression()

# Fit on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

