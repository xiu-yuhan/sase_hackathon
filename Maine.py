from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from stroker import returnPatientValueIntoMaine
# Assume:
# X is a 2D array-like structure of shape (n_samples, 11)
# y is a 1D array-like structure of shape (n_samples,)

# Example shapes:
# X.shape -> (n_samples, 11)
# y.shape -> (n_samples,)

PatientVariables = returnPatientValueIntoMaine
for patient in PatientVariables:
    print(patient.id + " " + patient.gender + " " + patient.age + " " + patient.hypertension + " " + patient.heart_disease + " " + patient.ever_married + " " + patient.work_type )


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