import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Load the dataset
tele_cus = pd.read_csv(r"D:\BIA\Project 1 - Churn Prediction\Telco_Customer_Churn.csv")

# Data preprocessing
# Fill missing values in 'TotalCharges' and convert to numeric
tele_cus['TotalCharges'] = pd.to_numeric(tele_cus['TotalCharges'], errors='coerce')
tele_cus['TotalCharges'].fillna(0, inplace=True)

# Convert 'Churn' to binary labels
label_encoder = LabelEncoder()
tele_cus['Churn'] = label_encoder.fit_transform(tele_cus['Churn'])

# Use Label Encoding for 'InternetService' and 'Contract'
tele_cus['InternetService'] = label_encoder.fit_transform(tele_cus['InternetService'])
tele_cus['Contract'] = label_encoder.fit_transform(tele_cus['Contract'])

# Select features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X = tele_cus[selected_features]
y = tele_cus['Churn']

# Train the Random Forest model
model = SVC(kernel='linear')
model.fit(X, y)

# Save the trained model to a file
dump(model, 'svm_model.joblib')