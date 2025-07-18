import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib # Used for saving the model and scaler

print("Starting model training...")

# 1. Load data
suv_car_df = pd.read_csv('suv_data.csv')

# 2. Split data
X = suv_car_df.iloc[:,[2,3]].values # Use .values to get a numpy array
y = suv_car_df.iloc[:,4].values

# Note: We train the scaler on the *entire* dataset for a final production model
# This is a common practice when the model is not being evaluated further.
# For rigorous testing, you'd fit only on the training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# 3. Scale the data
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)


# 4. Fit the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Save the model and the scaler
joblib.dump(model, 'suv_purchase_model.pkl')
joblib.dump(sc, 'scaler.pkl')

print("Model and scaler have been trained and saved successfully!")
print("Saved model as 'suv_purchase_model.pkl'")
print("Saved scaler as 'scaler.pkl'")