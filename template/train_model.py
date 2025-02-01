import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
crop_data = pd.read_csv('Crop_recommendation.csv')

# Prepare data
X = crop_data.drop('label', axis=1).values
y = crop_data['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=21)

# Train model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Save model and encoder
joblib.dump(rfc, 'crop_recommender.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and encoder saved successfully!")