import joblib

# Save the trained model
joblib.dump(rfc, 'crop_recommender.pkl')

# Save the label encoder (if you used one)
joblib.dump(label_encoder, 'label_encoder.pkl')