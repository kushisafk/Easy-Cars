import pickle
import numpy as np
import pandas as pd

# Load model and column structure
model = pickle.load(open("models/car_price_model.pkl", "rb"))
model_columns = pickle.load(open("models/model_columns.pkl", "rb"))

# Create empty input row with all zeros
input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0

# Manually fill required fields
input_df["model_year"] = 2018
input_df["milage"] = 45000

# Example dummy assignments (only if they exist)
if "brand_BMW" in input_df.columns:
    input_df["brand_BMW"] = 1

if "transmission_Manual" in input_df.columns:
    input_df["transmission_Manual"] = 0

# Predict (log scale)
log_prediction = model.predict(input_df)

# Convert back to dollars
price_prediction = np.exp(log_prediction)

print("Predicted price: $", round(price_prediction[0], 2))