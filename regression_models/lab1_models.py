# Language: python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import the different regression model classes from their files.
# Make sure these files are accessible in your PYTHONPATH.
from lab1_OLS import HotelRegression as HotelRegressionOLS  # [lab1_OLS.py-1](lab1_OLS.py-1-context)
from lab1_GLS import HotelRegressionGLS                      # [lab1_GLS.py](lab1_GLS.py-context)
from lab1_GLSAR import HotelRegressionGLSAR                  # [lab1_GLSAR.py](lab1_GLSAR.py-context)
from lab1_OLSHAC import HotelRegression as HotelRegressionOLSHAC  # [lab1_OLSHAC.py](lab1_OLSHAC.py-context)

# Configuration (same for all models)
DATA_PATH = 'dataset/hb_outlier_removed.csv'
DEPENDENT_VARIABLE = "adr"
FEATURES_TO_DROP = [
    "meal_BB",
    "babies",
    "market_segment_Online TA",
    "assigned_room_type_A",
    "deposit_type_Non Refund",
    "customer_type_Transient-Party",
    "arrival_date_week_number",
    "reservation_status_Check-Out",
    "reservation_status_Canceled",
    "reservation_status_No-Show",
    "reserved_room_type_A",
    "reserved_room_type_B",
    "reserved_room_type_C",
    "reserved_room_type_D",
    "reserved_room_type_E",
    "reserved_room_type_F",
    "reserved_room_type_G",
    "reserved_room_type_H",
    "reserved_room_type_L",
    "reserved_room_type_P",
    "year_2017",
    "month_12",
    "week_6",
    "distribution_channel_Direct",
    "distribution_channel_TA/TO",
    "distribution_channel_Undefined",
    "distribution_channel_Corporate",
    "distribution_channel_GDS"
]

def get_predictions(model_instance):
    # Ensure data is prepared and model is fitted
    model_instance.prepare_data()
    model_instance.fit_model()
    # Get fitted values from the underlying model
    fitted_vals = model_instance.model.predict(model_instance.X)
    return model_instance.y, fitted_vals

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    return mse, r2

# Create a dictionary to hold each model and its name.
models = {
    "OLS": HotelRegressionOLS(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP),
    "GLS": HotelRegressionGLS(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP),
    "GLSAR": HotelRegressionGLSAR(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP),
    "OLSHAC": HotelRegressionOLSHAC(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP)
}

# For GLSAR, we call fit_model with an AR order parameter (example: ar_order=3)
performance = {}

for name, model in models.items():
    if name == "GLSAR":
        # Use ar_order=1 for GLSAR as an example
        model.prepare_data()
        model.fit_model(ar_order=1)
        y_true = model.y
        y_pred = model.model.predict(model.X)
    else:
        y_true, y_pred = get_predictions(model)
    mse, r2 = evaluate_model(y_true, y_pred)
    performance[name] = {"MSE": mse, "R^2": r2}

print("Model Performance Comparison:")
for name, metrics in performance.items():
    print(f"{name}: MSE = {metrics['MSE']:.4f}, R^2 = {metrics['R^2']:.4f}")