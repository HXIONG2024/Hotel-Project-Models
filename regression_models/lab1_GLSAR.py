import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

print("pandas version:", pd.__version__)
print("statsmodels version:", sm.__version__)
print("numpy version:", np.__version__)

class HotelRegressionGLSAR: # Changed class name to HotelRegressionGLSAR to indicate GLSAR
    """
    Performs Generalized Least Squares with AR errors (GLSAR) regression on hotel booking data and runs diagnostics.
    """
    def __init__(self, data_path, dependent_variable, features_to_drop):
        """
        Initializes the HotelRegressionGLSAR class.

        Args:
            data_path (str): Path to the dataset CSV file.
            dependent_variable (str): Name of the dependent variable column.
            features_to_drop (list): List of feature names to drop from the dataset.
        """
        self.data = pd.read_csv(data_path, dtype={"agent": "object"}) # Load dataset
        # Convert reservation_status_date to datetime
        self.data['reservation_status_date'] = pd.to_datetime(self.data['reservation_status_date'])
        # Sort the DataFrame by reservation_status_date in ascending order
        self.data = self.data.sort_values(by='reservation_status_date', ascending=True)
        self.dependent_variable = dependent_variable # Set dependent variable
        self.features_to_drop = features_to_drop # Set features to drop
        self.X = None # Initialize independent variables
        self.y = None # Initialize dependent variable
        self.model = None # Initialize regression model
        self.results = None # Initialize model results

    def prepare_data(self):
        """
        Prepares the data for regression analysis.
        Selects numeric features and drops specified columns.
        """
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns # Select numeric columns
        self.X = self.data[numeric_columns].drop( # Select independent variables
            [self.dependent_variable] + self.features_to_drop, axis=1)
        self.y = self.data[self.dependent_variable] # Select dependent variable
        self.X = sm.add_constant(self.X) # Add a constant to the independent variables for intercept
        print(f"first 5 rows of X:\n{self.X.head()}") # Print first 5 rows of X

    def fit_model(self, ar_order=1): # Modified fit_model to accept ar_order, default to 1
        """
        Fits the Multiple Linear Regression model using Generalized Least Squares with AR errors (GLSAR).
        """
        #change column setting to show all columns and rows
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print("Shape of X:", self.X.shape) # ADD THIS LINE
        print("Shape of y:", self.y.shape) # ADD THIS LINE
        print("Checking for missing values:") # ADDED
        print("Missing values in y:", np.isnan(self.y).any()) # ADDED
        print("Missing values in X:", np.isnan(self.X).any()) # ADDED
        print("Data types:") # ADDED
        print("Data type of y:", self.y.dtypes) # ADDED
        print("Data type of X:", self.X.dtypes) # ADDED
        self.model = sm.GLSAR(self.y, self.X, order=ar_order).fit() # Changed sm.GLS to sm.GLSAR and added order parameter
        self.results = self.model.summary() # Get model summary

    def print_summary(self):
        """
        Prints the summary of the regression model.
        """
        print("\nModel Summary (GLSAR):") # Added (GLSAR) to summary title to indicate GLSAR model
        print(self.results) # Print model summary

    def plot_residuals_vs_fitted(self):
        """
        Generates and saves a scatter plot of residuals versus fitted values
        to check for linearity and homoscedasticity.
        """
        fitted_vals = self.model.predict(self.X) # Predict values
        residuals = self.y - fitted_vals # Calculate residuals
        plt.figure(figsize=(10,6)) # Set figure size
        sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.3) # Create scatter plot
        plt.axhline(0, color='red', linestyle='--') # Add horizontal line at zero
        plt.xlabel('Fitted Values') # Set x-axis label
        plt.ylabel('Residuals') # Set y-axis label
        plt.title('Residuals vs Fitted Values (GLSAR)') # Added (GLSAR) to plot title
        plt.savefig('output/residuals_vs_fitted_glsar.png') # Changed output file name to indicate GLSAR
    
    def plot_residual_acf_pacf(self):
        """
        Plots ACF and PACF of the residuals to check for remaining autocorrelation after GLSAR.
        """
        residuals = self.model.resid
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(residuals, ax=axes[0], title='ACF of GLSAR Residuals')
        plot_pacf(residuals, ax=axes[1], title='PACF of GLSAR Residuals', method='ywm') # Use method='ywm' for PACF
        plt.savefig('output/acf_pacf_residuals_glsar.png') # Save the plots

    def durbin_watson_test(self):
        """
        Performs the Durbin-Watson test to check for autocorrelation of residuals.
        Prints the Durbin-Watson statistic.
        """
        residuals = self.model.resid # Get residuals
        dw = durbin_watson(residuals) # Calculate Durbin-Watson statistic
        print(f'\nDurbin-Watson statistic (GLSAR): {dw}') # Added (GLSAR) to output

    def breusch_pagan_test(self):
        """
        Performs the Breusch-Pagan test to check for heteroscedasticity.
        Prints the results of the Breusch-Pagan test.
        """
        residuals = self.model.resid # Get residuals
        bp_test = het_breuschpagan(residuals, self.model.model.exog) # Perform Breusch-Pagan test
        labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'] # Labels for results
        bp_results = dict(zip(labels, bp_test)) # Create dictionary of results
        print('\nBreusch-Pagan test results (GLSAR):') # Added (GLSAR) to output
        for key in bp_results: # Print each result
            print(f'{key}: {bp_results[key]}')

    def normality_tests(self):
        """
        Generates a Q-Q plot and performs the Shapiro-Wilk test to check
        for normality of residuals. Saves the Q-Q plot and prints Shapiro-Wilk test results.
        """
        residuals = self.model.resid # Get residuals
        sm.qqplot(residuals, line='45', fit=True) # Generate Q-Q plot
        plt.title('Q-Q Plot of Residuals (GLSAR)') # Added (GLSAR) to plot title
        plt.savefig('output/qq_plot_residuals_glsar.png') # Changed output file name to indicate GLSAR

        residuals_sample = residuals.sample(5000, random_state=1) # Sample residuals for Shapiro-Wilk test
        shapiro_test = shapiro(residuals_sample) # Perform Shapiro-Wilk test
        print(f'\nShapiro-Wilk test statistic (GLSAR): {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}') # Added (GLSAR) to output

    def calculate_vif(self):
        """
        Calculates and prints the Variance Inflation Factor (VIF) for each independent variable
        to check for multicollinearity.
        """
        vif_data = pd.DataFrame() # Initialize DataFrame for VIF data
        vif_data['Feature'] = self.X.columns # Set feature names
        vif_data['VIF'] = [variance_inflation_factor(self.X.values, i) # Calculate VIF for each feature
                           for i in range(self.X.shape[1])]
        print('\nVariance Inflation Factor (VIF) for each feature (GLSAR):') # Added (GLSAR) to output
        print(vif_data) # Print VIF data


# Configuration
DATA_PATH = 'dataset/hb_outlier_removed.csv' # Path to dataset
DEPENDENT_VARIABLE = "adr" # Dependent variable name
FEATURES_TO_DROP = [ # Features to drop to avoid multicollinearity or undefined columns
    "meal_BB",
    "market_segment_Online TA",
    "assigned_room_type_A",
    "deposit_type_Non Refund",
    "customer_type_Transient-Party",
    "arrival_date_week_number",
    "babies",
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

# Instantiate the HotelRegressionGLSAR class with configuration # Changed to HotelRegressionGLSAR
hotel_regression_glsar = HotelRegressionGLSAR(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP) # Changed to HotelRegressionGLSAR

# Prepare data for regression
hotel_regression_glsar.prepare_data() # Changed to hotel_regression_glsar

# Fit the linear regression model - using GLSAR with AR order 1 as an example
hotel_regression_glsar.fit_model(ar_order=1) # Changed to hotel_regression_glsar and added ar_order

# Print model summary
hotel_regression_glsar.print_summary() # Changed to hotel_regression_glsar

# Perform and save diagnostic plots and tests
hotel_regression_glsar.plot_residuals_vs_fitted() # Changed to hotel_regression_glsar
hotel_regression_glsar.plot_residual_acf_pacf() 
hotel_regression_glsar.durbin_watson_test() # Changed to hotel_regression_glsar
hotel_regression_glsar.breusch_pagan_test() # Changed to hotel_regression_glsar
hotel_regression_glsar.normality_tests() # Changed to hotel_regression_glsar

# Configure pandas to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
hotel_regression_glsar.calculate_vif() # Changed to hotel_regression_glsar
