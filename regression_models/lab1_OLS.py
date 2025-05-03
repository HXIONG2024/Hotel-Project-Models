import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

class HotelRegression:
    """
    Performs linear regression on hotel booking data and runs diagnostics.
    """
    def __init__(self, data_path, dependent_variable, features_to_drop):
        """
        Initializes the HotelRegression class.

        Args:
            data_path (str): Path to the dataset CSV file.
            dependent_variable (str): Name of the dependent variable column.
            features_to_drop (list): List of feature names to drop from the dataset.
        """
        self.data = pd.read_csv(data_path, dtype={"agent": "object"}) # Load dataset
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

    def fit_model(self):
        """
        Fits the Multiple Linear Regression model using Ordinary Least Squares (OLS).
        """
        self.model = sm.OLS(self.y, self.X).fit() # Fit OLS model
        self.results = self.model.summary() # Get model summary

    def print_summary(self):
        """
        Prints the summary of the regression model.
        """
        print("\nModel Summary:")
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
        plt.title('Residuals vs Fitted Values') # Set plot title
        plt.savefig('output/residuals_vs_fitted.png') # Save the plot

    def durbin_watson_test(self):
        """
        Performs the Durbin-Watson test to check for autocorrelation of residuals.
        Prints the Durbin-Watson statistic.
        """
        residuals = self.model.resid # Get residuals
        dw = durbin_watson(residuals) # Calculate Durbin-Watson statistic
        print(f'\nDurbin-Watson statistic: {dw}') # Print statistic

    def breusch_pagan_test(self):
        """
        Performs the Breusch-Pagan test to check for heteroscedasticity.
        Prints the results of the Breusch-Pagan test.
        """
        residuals = self.model.resid # Get residuals
        bp_test = het_breuschpagan(residuals, self.model.model.exog) # Perform Breusch-Pagan test
        labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'] # Labels for results
        bp_results = dict(zip(labels, bp_test)) # Create dictionary of results
        print('\nBreusch-Pagan test results:')
        for key in bp_results: # Print each result
            print(f'{key}: {bp_results[key]}')

    def normality_tests(self):
        """
        Generates a Q-Q plot and performs the Shapiro-Wilk test to check
        for normality of residuals. Saves the Q-Q plot and prints Shapiro-Wilk test results.
        """
        residuals = self.model.resid # Get residuals
        sm.qqplot(residuals, line='45', fit=True) # Generate Q-Q plot
        plt.title('Q-Q Plot of Residuals') # Set plot title
        plt.savefig('output/qq_plot_residuals.png') # Save Q-Q plot

        residuals_sample = residuals.sample(5000, random_state=1) # Sample residuals for Shapiro-Wilk test
        shapiro_test = shapiro(residuals_sample) # Perform Shapiro-Wilk test
        print(f'\nShapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}') # Print Shapiro-Wilk test results

    def calculate_vif(self):
        """
        Calculates and prints the Variance Inflation Factor (VIF) for each independent variable
        to check for multicollinearity.
        """
        vif_data = pd.DataFrame() # Initialize DataFrame for VIF data
        vif_data['Feature'] = self.X.columns # Set feature names
        vif_data['VIF'] = [variance_inflation_factor(self.X.values, i) # Calculate VIF for each feature
                           for i in range(self.X.shape[1])]
        print('\nVariance Inflation Factor (VIF) for each feature:')
        print(vif_data) # Print VIF data


# Configuration
DATA_PATH = 'dataset/hb_outlier_removed.csv' # Path to dataset
DEPENDENT_VARIABLE = "adr" # Dependent variable name
FEATURES_TO_DROP = [ # Features to drop to avoid multicollinearity or undefined columns
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

# Instantiate the HotelRegression class with configuration
hotel_regression = HotelRegression(DATA_PATH, DEPENDENT_VARIABLE, FEATURES_TO_DROP)

# Prepare data for regression
hotel_regression.prepare_data()

# Fit the linear regression model
hotel_regression.fit_model()

# Print model summary
hotel_regression.print_summary()

# Perform and save diagnostic plots and tests
hotel_regression.plot_residuals_vs_fitted()
hotel_regression.durbin_watson_test()
hotel_regression.breusch_pagan_test()
hotel_regression.normality_tests()

# Configure pandas to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
hotel_regression.calculate_vif()
