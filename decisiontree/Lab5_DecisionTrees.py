# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class HotelBookingClassifierMinimal:
    """
    A minimal class for hotel booking cancellation classification using Decision Tree.
    For quick testing purposes with flexible feature selection and data input.
    """

    def __init__(self, test_size=0.2, random_state=42, max_depth=3, target_column='is_canceled', features_to_use=None):
        """
        Initializes the Minimal HotelBookingClassifier with flexible feature configurations.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_depth = max_depth
        self.target_column = target_column
        self.features_to_use = features_to_use

        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = ['Not Canceled', 'Canceled']
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dtc = None
        self.y_pred = None
        self.cm = None
        self.y_prob = None # To store predicted probabilities for ROC/AUC

    def load_data(self, dataframe=None, csv_filepath=None):
        """Loads data from DataFrame or CSV file, using only specified features."""
        if dataframe is not None:
            print("Loading dataset from provided Pandas DataFrame.")
            self.df = dataframe
        elif csv_filepath:
            print(f"Loading dataset from CSV file: {csv_filepath}")
            self.df = pd.read_csv(csv_filepath)
        else:
            raise ValueError("No dataset source provided. Please provide either dataframe or csv_filepath.")

        if self.features_to_use is None:
            self.features_to_use = [
                'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'is_city_hotel'
            ] # Default minimal feature set if not specified

        print("\nFirst five rows of the dataset with selected features:")
        print(self.df[self.features_to_use + [self.target_column]].head())

        self.X = self.df[self.features_to_use]
        self.y = self.df[self.target_column]
        self.feature_names = self.X.columns.tolist()

    def split_data(self):
        """Splits data into training and testing sets."""
        print("\nSplitting dataset into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        print(f"Training set size: {len(self.X_train)}")
        print(f"Testing set size: {len(self.X_test)}")

    def train_model(self):
        """Trains Decision Tree model."""
        print("\nTraining Decision Tree Classifier...")
        self.dtc = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.dtc.fit(self.X_train, self.y_train)
        print("Decision Tree Classifier trained.")

    def predict(self):
        """Makes predictions."""
        print("\nMaking predictions on the test set...")
        self.y_pred = self.dtc.predict(self.X_test)
        # Get probability estimates for ROC/AUC calculation
        self.y_prob = self.dtc.predict_proba(self.X_test)[:, 1] # Probability of class 1
        print("Predictions made.")

    def evaluate_model(self):
        """Evaluates model and prints essential metrics."""
        print("\nEvaluating the model performance:")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)

        print("\nClassification Report:")
        cr = classification_report(self.y_test, self.y_pred, target_names=self.target_names)
        print(cr)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")

        # Calculate and print AUC
        auc_score = roc_auc_score(self.y_test, self.y_prob)
        print(f"\nAUC: {auc_score:.2f}")


    def visualize_tree(self):
        """Visualizes the Decision Tree and saves it to a file."""
        print("\nVisualizing the Decision Tree and saving to file...")
        plt.figure(figsize=(12, 8))
        plot_tree(
            self.dtc,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title('Decision Tree - Hotel Booking Cancellation')

        # Save the figure BEFORE showing it
        plt.savefig("/teamspace/studios/this_studio/output/decision_tree.png", format='png')
        print("Decision tree image saved to: /teamspace/studios/this_studio/output/decision_tree.png")

    def plot_roc_curve(self):
        """
        Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).
        Saves the ROC curve plot to a file.
        """
        print("\nGenerating and saving ROC curve...")
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_prob)
        roc_auc = roc_auc_score(self.y_test, self.y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        # Save the ROC curve plot
        roc_curve_filepath = "/teamspace/studios/this_studio/output/roc_curve.png"
        plt.savefig(roc_curve_filepath, format='png')
        print(f"ROC curve image saved to: {roc_curve_filepath}")
        plt.show() # Show plot for immediate view, if needed, you can remove this for script execution

    def run(self, dataframe=None, csv_filepath=None, features=None):
        """Runs the classification process with flexible data and feature input."""
        print("Starting Hotel Booking Cancellation Classification Process...")
        if features is not None:
            self.features_to_use = features

        self.load_data(dataframe=dataframe, csv_filepath=csv_filepath)
        self.split_data()
        self.train_model()
        self.predict()
        self.evaluate_model()
        self.visualize_tree()
        self.plot_roc_curve() # Call the new function to plot ROC curve and AUC
        print("\nHotel Booking Cancellation Classification Process Completed.")


# --- Configuration and Execution for Flexible Test ---
if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv("/teamspace/studios/this_studio/dataset/hb_test4.csv")

    print("--- Hotel Booking Cancellation Classification Tests ---")

    # --- Configuration ---
    test_size_config = 0.25
    random_state_config = 42
    max_depth_config = 13
    target_column_config = 'is_canceled'
    dataframe_input = df # Use DataFrame input

    # --- Model testing ---
    print("\n--- Model Test ---")
    all_numeric_features_config = df.drop(
        columns=[
            'is_canceled',
            'reservation_status_date',
            'reservation_status_Canceled',
            'reservation_status_No-Show',
            'reservation_status_Check-Out',
        ]
    ).select_dtypes(include=np.number).columns.tolist()
    # Ensure target column is removed if it's numeric and accidentally included
    if target_column_config in all_numeric_features_config:
        all_numeric_features_config.remove(target_column_config)

    all_numeric_classifier = HotelBookingClassifierMinimal(
        test_size=test_size_config,
        random_state=random_state_config,
        max_depth=max_depth_config,
        target_column=target_column_config,
        features_to_use=all_numeric_features_config
    )
    all_numeric_classifier.run(dataframe=dataframe_input)

    print("\n--- Flexible Tests Completed ---")