import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBayesClassifier:
    """
    A class to perform Naive Bayes classification.

    This class encapsulates the process of loading data, splitting it, training a Gaussian Naive Bayes model,
    making predictions, and evaluating the model's performance. It is designed to be configurable
    for use with different datasets by modifying the data loading and class initialization parameters.
    """

    def __init__(self, test_size=0.25, random_state=42, model=None, stratify=True):
        """
        Initializes the NaiveBayesClassifier object.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.
            random_state (int, optional): Random state for splitting data to ensure reproducibility. Defaults to 42.
            model (sklearn.naive_bayes classifier, optional):  A pre-initialized Naive Bayes classifier model.
                Defaults to None, which initializes a GaussianNB model. You can pass other Naive Bayes variants here.
            stratify (bool, optional): Whether to stratify the train/test split based on target variable. Defaults to True.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        # Initialize Gaussian Naive Bayes model if no model is provided
        self.model = model if model is not None else GaussianNB()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None # Initialize data splits
        self.y_pred = None # Initialize predictions
        self.target_names = None # Initialize target names
        self.feature_names = None # Initialize feature names


    def load_data(self, data, target, feature_names=None, target_names=None):
        """
        Loads the dataset.

        This method takes your feature data and target data as input.
        It can handle numpy arrays or pandas DataFrames for features and numpy arrays or pandas Series for target.

        Args:
            data (array-like or DataFrame): Feature data.
            target (array-like or Series): Target data.
            feature_names (list of str, optional): Names of the features for better visualization and reporting. Defaults to None.
            target_names (list of str, optional): Names of the target classes for better visualization and reporting. Defaults to None.
        """
        self.X = data
        self.y = target
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i+1}' for i in range(self.X.shape[1])] if hasattr(self.X, 'shape') else ['feature'] # Generate default feature names if not provided
        if target_names is None:
            unique_targets = np.unique(self.y)
            self.target_names = [f'class_{t}' for t in unique_targets] # Generate default target names if not provided
        else:
            self.target_names = target_names


    def split_data(self):
        """
        Splits the loaded data into training and testing sets.

        Uses sklearn's train_test_split function to divide the data. Stratification is applied if specified in the constructor.
        """
        stratify_y = self.y if self.stratify else None # Stratify only if self.stratify is True
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_y
        )

    def train_model(self):
        """
        Trains the Naive Bayes classifier model.

        Fits the initialized Naive Bayes model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Makes predictions on the test set.

        Uses the trained model to predict the target values for the test feature data.
        """
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        """
        Evaluates the performance of the classifier.

        Prints the confusion matrix, classification report, and accuracy score.
        Also visualizes the confusion matrix as a heatmap.
        """
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)

        print("\nClassification Report:")
        cr = classification_report(self.y_test, self.y_pred, target_names=self.target_names)
        print(cr)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        auc_roc = self.calculate_auc_roc() # Calculate AUC-ROC score
        print(f"AUC-ROC Score: {auc_roc:.4f}") # Print AUC-ROC score

        self.plot_roc_curve() # Plot ROC Curve

        self.visualize_confusion_matrix(cm)

    def visualize_confusion_matrix(self, cm):
        """
        Visualizes the confusion matrix as a heatmap.

        Args:
            cm (ndarray): Confusion matrix to be visualized.
        """
        plt.figure(figsize=(8,6)) # Increased figure size for better readability
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names,
                    yticklabels=self.target_names)
        plt.ylabel('Actual', fontsize=12) # Added fontsize for labels
        plt.xlabel('Predicted', fontsize=12) # Added fontsize for labels
        plt.title('Confusion Matrix - Gaussian Naïve Bayes', fontsize=14) # Added fontsize for title
        plt.xticks(fontsize=10) # Added fontsize for ticks
        plt.yticks(fontsize=10) # Added fontsize for ticks
        plt.savefig("/teamspace/studios/this_studio/output/naive_bayes_confusion_matrix.png", dpi=300) # Save the figure with high resolution

    def calculate_auc_roc(self):
        """Calculates the Area Under the ROC Curve (AUC-ROC)."""
        # Get predicted probabilities for the positive class (class 1 - 'Canceled' in your case)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_prob) # Calculate AUC-ROC
        return auc_score


    def plot_roc_curve(self):
        """Plots the Receiver Operating Characteristic (ROC) curve."""
        y_prob = self.model.predict_proba(self.X_test)[:, 1] # Get predicted probabilities
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob) # Calculate FPR, TPR, and thresholds for ROC curve

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.calculate_auc_roc():.2f})') # Plot ROC curve
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing') # Plot diagonal line for random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig("/teamspace/studios/this_studio/output/naive_bayes_roc_curve.png", dpi=300) # Save the figure with high resolution
    
    def run_pipeline(self, data, target, feature_names=None, target_names=None):
        """
        Runs the complete classification pipeline.

        This method orchestrates the entire process: loading data, splitting data, training the model,
        making predictions, and evaluating the model.

        Args:
            data (array-like or DataFrame): Feature data.
            target (array-like or Series): Target data.
            feature_names (list of str, optional): Names of the features. Defaults to None.
            target_names (list of str, optional): Names of the target classes. Defaults to None.
        """
        self.load_data(data, target, feature_names, target_names)
        self.split_data()
        self.train_model()
        self.predict()
        self.evaluate_model()


if __name__ == '__main__':
    # --- Example Usage with Your Own Dataset ---
    print("\n--- Model Result ---")
     # --- Load Your Dataset ---
    own_df = pd.read_csv('/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv') # Assuming your CSV is in the same directory

    # --- Separate Features (X) and Target (y) ---
    X_own = own_df[[
        'lead_time', 
        'arrival_date_week_number', 
        'arrival_date_day_of_month',
        'adr'
        ]]
    y_own = own_df['is_canceled'] # Target: 'is_canceled' column

    # --- Define Feature Names (optional, but good practice) ---
    feature_names_own = list(X_own.columns) # Get feature names from DataFrame columns

    # --- Define Target Names (optional, for better reporting) ---
    target_names_own = ['Not Canceled', 'Canceled'] # Based on your 'is_canceled' values

    # --- Initialize and run the classifier for your own dataset ---
    own_classifier = NaiveBayesClassifier(test_size=0.25, random_state=42, stratify=True) # Configurable parameters
    own_classifier.run_pipeline(X_own, y_own, feature_names_own, target_names_own)