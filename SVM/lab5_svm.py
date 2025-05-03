# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import datasets and tools from scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve
)

# -----------------------------
# OOP SVM Model Classes
# -----------------------------

class SVMModel:
    """
    Base class for SVM models (Regression and Classification).
    Provides common functionalities like data splitting and scaling.
    """
    def __init__(self, X, y, test_size=0.2, random_state=42, scaler=StandardScaler()):
        """
        Initializes the SVMModel with features (X), target (y), and configuration parameters.

        Parameters:
            X (numpy.ndarray or pandas.DataFrame): Feature dataset.
            y (numpy.ndarray or pandas.Series): Target variable.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Random state for train_test split for reproducibility. Defaults to 42.
            scaler (sklearn.preprocessing.Scaler, optional): Scaler object for feature scaling.
                                                             Defaults to StandardScaler().
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = scaler
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        self.X_train_scaled, self.X_test_scaled = self._scale_features()
        self.model = None  # Placeholder for the specific SVM model (SVR or SVC)
        self.grid_search = None # Placeholder for GridSearchCV object
        self.best_model = None  # Placeholder for the best model from GridSearchCV

    def _split_data(self):
        """Splits the dataset into training and testing sets."""
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def _scale_features(self):
        """Scales the features using the provided scaler."""
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def train(self, param_grid, cv=5, scoring=None):
        """
        Trains the SVM model using GridSearchCV for hyperparameter tuning.

        Parameters:
            param_grid (dict): Dictionary of hyperparameters to tune.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            scoring (str or callable, optional): Scoring metric to use in GridSearchCV.
                                                Must be specified by subclass.
        Raises:
            NotImplementedError: If scoring is not provided in subclass train method.
        """
        if scoring is None:
            raise NotImplementedError("Scoring parameter must be defined in the subclass train method.")
        self.grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring
        )
        self.grid_search.fit(self.X_train_scaled, self.y_train)
        self.best_model = self.grid_search.best_estimator_
        print(f"Best Model Parameters: {self.grid_search.best_params_}")
        if hasattr(self.best_model, 'coef_'):
            print(f"Model Coefficients: {self.best_model.coef_}")
        else:
            print("Model does not provide coefficients (likely due to kernel type).")

    def evaluate(self):
        """Evaluates the trained SVM model. Must be implemented by subclass."""
        raise NotImplementedError("Evaluate method must be implemented by subclass.")

    def predict(self, X_new=None):
        """Predicts on new data. Uses test data if X_new is None."""
        if X_new is None:
            X_new_scaled = self.X_test_scaled
        else:
            X_new_scaled = self.scaler.transform(X_new) # Scale new data using the fitted scaler
        return self.best_model.predict(X_new_scaled)


class SVMRegressionModel(SVMModel):
    """
    SVM Regression Model class using SVR.
    Inherits from SVMModel and implements regression-specific functionalities.
    """
    def __init__(self, X, y, test_size=0.2, random_state=42, scaler=StandardScaler(), dataset_name="Regression Dataset"):
        """
        Initializes SVMRegressionModel.

        Parameters:
            X (numpy.ndarray or pandas.DataFrame): Feature dataset.
            y (numpy.ndarray or pandas.Series): Target variable.
            test_size (float, optional): Test set size. Defaults to 0.2.
            random_state (int, optional): Random state for data splitting. Defaults to 42.
            scaler (sklearn.preprocessing.Scaler, optional): Scaler object. Defaults to StandardScaler().
            dataset_name (str, optional): Name of the dataset for reporting. Defaults to "Regression Dataset".
        """
        super().__init__(X, y, test_size, random_state, scaler)
        self.model = SVR() # Initialize SVR model
        self.dataset_name = dataset_name

    def describe_data(self, feature_names=None, target_name="target"):
        """Prints descriptive statistics of the dataset."""
        df = pd.DataFrame(self.X, columns=feature_names)
        df['target'] = self.y
        print(f"Regression Task Descriptive Statistics for: {self.dataset_name}")
        if feature_names:
            print("Independent Variables (features):", feature_names)
        print("Dependent Variable (target):", target_name)
        print(df.describe())
        print("\n" + "="*60 + "\n")


    def train(self, param_grid, cv=5, scoring='neg_mean_squared_error'):
        """Trains the SVR model using GridSearchCV."""
        print(f"Training SVR model on {self.dataset_name}...")
        super().train(param_grid, cv, scoring) # Call train method from parent class

    def evaluate(self):
        """Evaluates the trained SVR model and prints metrics, plots residuals."""
        y_pred = self.predict()
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"SVR Regression on {self.dataset_name} Mean Squared Error: {mse:.4f}")
        print(f"SVR Regression on {self.dataset_name} R2 Score: {r2:.4f}")
        self.plot_residuals(y_pred)

    def plot_residuals(self, y_pred):
        """Plots residuals to diagnose regression model errors."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, self.y_test - y_pred, alpha=0.7)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot for SVR Regression on {self.dataset_name}')
        plt.axhline(0, color='red', linestyle='--')
        plt.savefig('/teamspace/studios/this_studio/output/residual_plot_SVR.png')


class SVMClassificationModel(SVMModel):
    """
    SVM Classification Model class using SVC.
    Inherits from SVMModel and implements classification-specific functionalities.
    """
    def __init__(self, X, y, test_size=0.2, random_state=42, scaler=StandardScaler(), dataset_name="Classification Dataset", target_names=None):
        """
        Initializes SVMClassificationModel.

        Parameters:
            X (numpy.ndarray or pandas.DataFrame): Feature dataset.
            y (numpy.ndarray or pandas.Series): Target variable.
            test_size (float, optional): Test set size. Defaults to 0.2.
            random_state (int, optional): Random state for data splitting. Defaults to 42.
            scaler (sklearn.preprocessing.Scaler, optional): Scaler object. Defaults to StandardScaler().
            dataset_name (str, optional): Name of the dataset for reporting. Defaults to "Classification Dataset".
            target_names (list of str, optional): Names of the target classes for plot labels. Defaults to None.
        """
        super().__init__(X, y, test_size, random_state, scaler)
        self.model = SVC(probability=True) # Initialize SVC model with probability=True for probability estimates if needed
        self.dataset_name = dataset_name
        self.target_names = target_names

    def describe_data(self, feature_names=None, target_name="target"):
        """Prints descriptive statistics of the dataset."""
        df = pd.DataFrame(self.X, columns=feature_names)
        df['target'] = self.y
        print(f"Classification Task Descriptive Statistics for: {self.dataset_name}")
        if feature_names is not None and not feature_names.empty:
            print("Independent Variables (features):", feature_names)
        if self.target_names:
            print("Dependent Variable (target) mapped to classes:", self.target_names)
        else:
            print("Dependent Variable (target):", target_name)
        print(df.describe())
        print("\n" + "="*60 + "\n")

    def train(self, param_grid, cv=5, scoring='accuracy'):
        """Trains the SVC model using GridSearchCV."""
        print(f"Training SVC model on {self.dataset_name}...")
        super().train(param_grid, cv, scoring) # Call train method from parent class

    def evaluate(self):
        """Evaluates the trained SVC model and prints metrics, plots confusion matrix."""
        y_pred = self.predict()
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"SVC Classification on {self.dataset_name} Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(self.y_test, y_pred, target_names=self.target_names))
        self.plot_confusion_matrix(y_pred)
        self.plot_roc_curve() # Call ROC curve plotting function
        self.report_roc_auc_score() # Call ROC AUC reporting function


    def plot_confusion_matrix(self, y_pred):
        """Plots the confusion matrix for classification model evaluation."""
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.target_names if self.target_names else sorted(list(set(self.y))),
                    yticklabels=self.target_names if self.target_names else sorted(list(set(self.y))))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for SVC Classification on {self.dataset_name}')
        plt.savefig('/teamspace/studios/this_studio/output/confusion_matrixSVC.png')

    def plot_roc_curve(self):
        """Plots the ROC curve."""
        y_prob = self.best_model.predict_proba(self.X_test_scaled)[:, 1] # Probability estimates for the positive class
        fpr, tpr, thresholds = roc_curve(self.y_test, y_prob)
        roc_auc = roc_auc_score(self.y_test, y_prob) # Calculate ROC AUC score

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for SVC on {self.dataset_name}')
        plt.legend(loc="lower right")
        plt.savefig('/teamspace/studios/this_studio/output/roc_curve_SVC.png')

    def report_roc_auc_score(self):
        """Prints ROC AUC score."""
        y_prob = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(self.y_test, y_prob)
        print(f"SVC Classification on {self.dataset_name} ROC AUC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    # -----------------------------
    # Configuration and Example Usage
    # -----------------------------

    # --- Datasets ---
    # Load example datasets (you can replace these with your own data)
    data = pd.read_csv('/teamspace/studios/this_studio/dataset/hb_test4.csv')

    #take samples
    df = data.sample(n=1000, random_state=42) 

    X_features = df.drop(columns=[
        'is_canceled',
        'reservation_status_date',
        'reservation_status_Canceled',
        'reservation_status_Check-Out',
        'reservation_status_No-Show'
    ])
    y_tartget = df['is_canceled']

    # --- Configuration Parameters ---
    # Common parameters for both Regression and Classification
    TEST_SIZE = 0.25
    RANDOM_STATE = 42

    # Regression specific parameters
    REG_PARAM_GRID = {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5]
    }

    # Classification specific parameters
    CLF_PARAM_GRID = {
        'kernel': ['linear', 'rbf'],
        'C': [8],
        'gamma': ['scale', 'auto']
    }

    # --- Example Usage with Dataset (Classification) ---
    print("="*60 + "\nDataset - Classification Example\n" + "="*60)

    # Initialize and train the SVM Classification model
    hotel_clf_model = SVMClassificationModel(
        X=X_features,
        y=y_tartget,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        dataset_name="Hotel Dataset",
        target_names=['Not Canceled', 'Canceled']
    )

    hotel_clf_model.describe_data(feature_names=X_features.columns)
    hotel_clf_model.train(CLF_PARAM_GRID)
    hotel_clf_model.evaluate()

    print("="*60 + "\nEnd of Examples\n" + "="*60)