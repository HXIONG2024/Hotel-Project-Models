# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
#show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AssociationRuleMiner:
    """
    A class to perform association rule mining on a dataset using the Apriori algorithm.

    This class is designed to be flexible in handling different categories of columns
    for association rule mining without requiring modifications to the constructor
    when new column categories are added. You can specify column groups in the config.
    """
    def __init__(
        self,
        dataset_path: str,
        sample_size: int,
        random_state: int,
        column_groups: dict = None,
        min_support: float = 0.2,
        rule_metric: str = "confidence",
        min_threshold: float = 0.6,
        output_plot_path: str = "output/association_rules_scatter_plot.png",
        association_cols: list = None,
        output_csv_path: str = "output/top_association_rules.csv" # Add output_csv_path to init
    ):
        """
        Initializes the miner with dataset configurations and mining parameters.

        :param dataset_path: Path to the CSV dataset.
        :param sample_size: Number of random rows to sample. If None, the entire dataset is used.
        :param random_state: Random state for reproducibility.
        :param column_groups: A dictionary where keys are group names (e.g., 'meals', 'rooms')
                              and values are lists of column names belonging to that group.
                              Example: {'meals': ['meal_BB', 'meal_FB'], 'rooms': ['room_A', 'room_B']}.
                              If provided, association_cols will be automatically constructed
                              from these groups, unless association_cols is explicitly provided.
        :param min_support: Minimum support for the apriori algorithm.
        :param rule_metric: Metric to evaluate association rules (e.g., 'confidence', 'lift').
        :param min_threshold: Minimum threshold for the rule metric.
        :param output_plot_path: Path to save the scatter plot of association rules.
        :param association_cols: Optional list of columns to perform association mining.
                                 If provided, it overrides column_groups for defining association columns.
                                 If not provided and column_groups is given, association_cols will be
                                 derived from column_groups. If neither is provided, it will raise a ValueError.
        :param output_csv_path: Path to save the top association rules to a CSV file. # Docstring for output_csv_path
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.column_groups = column_groups
        self.min_support = min_support
        self.rule_metric = rule_metric
        self.min_threshold = min_threshold
        self.output_plot_path = output_plot_path
        self.output_csv_path = output_csv_path # Store output_csv_path

        if association_cols is not None:
            # If association_cols is directly provided, use it.
            self.association_cols = association_cols
        elif column_groups is not None:
            # If column_groups is provided, construct association_cols from it.
            self.association_cols = []
            for group_name, cols in column_groups.items():
                self.association_cols.extend(cols)
            if not self.association_cols:
                raise ValueError("column_groups is provided but is empty, resulting in no association columns.")
        else:
            raise ValueError("Either association_cols or column_groups must be provided to define columns for association mining.")

        # These will be populated as the script runs:
        self.data = None
        self.sample_df = None
        self.df_assoc = None
        self.frequent_itemsets = None
        self.rules = None

    def load_data(self):
        """Loads the dataset from CSV."""
        self.data = pd.read_csv(self.dataset_path)
        print("Dataset loaded successfully.")

    def sample_data(self):
        """Takes a random sample of rows from the full dataset, or uses the full dataset if sample_size is None."""
        if self.data is None:
            raise ValueError("Data not loaded. Please run load_data() first.")
        if self.sample_size is None:
            self.sample_df = self.data.copy() # Use the entire dataset
            print("Using the full dataset for association mining.")
        else:
            self.sample_df = self.data.sample(n=self.sample_size, random_state=self.random_state)
            print(f"Random sample of {self.sample_size} rows created.")

    def subset_data(self):
        """
        Subsets the sample data for association mining based on the specified association columns.
        """
        self.df_assoc = self.sample_df[self.association_cols]
        print("Data subset for association mining:")
        print(self.df_assoc.head())

    def mine_frequent_itemsets(self):
        """Applies the Apriori algorithm to find frequent itemsets."""
        if self.df_assoc is None:
            raise ValueError("Data subset not created. Please run subset_data() first.")
        self.frequent_itemsets = apriori(self.df_assoc, min_support=self.min_support, use_colnames=True)
        print("\nFrequent Itemsets:")
        print(self.frequent_itemsets)

    def format_rule_itemset(self, itemset):
        """
        Formats an itemset (antecedents or consequents) into a string like "Item1 -> Item2 -> Item3".
        """
        return ' -> '.join(sorted(list(itemset)))

    def generate_rules(self):
        """Generates association rules from the frequent itemsets and formats antecedents/consequents."""
        if self.frequent_itemsets is None:
            raise ValueError("Frequent itemsets not mined. Please run mine_frequent_itemsets() first.")
        self.rules = association_rules(self.frequent_itemsets, metric=self.rule_metric, min_threshold=self.min_threshold, support_only=False)

        # Format antecedents and consequents to desired string representation
        if not self.rules.empty: # Check if rules are generated
            self.rules['antecedents_str'] = self.rules['antecedents'].apply(self.format_rule_itemset)
            self.rules['consequents_str'] = self.rules['consequents'].apply(self.format_rule_itemset)

        print("\nAssociation Rules:")
        if not self.rules.empty:
            print(self.rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'conviction']])
        else:
            print("No association rules found based on the given criteria.")


    def visualize_rules(self):
        """Visualizes the association rules using a scatter plot."""
        if self.rules is None or self.rules.empty: # Check if rules is None or empty
            raise ValueError("Rules have not been generated or no rules found. Please run generate_rules() first and ensure rules are generated.")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='support',
            y='confidence',
            size='lift',
            data=self.rules,
            hue='lift',
            palette='viridis',
            sizes=(100, 1000),
            alpha=0.7
        )
        plt.title('Association Rules Scatter Plot')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.legend(title='Lift', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_plot_path)
        print(f"Scatter plot saved to {self.output_plot_path}")

    def display_top_rules(self):
        """Prints the top association rules sorted by lift, using the formatted string representation."""
        if self.rules is None or self.rules.empty: # Check if rules is None or empty
            raise ValueError("Rules have not been generated or no rules found. Please run generate_rules() first and ensure rules are generated.")
        sorted_rules = self.rules.sort_values(by='lift', ascending=False)
        print("\nTop Association Rules Sorted by Lift:")
        print(sorted_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'conviction']])
        return sorted_rules # Return sorted_rules for saving to CSV


    def save_top_rules_to_csv(self):
        """Saves the top association rules (sorted by lift) to a CSV file."""
        if self.rules is None or self.rules.empty:
            print("No rules to save to CSV. Please generate rules first.")
            return

        top_rules_df = self.display_top_rules() # Reuse display_top_rules to get sorted rules
        if top_rules_df is not None and not top_rules_df.empty:
            top_rules_df[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'conviction']].to_csv(
                self.output_csv_path, index=False
            )
            print(f"Top association rules saved to: {self.output_csv_path}")
        else:
            print("No top rules to save to CSV.")


    def run(self):
        """Executes the full pipeline for association rule mining."""
        self.load_data()
        self.sample_data()
        self.subset_data()
        self.mine_frequent_itemsets()
        self.generate_rules()
        if self.rules is not None and not self.rules.empty: # Only visualize and display if rules are found
            self.visualize_rules()
            self.display_top_rules()
            self.save_top_rules_to_csv() # Call save_top_rules_to_csv to save output
        else:
            print("No association rules to visualize, display or save.")


if __name__ == "__main__":
    # Configuration parameters using column_groups for flexibility
    CONFIG = {
        "dataset_path": "dataset/hb_association.csv",
        "sample_size": None, # Set sample_size to None to use the full dataset
        "random_state": 42,
        "column_groups": {
            "meal_plans": ["meal_BB", "meal_FB", "meal_HB", "meal_SC", "meal_Undefined"],
            "room_types": [
                "assigned_room_type_A", "assigned_room_type_B", "assigned_room_type_C",
                "assigned_room_type_D", "assigned_room_type_E", "assigned_room_type_F",
                "assigned_room_type_G", "assigned_room_type_H", "assigned_room_type_I",
                "assigned_room_type_K", "assigned_room_type_L", "assigned_room_type_P"
            ],
            "week_numbers": ["week_0", "week_1", "week_2", "week_3", "week_4", "week_5", "week_6"],
            "customer_types": [
                "customer_type_Contract", "customer_type_Group", "customer_type_Transient",
                "customer_type_Transient-Party"
            ],
            "market_segments": [
                "market_segment_Aviation", "market_segment_Complementary",
                "market_segment_Corporate", "market_segment_Direct",
                "market_segment_Groups", "market_segment_Offline TA/TO",
                "market_segment_Online TA", "market_segment_Undefined"
            ],
            "required_parking_spaces": [
                "required_car_parking_spaces_0", "required_car_parking_spaces_1",
                "required_car_parking_spaces_2", "required_car_parking_spaces_3",
                "required_car_parking_spaces_8"
            ],
            "total_of_special_requests": [
                "total_of_special_requests_0", "total_of_special_requests_1",
                "total_of_special_requests_2", "total_of_special_requests_3",
                "total_of_special_requests_4", "total_of_special_requests_5"
            ],
            "children_count": [
                "children_0.0", "children_1.0", "children_2.0", "children_3.0"
            ],
            "adults_count": [
                "adults_1", "adults_2", "adults_3", "adults_4"
            ],
            "stays_in_weekend_nights": [
                "stays_in_weekend_nights_0", "stays_in_weekend_nights_1",
                "stays_in_weekend_nights_2", "stays_in_weekend_nights_3",
                "stays_in_weekend_nights_4"
            ],
            "stays_in_week_nights": [
                "stays_in_week_nights_0", "stays_in_week_nights_1",
                "stays_in_week_nights_2", "stays_in_week_nights_3",
                "stays_in_week_nights_4", "stays_in_week_nights_5"
            ]
        },
        "min_support": 0.2, # minimum support threshold for frequent itemsets
        "rule_metric": "confidence",
        "min_threshold": 0.6, # minimum Confidence threshold for rules
        "output_plot_path": "/teamspace/studios/this_studio/output/association_rules_scatter_plot.png",
        "output_csv_path": "/teamspace/studios/this_studio/output/top_association_rules.csv" # Output CSV path config
    }

    # Instantiate and run the association rule miner using the config parameters
    miner = AssociationRuleMiner(**CONFIG)
    miner.run()