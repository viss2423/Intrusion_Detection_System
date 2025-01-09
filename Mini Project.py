import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer  # Import the SimpleImputer class

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()  # Remove leading and trailing whitespaces
    return df

# Function to perform data cleaning, normalization, and feature selection for a dataset
def process_dataset(dataset_file):
    # Load the dataset
    df = pd.read_csv(dataset_file)

    # Clean column names
    df = clean_column_names(df)

    # Drop rows with missing values in the 'Label' column
    df.dropna(subset=['Label'], inplace=True)

    # Extract the 'Label' column before normalization
    labels = df['Label']
    df.drop(columns=['Label'], inplace=True)

    # Check for and remove infinite or very large values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle missing values using mean imputation on all numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Normalize the data (excluding 'Label' column)
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Remove constant features before feature selection
    constant_features = df.columns[df.nunique() == 1]
    df.drop(columns=constant_features, inplace=True)

    # Combine the 'Label' column back with the processed data
    df['Label'] = labels

    return df


# Directory containing all the datasets
dataset_directory = "E:/MINI PROJECT/MachineLearningCSV/MachineLearningCVE/"

# List all the files in the directory
dataset_files = os.listdir(dataset_directory)

# Process and combine all datasets
datasets = [process_dataset(os.path.join(dataset_directory, file)) for file in dataset_files]
combined_data = pd.concat(datasets)

# Display individual datasets
for i, file in enumerate(dataset_files):
    dataset_name = os.path.splitext(file)[0]
    df = datasets[i]
    print(f"\nDataset {i+1}: {dataset_name}")
    print(df.head())

    # Display number of rows and columns for each dataset
    num_rows, num_cols = df.shape
    print(f"Number of rows: {num_rows}, Number of columns: {num_cols}\n")


#
#   PREPROCESSING ON COMBINED DATASET
#


# Reset the index of the combined dataset
combined_data.reset_index(drop=True, inplace=True)

# Handle missing values in the combined dataset
numeric_columns = combined_data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
combined_data[numeric_columns] = imputer.fit_transform(combined_data[numeric_columns])

# Normalize the combined dataset (excluding 'Label' column)
scaler = MinMaxScaler()
combined_data[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])

# Remove constant features before feature selection
constant_features = combined_data.columns[combined_data.nunique() == 1]
combined_data.drop(columns=constant_features, inplace=True)

# Drop rows with missing values in the 'Label' column
combined_data.dropna(subset=['Label'], inplace=True)

# Apply feature selection using SelectKBest to the combined dataset
num_features = 20  # Choose the number of top features you want to keep
selector = SelectKBest(score_func=f_classif, k=num_features)
X_new = selector.fit_transform(combined_data.drop(columns=['Label']), combined_data['Label'])
selected_features = combined_data.drop(columns=['Label']).columns[selector.get_support()]

# Combine the 'Label' column back with the selected features
selected_features = list(selected_features) + ['Label']
combined_data = combined_data[selected_features]

# Display the column names of the updated combined dataset (including the 'Label' column)
print("\nColumn names of the updated combined dataset:")
print(combined_data.columns)

# Display the combined dataset with the selected features and the 'Label' column
print("\nCombined Dataset with Selected Features:")
print(combined_data.head())

# Count the occurrences of each unique value in the 'Label' column
attack_counts = combined_data['Label'].value_counts()

# Display the count of different attacks
print(attack_counts)

# Display number of rows and columns for the updated combined dataset
num_rows, num_cols = combined_data.shape
print(f"\nNumber of rows in the updated combined dataset: {num_rows}")
print(f"Number of columns in the updated combined dataset: {num_cols}")



#
#  EXPLANATORY DATA ANALYSIS
#


# Function to perform basic EDA on a dataset
def perform_eda(dataset, dataset_name):
    # Create a new figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f"Exploratory Data Analysis for {dataset_name}", fontsize=16)

    # Display the first few rows of the dataset
    print(f"\nDataset: {dataset_name}")
    print("First few rows of the dataset:")
    print(dataset.head(), "\n")

    # Display the total number of rows and columns
    print("Number of rows and columns:")
    print(dataset.shape)

    # Check for missing values
    print("\nMissing values:")
    print(dataset.isnull().sum())

    # Get statistical summary
    print("\nStatistical summary:")
    print(dataset.describe())

    # Visualize the distribution of 'Flow Duration'
    axes[0, 0].hist(dataset['Flow Duration'], bins=50)
    axes[0, 0].set_title("Distribution of 'Flow Duration'")
    axes[0, 0].set_xlabel('Flow Duration')
    axes[0, 0].set_ylabel('Frequency')

    # Visualize the distribution of 'Total Fwd Packets'
    axes[0, 1].hist(dataset['Total Fwd Packets'], bins=50)
    axes[0, 1].set_title("Distribution of 'Total Fwd Packets'")
    axes[0, 1].set_xlabel('Total Fwd Packets')
    axes[0, 1].set_ylabel('Frequency')

    # Additional visualizations (customize as needed)
    sns.countplot(x='Label', data=dataset, ax=axes[1, 0])  # Change here
    axes[1, 0].set_title("Distribution of Labels")
    axes[1, 0].set_xlabel('Labels')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Show the plots
    plt.tight_layout()
    plt.show()

# Perform EDA on individual datasets
for i, file in enumerate(dataset_files):
    dataset_name = os.path.splitext(file)[0]
    df = datasets[i]
    perform_eda(df, dataset_name)
    print("-------------------------")


#
#        ADVANCED EDA ON COMBINED DATASET
#


plt.figure(figsize=(8, 6))
sns.countplot(x='Label', data=combined_data)
plt.title("Class Distribution in Combined Dataset")
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Compute correlation matrix for the combined dataset
correlation_matrix = combined_data.drop(columns=['Label']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Correlation Matrix of Features in Combined Dataset")
plt.show()

# Apply feature selection using SelectKBest to the combined dataset
num_features = 20  # Choose the number of top features you want to keep
selector = SelectKBest(score_func=f_classif, k=num_features)
X_new = selector.fit_transform(combined_data.drop(columns=['Label']), combined_data['Label'])

# Get feature importances from SelectKBest
feature_importance = selector.scores_
sorted_idx = np.argsort(feature_importance)[::-1]
sorted_features = combined_data.drop(columns=['Label']).columns[sorted_idx]
sorted_importance = feature_importance[sorted_idx]

# Plot the feature importances for the top num_features features
plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importance[:num_features], y=sorted_features[:num_features], palette="coolwarm")
plt.title("Top Feature Importances")
plt.xlabel("Score (ANOVA F-value)")
plt.ylabel("Feature")
plt.show()