# IMPORTS -----------------------------------------------------------------
##############################
# Data Manipulation & Utilities
##############################
import os
import time
import math
import sqlite3
import datetime
import textwrap
from itertools import combinations, product
from collections import Counter
from dateutil.relativedelta import relativedelta
import dataframe_image as dfi
import tensorflow as tf

##############################
# Core Scientific & Numeric Libraries
##############################
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, uniform, randint
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, LassoCV, Perceptron
from sklearn.base import BaseEstimator, ClassifierMixin
import jinja2
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

##############################
# Visualization Libraries
##############################
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from pandas.plotting import parallel_coordinates

##############################
# Machine Learning Preprocessing
##############################
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, 
    StandardScaler, 
    RobustScaler, 
    PowerTransformer,
    OrdinalEncoder, 
    LabelEncoder, 
    OneHotEncoder
)
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss, TomekLinks, ClusterCentroids

##############################
# Feature Selection
##############################
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2

##############################
# Model Building
##############################
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    StackingClassifier, 
    RandomForestRegressor, 
    BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

##############################
# Model Evaluation & Selection
##############################
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV, 
    cross_val_score, 
    StratifiedKFold,
    PredefinedSplit
)
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    classification_report, 
    make_scorer, 
    precision_recall_curve, 
    roc_curve, 
    auc,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    balanced_accuracy_score
)

##############################
# Pipeline & Column Transformations
##############################
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

##############################
# Data Balancing & Sampling
##############################
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek



# FUNCTIONS ---------------------------------------------------------------
### EDA -------------------------------------------------------------------

def delete_missing_rows(df):
    """
    Remove rows where all columns, except 'Assembly Date' and 'Claim Identifier', have NaN values.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.

    Returns:
    - pd.DataFrame: A new DataFrame with rows containing all NaN values (except in 'Assembly Date' 
      and 'Claim Identifier') removed.
    """

    # Identifies rows where all columns, except 'Assembly Date' and 'Claim Identifier', have NaN values
    missing_rows = df.drop(['Assembly Date', 'Claim Identifier'], axis=1).isnull().all(axis=1)
    # Deletes these rows from the DataFrame
    df = df[~missing_rows]
    return df

def convert_to_datetime(df, date_columns):
    """
    Convert specified columns to datetime format, coercing invalid entries to NaT (Not a Time).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be converted.
    - date_columns (list of str): List of column names to be converted to datetime format.

    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns converted to datetime format.
    """

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
    return df

def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be dropped.
    - columns_to_drop (list of str): List of column names to drop from the DataFrame.

    Returns:
    - pd.DataFrame: The modified DataFrame with the specified columns removed.
    """

    df.drop(columns=columns_to_drop, inplace=True)
    return df

def datetime_statistics_optimized(df, date_column):
    """
    Generate detailed statistics for a datetime column, including range, mean, median, 
    and standard deviation, with dates displayed in string format (YYYY-MM-DD).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the datetime column.
    - date_column (str): The name of the datetime column to analyze.

    Returns:
    - dict: A dictionary containing the following statistics:
        - 'Earliest Date': The earliest date in the column.
        - 'Latest Date': The latest date in the column.
        - 'Range of Dates': The difference between the earliest and latest dates in years, months, and days.
        - 'Number of Records': The count of non-missing values in the column.
        - 'Number of Missing Values': The count of missing (NaT) values in the column.
        - 'Distinct Values': The number of unique dates in the column.
        - 'Mean Date': The average date in the column.
        - 'Median Date': The median date in the column.
        - 'Standard Deviation (approx.)': The standard deviation of dates, expressed as years, months, and days.
        - 'Most Frequent Date': The most common date in the column.

    Notes:
    - Invalid datetime entries are coerced to NaT.
    - The range and standard deviation are approximations, using common calendar assumptions 
      (e.g., 365.25 days per year, 30.44 days per month).
    """

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')  
    
    # Calculate basic statistics
    earliest_date = df[date_column].min()
    latest_date = df[date_column].max()
    
    # Calculate range in years, months, and days
    if pd.notna(earliest_date) and pd.notna(latest_date):
        date_range = relativedelta(latest_date, earliest_date)
        range_str = f"{date_range.years} years, {date_range.months} months, {date_range.days} days"
    else:
        range_str = "N/A"
    
    # Standard deviation in days, then converted to (years, months, days)
    std_dev_days = (df[date_column] - df[date_column].mean()).dt.days.std()
    std_years, std_months, std_days = convert_days_to_years_months_days(std_dev_days)
    std_dev_str = f"{std_years} years, {std_months} months, {std_days} days"
    
    # Final statistics
    stats = {
        'Earliest Date': earliest_date,
        'Latest Date': latest_date,
        'Range of Dates': range_str,
        'Number of Records': df[date_column].count(),
        'Number of Missing Values': df[date_column].isna().sum(),
        'Distinct Values': df[date_column].nunique(),
        'Mean Date': df[date_column].mean(),
        'Median Date': df[date_column].median(),
        'Standard Deviation (approx.)': std_dev_str,
        'Most Frequent Date': df[date_column].mode()[0] if not df[date_column].mode().empty else 'N/A'
    }

    for key in ['Earliest Date', 'Latest Date', 'Mean Date', 'Median Date', 'Most Frequent Date']:
        if pd.notna(stats[key]):
            stats[key] = stats[key].strftime('%Y-%m-%d')

    return stats

def convert_days_to_years_months_days(days):
    """
    Convert a number of days into an approximate (years, months, days) format.

    Parameters:
    - days (int or float): The total number of days to be converted.

    Returns:
    - tuple: A tuple (years, months, days), where:
        - years (int): The approximate number of years.
        - months (int): The approximate number of months.
        - days (int): The remaining days after calculating years and months.
    """

    years = days // 365.25
    remaining_days = days % 365.25
    months = remaining_days // 30.44
    days = remaining_days % 30.44
    return int(years), int(months), int(round(days))

def invalid_claims(df, column):
    """
    Count the number of invalid claims where the time difference in the specified column is negative.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to evaluate.
    - column (str): The name of the column to check for negative values.

    Returns:
    - tuple: A tuple containing:
        - invalid_claims_count (int): The number of rows with negative values in the specified column.
        - all_claims_count (int): The total number of rows in the DataFrame.
    """

    # Count negative (invalid) claims in the specified column
    invalid_claims_count = (df[column] < 0).sum()
    # Total number of claims
    all_claims_count = df.shape[0]
    return invalid_claims_count, all_claims_count

def numerical_statistics(df, numerical_column):
    """
    Calculate and return a dictionary of statistics for a specified numerical column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the numerical column.
    - numerical_column (str): The name of the numerical column to analyze.

    Returns:
    - dict: A dictionary containing the following statistics:
        - 'Minimum Value': The smallest value in the column.
        - 'Maximum Value': The largest value in the column.
        - 'Range': The difference between the maximum and minimum values.
        - 'Number of Records': The count of non-missing values in the column.
        - 'Number of Missing Values': The count of missing values in the column.
        - 'Missing Values (%)': The percentage of missing values relative to the total number of records.
        - 'Distinct Values': The number of unique values in the column.
        - 'Mean': The arithmetic mean of the values.
        - 'Median': The median value of the column.
        - 'Standard Deviation': The standard deviation of the values, rounded to 2 decimal places.
        - 'Most Frequent Value': The most common value in the column (mode), or 'N/A' if no mode exists.
    """

    total_records = df.shape[0]

    # Calculate and store statistics
    stats = {
        'Minimum Value': df[numerical_column].min(),
        'Maximum Value': df[numerical_column].max(),
        'Range': df[numerical_column].max() - df[numerical_column].min(),
        'Number of Records': df[numerical_column].count(),
        'Number of Missing Values': df[numerical_column].isna().sum(),
        'Missing Values (%)': f"{(df[numerical_column].isna().sum() / total_records) * 100:.2f}%",
        'Distinct Values': df[numerical_column].nunique(),
        'Mean': df[numerical_column].mean(),
        'Median': df[numerical_column].median(),
        'Standard Deviation': round(df[numerical_column].std(), 2),
        'Most Frequent Value': df[numerical_column].mode()[0] if not df[numerical_column].mode().empty else 'N/A'
    }
    
    return stats

def replace_zeros_with_nan(df, features):
    """
    Replace 0 values with NaN in the specified columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to be modified.
    - features (list of str): List of column names where 0 values should be replaced with NaN.

    Returns:
    - pd.DataFrame: The modified DataFrame with 0 values replaced by NaN in the specified columns.
    """

    for feature in features:
        df[feature] = df[feature].replace(0, np.nan)
    return df

def add_counts(ax):
    """
    Add count labels on top of each bar in a bar plot.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object containing the bar plot.

    Notes:
    - The count is placed at the center of each bar and slightly above its height.
    - Font size, alignment, and offset are predefined for readability.
    """

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
        
def wrap_labels(labels, max_width=20):
    """
    Wrap labels to the next line if they exceed a specified maximum width.

    Parameters:
    - labels (list of str): A list of labels to be wrapped.
    - max_width (int): The maximum character width before wrapping (default: 20).

    Returns:
    - list of str: A list of labels with long text wrapped to fit within the specified width.
    """
    
    wrapped_labels = []
    for label in labels:
        if len(label) > max_width:
            split_point = label[:max_width].rfind(' ') 
            if split_point == -1:  
                split_point = max_width
            wrapped_label = label[:split_point] + '\n' + label[split_point + 1:]
            wrapped_labels.append(wrapped_label)
        else:
            wrapped_labels.append(label)
    return wrapped_labels

def plot_top_histograms(dataset, variables_list, top_n, rows, cols):
    """
    Plot horizontal histograms for the top N categories in each variable from the dataset.

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data to plot.
    - variables_list (list of str): List of column names to plot.
    - top_n (int): The number of top categories to display in each histogram.
    - rows (int): The number of rows in the subplot grid.
    - cols (int): The number of columns in the subplot grid.

    Notes:
    - Any text labels exceeding 40 characters are wrapped to fit better on the plots.
    - Unused subplot axes (if `rows * cols` > `len(variables_list)`) are turned off.
    - Each subplot includes a title (the column name) and frequency labels on the x-axis.

    Displays:
    - A grid of horizontal bar plots showing the frequency of the top N categories for each variable.
    """

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    axes = axes.flatten()  
    
    for i, column in enumerate(variables_list):
        if i >= len(axes):  
            break

        # Top N or fewer values in each categorical column
        top_values = dataset[column].value_counts().nlargest(top_n)

        # Descending order in horizontal bar plots
        top_values = top_values[::-1]
        
        wrapped_labels = wrap_labels(top_values.index, max_width=40)
        
        axes[i].barh(wrapped_labels, top_values.values, color=sns.color_palette("viridis", as_cmap=True)(0.4), align='center')
        axes[i].set_title(column)
        axes[i].set_xlabel("Frequency", fontsize=10) 
        axes[i].tick_params(axis='y', labelsize=8)  
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def replace_unknown_with_na(df, feature, unknown):
    """
    Replace specified "unknown" values with NaN (pd.NA) in a given column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column to modify.
    - feature (str): The name of the column in which "unknown" values will be replaced.
    - unknown: The value in the column to be treated as "unknown" and replaced with pd.NA.

    Returns:
    - pd.DataFrame: A new DataFrame with "unknown" values replaced by pd.NA in the specified column.
    """

    new_df = df.copy()
    new_df[feature] = new_df[feature].replace(unknown, pd.NA)
    return new_df

def plot_heatmap_with_zero_mask(feature1, feature2, contingency_table):
    """
    Create a heatmap for a contingency table, masking zero values and highlighting them with a specific color.

    Parameters:
    - feature1 (str): The name of the first feature (used for the y-axis labels).
    - feature2 (str): The name of the second feature (used for the x-axis labels).
    - contingency_table (pd.DataFrame): A contingency table (cross-tabulation) of feature1 and feature2.

    Returns:
    - Displays the heatmap plot.
    """

   
    masked_data = contingency_table.mask(contingency_table == 0)

    plt.figure(figsize=(20, 8))

    # Heatmap 
    ax = sns.heatmap(masked_data, annot=False, fmt="d", cmap="YlGnBu", cbar=True, 
                     cbar_kws={'label': 'Frequency'}, linewidths=0.5, linecolor='black')

    # Specific color for zero values using a second layer of heatmap
    sns.heatmap(contingency_table.applymap(lambda x: np.nan if x != 0 else 0), 
                annot=False, fmt="d", cmap=ListedColormap(['lightgrey']), 
                cbar=False, ax=ax, linewidths=0.5, linecolor='black')

 
    plt.title(f'Heatmap of {feature1} vs. {feature2}')
    plt.xlabel(feature2)
    plt.ylabel(feature1)


    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_categorical_target_relation(categorical_var, target_var, df):
    """
    Plot the relationship between a categorical variable and a target variable using a stacked horizontal bar chart.

    Parameters:
    - categorical_var (str): The name of the categorical variable to analyze.
    - target_var (str): The name of the target variable to analyze the relationship with.
    - df (pd.DataFrame): The DataFrame containing the data for the variables.

    Returns:
    - Displays the stacked horizontal bar chart.
    """

    df_copy = df.copy()
    df_copy[categorical_var] = df_copy[categorical_var].fillna('Missing')
    
    # Absolute counts and percentages for each category
    count_data = df_copy.groupby([categorical_var, target_var]).size().unstack(fill_value=0)
    percent_data = count_data.apply(lambda x: x / x.sum() * 100, axis=1)

    ax = percent_data.plot(kind='barh', stacked=True, figsize=(12, 4), colormap='viridis')
    plt.title(f'Relationship between {categorical_var} and {target_var} (%)')
    plt.xlabel('%', fontsize=7)
    ax.set_ylabel(target_var, fontsize=7)  # Set y-axis label with smaller font size
    plt.xticks(fontsize=7)

    new_labels = [textwrap.fill(label.get_text(), 20) for label in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks())  # Set the tick positions
    ax.set_yticklabels(new_labels, fontsize=7)  # Set the wrapped labels

    for i, (index, row) in enumerate(count_data.iterrows()):
        total_count = row.sum()
        ax.text(100, i, f'{int(total_count)}', va='center', ha='left', color='black', fontsize=7)

    legend = plt.legend(
        title=target_var, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), 
        fontsize=6,  
        title_fontsize=6  
    )
    legend.get_frame().set_linewidth(0.5)  

    plt.tight_layout()
    plt.show()


def plot_numerical_target_relation(df, numerical_var, target_var, percentile=0.95):
    """
    Plot the distribution of a numerical variable across each class of a target variable using a box plot.
    The plot focuses on the first specified percentile of the numerical variable to reduce the impact of outliers.

    Parameters:
    - numerical_var (str): The name of the numerical variable to analyze.
    - target_var (str): The name of the target variable to analyze the relationship with.
    - percentile (float): The percentile threshold to filter the numerical variable (default: 0.95).

    Returns:
    - Displays the box plot.
    """

    # Filter data to the first specified percentile
    threshold = df[numerical_var].quantile(percentile)
    df_filtered = df[df[numerical_var] <= threshold]

    # Target variable sorted in alphabetical order
    category_order = sorted(df_filtered[target_var].unique())

    plt.figure(figsize=(12, 6))
    
    sns.boxplot(
        data=df_filtered,
        y=target_var,
        x=numerical_var,
        order=category_order,
        palette=sns.color_palette("viridis", len(category_order))
    )

    plt.title(f'Relationship between {numerical_var} and {target_var} (First {int(percentile * 100)}%)', fontsize=14)
    plt.xlabel(numerical_var, fontsize=12)
    plt.ylabel(target_var, fontsize=12)
    plt.yticks(rotation=0)
    plt.show()

def cramers_v(x, y):
    """
    Calculate Cramér's V, a measure of association between two categorical variables.

    Parameters:
    - x (pd.Series or array-like): The first categorical variable.
    - y (pd.Series or array-like): The second categorical variable.

    Returns:
    - float: The Cramér's V value, ranging from 0 (no association) to 1 (perfect association).
    """
    
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def cramers_v_matrix(dataset, list_variables):
    """
    Compute a matrix of Cramér's V values for a list of categorical variables in a dataset.

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the categorical variables.
    - list_variables (list of str): List of column names to include in the Cramér's V matrix.

    Returns:
    - pd.DataFrame: A square DataFrame where each cell (i, j) contains the Cramér's V value 
      for the association between variables `list_variables[i]` and `list_variables[j]`.
    """

    cramers_v_matrix = pd.DataFrame(np.zeros((len(list_variables), len(list_variables))), 
                                index=list_variables, columns=list_variables)
    for col1 in list_variables:
        for col2 in list_variables:
            cramers_v_matrix.loc[col1, col2] = cramers_v(dataset[col1], dataset[col2])
    return cramers_v_matrix


def plot_missing_vs_existing_all_vars(df, target_var):
    """
    Plot the relationship between missing/existing values for all variables and a target variable.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to analyze.
    - target_var (str): The name of the target variable to analyze against missing/existing statuses.

    Returns:
    - None: Displays a series of stacked bar plots for each variable in the dataset.

    """

    # Treat 'UK' as missing for 'Medical Fee Region' and 'U' as missing for 'Gender'
    df = df.copy() 
    
    # All columns except the target variable
    variables = [col for col in df.columns if col != target_var]
    num_vars = len(variables)
    cols = 2  
    rows = (num_vars + 1) // cols  
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() 

    for i, variable in enumerate(variables):
        df_copy = df.copy()
        
        df_copy['status_flag'] = df_copy[variable].apply(
            lambda x: 'Missing' if pd.isna(x) else ('Is 0' if x == 0 else 'Existing')
        )

        # Counts and percentages for each target class within Missing, Is 0, and Existing
        count_data = df_copy.groupby(['status_flag', target_var]).size().unstack(fill_value=0)
        percent_data = count_data.apply(lambda x: x / x.sum() * 100, axis=1)

        ax = percent_data.plot(kind='bar', stacked=True, colormap='viridis', ax=axes[i])
        ax.set_title(f'Relation between presence/absence of {variable} and {target_var}')
        ax.set_ylabel('Percentage')
        ax.legend(title=target_var, bbox_to_anchor=(1, 0.5), loc='center left')
        
        ax.set_xticklabels(count_data.index, rotation=0)
        
        for j, (index, row) in enumerate(count_data.iterrows()):
            total_count = row.sum()
            ax.text(j, 100, f'Total: {int(total_count)}', ha='center', va='bottom', color='black', fontsize=9)
    
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    plt.show()


def create_indicator_features(df, features, indicator_type='missing'):
    """
    Create indicator features for missing or zero values in specified columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns to process.
    - features (list of str): List of column names for which to create indicator features.
    - indicator_type (str): Type of indicator to create, either 'missing' (default) or 'zero'.

    Returns:
    - pd.DataFrame: The DataFrame with added indicator features.
    """

    for feature in features:
        if indicator_type == 'missing':
            # Missing indicator feature
            df[f'{feature} Missing'] = df[feature].apply(lambda x: 1 if pd.isna(x) else 0)
        elif indicator_type == 'zero':
            # Zero indicator feature
            df[f'{feature} Zero'] = df[feature].apply(lambda x: 0 if x == 0 else 1)
        else:
            raise ValueError("indicator_type must be 'missing' or 'zero'")
    
    return df





def calculate_date_difference(df, start_col, end_col, date_list):
    condition = pd.Series(True, index=df.index)  # Initialize with all True
    if not date_list:  # If no date list is provided, no filtering
        filtered_df = df
    else:
        for col in date_list:
            if end_col == 'C-2 Date' or col == 'Assembly Date':
                condition &= (df[end_col] <= df[col]) | (df[col].isna())
            else:
                condition &= (df[end_col] < df[col]) | (df[col].isna())
        filtered_df = df[condition]
        
    date_diff = (filtered_df[end_col] - filtered_df[start_col]).dt.days.dropna()
    return date_diff



def accident_date_imputation(df, ref_df, verbose=False):
    def impute_accident_date(row, medians):
        c2_cond = (row['C-2 Date'] <= row['C-3 Date'] or pd.isna(row['C-3 Date'])) and \
                  (row['C-2 Date'] <= row['Assembly Date'] or pd.isna(row['Assembly Date']))
        c3_cond = (row['C-3 Date'] < row['C-2 Date'] or pd.isna(row['C-2 Date'])) and \
                  (row['C-3 Date'] <= row['Assembly Date'] or pd.isna(row['Assembly Date']))

        if c2_cond:
            return row['C-2 Date'] - pd.to_timedelta(medians['c2'], unit='D')
        elif c3_cond:
            return row['C-3 Date'] - pd.to_timedelta(medians['c3'], unit='D')
        else:
            return row['Assembly Date'] - pd.to_timedelta(medians['assembly'], unit='D')

    # Calculate medians
    medians = {
        'c2': calculate_date_difference(ref_df, 'Accident Date', 'C-2 Date', ['C-3 Date', 'Assembly Date']).median(),
        'c3': calculate_date_difference(ref_df, 'Accident Date', 'C-3 Date', ['C-2 Date', 'Assembly Date']).median(),
        'assembly': calculate_date_difference(ref_df, 'Accident Date', 'Assembly Date', ['C-2 Date', 'C-3 Date']).median(),
    }

    # Apply imputation logic
    missing_accident_date_df = df[df['Accident Date'].isna()].copy()
    missing_accident_date_df['Accident Date'] = missing_accident_date_df.apply(
        lambda row: impute_accident_date(row, medians), axis=1
    )

    # Update original DataFrame
    new_df = df.copy()
    new_df.loc[missing_accident_date_df.index, 'Accident Date'] = missing_accident_date_df['Accident Date']

    if verbose:
        for key, diff in medians.items():
            print(f"Median Accident to {key.capitalize()} Difference: {diff:.2f} days")

    return new_df





def other_dates_imputation(df, ref_df, date_to_impute, verbose = False):
    """
    Imputes missing dates in a DataFrame based on the median difference between a reference date 
    ('Accident Date') and the target date to impute ('date_to_impute').

    """

    accident_to_date =  calculate_date_difference(ref_df, 'Accident Date', date_to_impute, [])
    accident_to_date_median = accident_to_date.median()

    missing_date_df = df[df[date_to_impute].isna()].copy()

    missing_date_df[date_to_impute] = missing_date_df['Accident Date'] + pd.to_timedelta(accident_to_date_median, unit='D')

    new_df = df.copy()
    new_df.loc[missing_date_df.index, date_to_impute] = missing_date_df[date_to_impute]

    if verbose:
        print(f"Description of 'Accident to {date_to_impute}' in days:")
        print(accident_to_date.describe().round(0).astype(int))

    return new_df


def year_differences(df, ref_df, col_to_impute, col_to_use):
    new_df = df.copy()

    # Impute based on year differences
    condition = new_df[col_to_use].notna()
    new_df.loc[condition, col_to_impute] = new_df.loc[condition, col_to_impute].fillna(
        pd.to_datetime(new_df['Accident Date']).dt.year - new_df.loc[condition, col_to_use]
    )

    # Fallback: Fill remaining with median
    median = ref_df[col_to_impute].median()
    new_df[col_to_impute] = new_df[col_to_impute].fillna(median)

    return new_df


def impute_with_group_modes(df, ref_df, target_col, group_cols, fallback_group_col):
    """
    Impute missing values in the target column based on modes calculated within groups.

    """
    new_df = df.copy()

    # Compute modes for primary groups
    primary_group_mode = (
        ref_df.groupby(group_cols)[target_col]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .to_dict()
    )

    #Compute modes for the fallback group
    fallback_group_mode = (
        ref_df.groupby(fallback_group_col)[target_col]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .to_dict()
    )

    #Overall mode for final fallback
    overall_mode = ref_df[target_col].mode()[0]

    #Impute using primary group mode
    new_df[target_col] = new_df.apply(
        lambda row: primary_group_mode.get(tuple(row[col] for col in group_cols), row[target_col])
        if pd.isna(row[target_col])
        else row[target_col],
        axis=1,
    )

    #Impute using fallback group mode
    new_df[target_col] = new_df.apply(
        lambda row: fallback_group_mode.get(row[fallback_group_col], row[target_col])
        if pd.isna(row[target_col])
        else row[target_col],
        axis=1,
    )

    # Impute using overall mode for any remaining NaNs
    new_df[target_col] = new_df[target_col].fillna(overall_mode)

    return new_df


def impute_with_group_mean(df, ref_df, target_col, group_cols):
    """
    Impute missing values in the target column based on the mean within specified groups.
    Optionally treat zeros as missing values by replacing them with NaNs before imputation.

    """
    new_df = df.copy()
    new_ref_df = ref_df.copy()

    # Step 2: Compute group means
    group_means = new_ref_df.groupby(group_cols)[target_col].mean().to_dict()

    # Step 4: Impute missing values
    new_df[target_col] = new_df.apply(
        lambda row: group_means[row[group_cols]] 
        if pd.isnull(row[target_col]) else row[target_col], 
        axis=1)

    return new_df



def impute_with_random_forest(df, ref_df, target_col, predictors, zero_to_nan=True):
    """
    Impute missing values in a target column using a Random Forest regression model 
    with specified predictors.

    """

    new_ref_df = ref_df.copy()
    new_df = df.copy()

    # Separating data into training and missing subsets 
    train_df = new_ref_df[~new_ref_df[target_col].isna()]
    test_df_with_na = new_df[new_df[target_col].isna()]

    # Default regression model pipeline 
    # Preprocessing steps for numeric and categorical predictors
    categorical_features = [col for col in predictors if new_ref_df[col].dtype == 'object']
    
    # ColumnTransformer directly using SimpleImputer and OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough' )

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=10, random_state=0))])

    # Training the model on rows with known target values
    X_train_impute = train_df[predictors]
    y_train_impute = train_df[target_col]
    model.fit(X_train_impute, y_train_impute)

    # Predicting missing values for training set
    new_df.loc[test_df_with_na.index, target_col] = model.predict(test_df_with_na[predictors])

    return new_df


def replace_nan_with_zeros(df, features):
    """
    Replace NaN values with 0 in the specified columns.

    """

    for feature in features:
        df[feature] = df[feature].replace(np.nan, 0)
    return df


def impute_mode(df, ref_df, feature):
    """
    Impute missing values in a specified column using the mode from a reference DataFrame.
    """
    new_df = df.copy()
    mode_value = ref_df[feature].mode()[0]
    new_df.loc[new_df[feature].isna(), feature] = mode_value
    return new_df


def outliers(X, y, ref_X, features, lower_p, upper_p, drop=True):
    new_X = X.copy()
    new_y = y.copy()

    for feature in features:
        Q1 = ref_X[feature].quantile(0.25)
        Q3 = ref_X[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        lower_threshold = ref_X[feature].quantile(lower_p)
        upper_threshold = ref_X[feature].quantile(upper_p)

        # Drop rows only if upper bound exceeds upper threshold
        if upper_bound > upper_threshold:
            if drop:
                claims_to_drop = new_X[new_X[feature] > upper_bound].index
                new_X = new_X.drop(index=claims_to_drop)
                new_y = new_y.drop(index=claims_to_drop)
        new_X[feature] = new_X[feature].apply(lambda x: min(x, upper_threshold))

        # Drop rows only if lower bound is below lower threshold
        if lower_bound < lower_threshold:
            if drop:
                claims_to_drop = new_X[new_X[feature] < lower_bound].index
                new_X = new_X.drop(index=claims_to_drop)
                new_y = new_y.drop(index=claims_to_drop)
        new_X[feature] = new_X[feature].apply(lambda x: max(x, lower_threshold))

    return new_X, new_y


def count_encoding(df, col_list):
    """
    Perform count encoding on the specified columns of a DataFrame.

    """
    new_df = df.copy()
    
    for col in col_list:
        # Compute the counts of each category
        counts = df[col].value_counts()
        # Map the counts to the column and ensure they are integers
        new_df[f'ce_{col}'] = df[col].map(counts).astype(int)
    
    return new_df[[f'ce_{col}' for col in col_list]]


# Association between missing values and a categorical variable and plot a heatmap
def analyze_missing_association(df, target_col, cat_var, alpha=0.05):
    """
    Analyzes the association between missing values in a target column and a categorical variable
    using a chi-square test and displays a heatmap of the contingency table.

    """
    # Mask for missing values in the target column
    missing_mask = df[target_col].isna()

    # Contingency table between missing status and the categorical variable
    contingency_table = pd.crosstab(missing_mask, df[cat_var])

    # Chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

    if p_val < alpha:
        print(f"'{target_col}' and '{cat_var}': The Chi Square is {chi2_stat.round(2)} and the p_value < {alpha} so there's a strong relationship.")
    else:
        print(f"'{target_col}' and '{cat_var}': The Chi Square is {chi2_stat} and the p_value > {alpha} so there's no significant relationship.")


def one_hot_encoding(df, col_list):
    """
    Perform one-hot encoding on the specified columns of a DataFrame.
    """

    ohc = OneHotEncoder(sparse_output=False)
    ohc_feat = ohc.fit_transform(df[col_list])
    ohc_feat_names = ohc.get_feature_names_out()
    new_df = pd.DataFrame(ohc_feat, index=df.index, columns = ohc_feat_names)
    rename_ohc_cols = {}
    for i in col_list:
        for j in new_df.columns[new_df.columns.str.startswith(i)].to_list():
            rename_ohc_cols[j]='oh_' + j
    new_df.rename(columns=rename_ohc_cols, inplace=True)
    return new_df


def binary_categ_treatment(df, cols):
    """
    Convert binary categorical values in a column to numeric values (0 and 1).
    """

    new_df = df.copy()

    for col in cols:
        # Specific case for First Report Submitter as it is not Y/N
        if col == 'First Report Submitter':
            new_df[col] = new_df[col].replace(['Employee', 'Employer'], [1, 0]).astype(int)
        # Mapping Yes to 1 and No to 0
        elif col =='Gender':
            new_df[col] = new_df[col].replace(['F', 'M'], [1, 0]).astype(int)
    
        else:
            new_df[col] = new_df[col].replace(['Y', 'N'], [1, 0]).astype(int)

    return new_df

def treat_zip_code(df):
    feature = 'Zip Code'
    new_X = df.copy()

    # Ensure the column is treated as object initially
    new_X[feature] = new_X[feature].astype('object')

    # Replace text-based invalid entries with NaN
    is_text = new_X[feature].apply(lambda x: isinstance(x, str) and not x.replace('.', '', 1).isdigit())
    new_X.loc[is_text, feature] = np.nan

    # Convert valid entries to numeric for processing
    new_X[feature] = pd.to_numeric(new_X[feature], errors='coerce')

    # Handle numeric adjustments
    new_X.loc[new_X[feature] < 100, feature] = np.nan
    new_X.loc[new_X[feature] < 10000, feature] *= 10

    # Convert numeric values back to strings and ensure column is object
    new_X[feature] = new_X[feature].apply(lambda x: str(int(x)) if pd.notna(x) else np.nan).astype('object')

    return new_X



### Functions Preprocessing
def preprocessing(X_train, X_val, y_train, scaler=None):

    new_X_train = X_train.copy()
    new_X_val = X_val.copy()
    new_y_train = y_train.copy()

    reference_date = datetime.datetime(2020,1,1)
    
    # Setting claim identifier as index
    if 'Claim Identifier' in new_X_val.columns:
        new_X_val.set_index('Claim Identifier', inplace=True)

    # Treating date types
    date_variables = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']
    for column in date_variables:
        new_X_val[column] = pd.to_datetime(new_X_val[column], errors='coerce')
    
    # Treating COVID 19 Indicator
    new_X_train.loc[(new_X_train['COVID-19 Indicator'] != 'Y') & (new_X_train['WCIO Nature of Injury Description'] == 'COVID-19'), 'COVID-19 Indicator'] = 'Y' # tirar depois
    new_X_val.loc[(new_X_val['COVID-19 Indicator'] != 'Y') & (new_X_val['WCIO Nature of Injury Description'] == 'COVID-19'), 'COVID-19 Indicator'] = 'Y'

    # Treating zip codes
    new_X_val = treat_zip_code(new_X_val)

    # Treating Carrier Type
    new_X_val["Carrier Type"] = new_X_val["Carrier Type"].apply(lambda x: "5X SPECIAL FUND" if str(x).startswith("5") else x)

    # Replacing 0 and unknown values to missing
    columns_of_interest = ['Age at Injury', 'Birth Year']
    new_X_val = replace_zeros_with_nan(new_X_val, columns_of_interest)
    new_X_val = replace_unknown_with_na(new_X_val, 'Medical Fee Region', 'UK')
    new_X_val = replace_unknown_with_na(new_X_val, 'Gender', 'U') 
    new_X_val = replace_unknown_with_na(new_X_val, 'Gender', 'X')
    new_X_val = replace_unknown_with_na(new_X_val, 'Alternative Dispute Resolution', 'U')
    new_X_val = replace_unknown_with_na(new_X_val, 'Carrier Type', 'UNKNOWN')

    # Outliers
    binary_features = [col for col in new_X_train.columns if new_X_train[col].nunique() == 2]
    numerical_features = [ col for col in new_X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns
                            if col not in binary_features and (not col.endswith('Code'))]

    new_X_val, new_y_val = outliers(new_X_val, new_y_train, new_X_train, numerical_features, 0.005, 0.995, drop=False)
    new_X_train, new_y_train = outliers(new_X_train, new_y_train, new_X_train, numerical_features, 0.005, 0.995)


    # Missing Values
    # Creating features regarding missing values
    features_of_interest = ['C-2 Date', 'C-3 Date', 'First Hearing Date', 'Average Weekly Wage', 'IME-4 Count', 'Industry Code Description', 'WCIO Part Of Body Description', 'Carrier Type']
    new_X_val = create_indicator_features(new_X_val, features_of_interest, indicator_type='missing')
    new_X_val = create_indicator_features(new_X_val, ['Average Weekly Wage'], indicator_type='zero')

    # Treat missing values of dates
    date_columns = ['Accident Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date', 'Assembly Date']
    for col in date_columns:
        if col in new_X_train.columns:
            new_X_train[col] = pd.to_datetime(new_X_train[col])
        if col in new_X_val.columns:
            new_X_val[col] = pd.to_datetime(new_X_val[col])

    new_X_val = accident_date_imputation(new_X_val, new_X_train)
    new_X_train = accident_date_imputation(new_X_train, new_X_train)

    new_X_val = other_dates_imputation(new_X_val, new_X_train, 'C-2 Date')
    new_X_train = other_dates_imputation(new_X_train, new_X_train, 'C-2 Date')

    new_X_val = other_dates_imputation(new_X_val, new_X_train, 'C-3 Date')
    new_X_train = other_dates_imputation(new_X_train, new_X_train, 'C-3 Date')

    # Treat missing values of Birth Year and Age at Injury
    new_X_train = year_differences(new_X_train, new_X_train, 'Age at Injury', 'Birth Year')
    new_X_train = year_differences(new_X_train, new_X_train, 'Birth Year', 'Age at Injury')

    new_X_val = year_differences(new_X_val, new_X_train, 'Age at Injury', 'Birth Year')
    new_X_val = year_differences(new_X_val, new_X_train, 'Birth Year', 'Age at Injury')

    # Drop rows with nan in all codes and descriptions
    filtered_indices = new_X_train[~(new_X_train['Industry Code'].isna() & 
            new_X_train['Industry Code Description'].isna() & 
            new_X_train['WCIO Cause of Injury Code'].isna() & 
            new_X_train['WCIO Cause of Injury Description'].isna() & 
            new_X_train['WCIO Nature of Injury Code'].isna() & 
            new_X_train['WCIO Nature of Injury Description'].isna() & 
            new_X_train['WCIO Part Of Body Code'].isna() &  
            new_X_train['WCIO Part Of Body Description'].isna())].index
    new_X_train = new_X_train.loc[filtered_indices]
    new_y_train = new_y_train.loc[filtered_indices]

    # Treat missing values with group mode
    new_X_train = impute_mode(new_X_train, new_X_train, 'Carrier Type') 
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'Industry Code Description', ['Carrier Type'], 'Industry Code Description')

    new_X_val = impute_mode(new_X_val, new_X_train, 'Carrier Type') 
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'Industry Code Description', ['Carrier Type'], 'Industry Code Description')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'Gender', ['Industry Code Description'], 'Gender')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'Medical Fee Region', ['County of Injury', 'District Name'], 'Medical Fee Region')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'Zip Code', ['County of Injury', 'District Name'], 'Zip Code')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'WCIO Cause of Injury Description', ['Industry Code Description'], 'WCIO Cause of Injury Description')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'WCIO Nature of Injury Description', ['Industry Code Description'], 'WCIO Nature of Injury Description')
    new_X_val = impute_with_group_modes(new_X_val, new_X_train, 'WCIO Part Of Body Description', ['Industry Code Description'], 'WCIO Part Of Body Description')

    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'Gender', ['Industry Code Description'], 'Gender')
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'Medical Fee Region', ['County of Injury', 'District Name'], 'Medical Fee Region')
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'Zip Code', ['County of Injury', 'District Name'], 'Zip Code')
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'WCIO Cause of Injury Description', ['Industry Code Description'], 'WCIO Cause of Injury Description')
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'WCIO Nature of Injury Description', ['Industry Code Description'], 'WCIO Nature of Injury Description')
    new_X_train = impute_with_group_modes(new_X_train, new_X_train, 'WCIO Part Of Body Description', ['Industry Code Description'], 'WCIO Part Of Body Description')

    # Replacing nan as 0 for IME-4 Count
    new_X_train = replace_nan_with_zeros(new_X_train, ['IME-4 Count'])
    new_X_val = replace_nan_with_zeros(new_X_val, ['IME-4 Count'])

    # Treat missing values with mode
    new_X_val = impute_mode(new_X_val, new_X_train, 'Alternative Dispute Resolution')
    new_X_train = impute_mode(new_X_train, new_X_train, 'Alternative Dispute Resolution')

    # Treat aww with random forest
    top_50_counties = (new_X_train['County of Injury'].value_counts().head(50).index)
    new_X_train['County of Injury (Imputation)'] = new_X_train['County of Injury'].apply(lambda x: x if x in top_50_counties else None)
    new_X_val['County of Injury (Imputation)'] = new_X_val['County of Injury'].apply(lambda x: x if x in top_50_counties else None)
    
    new_X_train = replace_zeros_with_nan(new_X_train, ['Average Weekly Wage'])
    new_X_val= replace_zeros_with_nan(new_X_val, ['Average Weekly Wage'])

    new_X_val = impute_with_random_forest(new_X_val, new_X_train, 'Average Weekly Wage', ["Industry Code Description", "Gender", "Attorney/Representative", "Carrier Type", "Birth Year", "County of Injury (Imputation)"])
    new_X_train = impute_with_random_forest(new_X_train, new_X_train, 'Average Weekly Wage', ["Industry Code Description", "Gender", "Attorney/Representative", "Carrier Type", "Birth Year", "County of Injury (Imputation)"])

    new_X_val.drop(columns='County of Injury (Imputation)', inplace=True)
    new_X_train.drop(columns='County of Injury (Imputation)', inplace=True)

    # Feature Engineering
    new_X_train['Days until First Report'] = (new_X_train[['C-3 Date', 'C-2 Date']].min(axis=1) - new_X_train['Accident Date']).dt.days
    new_X_val['Days until First Report'] = (new_X_val[['C-3 Date', 'C-2 Date']].min(axis=1) - new_X_val['Accident Date']).dt.days

    new_X_train['Days until Assembly'] = (new_X_train['Assembly Date'] - new_X_train['Accident Date']).dt.days
    new_X_val['Days until Assembly'] = (new_X_val['Assembly Date'] - new_X_val['Accident Date']).dt.days

    new_X_train['First Report Submitter'] = new_X_train.apply(lambda row: 'Employer' if pd.isna(row['C-3 Date']) or (row['C-2 Date'] <= row['C-3 Date']) else 'Employee', axis=1)
    new_X_val['First Report Submitter'] = new_X_val.apply(lambda row: 'Employer' if pd.isna(row['C-3 Date']) or (row['C-2 Date'] <= row['C-3 Date']) else 'Employee', axis=1)

    new_X_train['Assembly Quarter'] = new_X_train['Assembly Date'].dt.quarter
    new_X_val['Assembly Quarter'] = new_X_val['Assembly Date'].dt.quarter

    new_X_train['Assembly Date Referenced'] = ( new_X_train['Assembly Date']-reference_date).dt.days
    new_X_val['Assembly Date Referenced'] = ( new_X_val['Assembly Date']-reference_date).dt.days

    new_X_train['Accident Date Referenced'] = (new_X_train['Accident Date'] - reference_date).dt.days
    new_X_val['Accident Date Referenced'] = (new_X_val['Accident Date'] - reference_date).dt.days

    new_X_train['First Report Referenced'] = new_X_train.apply(
        lambda row: (row['C-3 Date'] - reference_date) 
        if pd.isna(row['C-2 Date']) or (pd.notna(row['C-3 Date']) and row['C-2 Date'] <= row['C-3 Date']) 
        else (row['C-2 Date'] - reference_date), 
        axis=1).dt.days
    new_X_val['First Report Referenced'] = new_X_val.apply(
        lambda row: (row['C-3 Date'] - reference_date) 
        if pd.isna(row['C-2 Date']) or (pd.notna(row['C-3 Date']) and row['C-2 Date'] <= row['C-3 Date']) 
        else (row['C-2 Date'] - reference_date), 
        axis=1
    ).dt.days

    new_X_train['Log Average Weekly Wage'] = new_X_train['Average Weekly Wage'].apply(lambda x: 0 if x == 0 else np.log(x))
    new_X_val['Log Average Weekly Wage'] = new_X_val['Average Weekly Wage'].apply(lambda x: 0 if x == 0 else np.log(x))


    # Dealing with categoricals
    binary_columns = ['Alternative Dispute Resolution', 'Attorney/Representative', 'First Report Submitter', 'COVID-19 Indicator', 'Gender']
    new_X_train = binary_categ_treatment(new_X_train, binary_columns)
    new_X_val = binary_categ_treatment(new_X_val, binary_columns)

    # Encoding
    binary_features = [col for col in new_X_train.columns if new_X_train[col].nunique() == 2]
    numerical_features = [ col for col in new_X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns
                            if col not in binary_features and (not col.endswith('Code'))]
    date_features = [col for col in new_X_train.columns if col.endswith('Date')]
    categorical_features = [
    col for col in new_X_train.columns 
    if (col not in numerical_features + binary_features + date_features) 
    and (col == 'Zip Code' or not col.endswith('Code'))]

    new_X_train_encoded = count_encoding(new_X_train, categorical_features)
    new_X_val_encoded = count_encoding(new_X_val, categorical_features)
    new_X_train = pd.concat([new_X_train, new_X_train_encoded], axis=1)
    new_X_val = pd.concat([new_X_val, new_X_val_encoded], axis=1)

    # Scaling
    if scaler != None:
        binary_features = [col for col in new_X_train.columns if new_X_train[col].nunique() == 2]
        numerical_features =[col for col in new_X_train.columns 
                        if col not in binary_features and (col.startswith('ce_') or (col in new_X_train.select_dtypes(include=['int64', 'float64']).columns and not col.endswith('Code')))]

        new_X_train[numerical_features] = pd.DataFrame(scaler.fit_transform(new_X_train[numerical_features]), 
                                        index=new_X_train.index, 
                                        columns=numerical_features)
                                        
        new_X_val[numerical_features] = pd.DataFrame(scaler.transform(new_X_val[numerical_features]), 
                                    index=new_X_val.index, 
                                    columns=numerical_features)
    

    return new_X_train, new_X_val, new_y_train


        
### FEATURE SELECTION -----------------------------------------------------

def cor_heatmap(cor):
    plt.figure(figsize=(16,14))
    sns.heatmap(data=cor, annot=True, cmap="coolwarm", center=0, fmt='.2f')
    plt.show()


def rfe_feature_selection(X, y, model, categorical_features=None, scoring='f1_macro', cv=5):
    """
    Perform Recursive Feature Elimination (RFE) with a user-defined model and handle categorical features.

    """
    # Set categorical features for models that support them (e.g., LightGBM, CatBoost)
    if categorical_features and hasattr(model, 'set_params'):
        model.set_params(categorical_feature=categorical_features)
    # List to store cross-validation scores
    scores = []
    num_features_list = list(range(1, X.shape[1] + 1))
    # Perform RFE for each number of features
    for n_features in num_features_list:
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=2)
        X_rfe = rfe.fit_transform(X, y)
        # Cross-validation score
        score = np.mean(cross_val_score(model, X_rfe, y, scoring=scoring, cv=cv))
        scores.append(score)
    # Find the optimal number of features
    optimal_index = np.argmax(scores)
    optimal_n_features = num_features_list[optimal_index]
    # Perform RFE with the optimal number of features
    rfe = RFE(estimator=model, n_features_to_select=optimal_n_features)
    rfe.fit(X, y)
    # Create a DataFrame with feature rankings
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values(by='Ranking', ascending=True)
    # Create a DataFrame of cross-validation scores
    scores_df = pd.DataFrame({
        'Num Features': num_features_list,
        'CV Score': scores
    })
    return optimal_n_features, scores_df, feature_ranking


def cramers_v(x, y):

    # Create a contingency table
    contingency_table = pd.crosstab(x, y)
    # Compute Chi-Square statistic
    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
    # Compute Cramér's V
    n = contingency_table.sum().sum()  # Total sample size
    k = min(contingency_table.shape)  # Minimum dimension of the table
    return np.sqrt(chi2_stat / (n * (k - 1)))
    

def cramers_v_heatmap(data):
    """
    Calculates Cramér's V for all pairs of categorical variables and plots a heatmap.
    """
    # Create a DataFrame to store Cramér's V values
    columns = data.columns
    cramers_v_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), 
                                    index=columns, columns=columns)
    # Compute Cramér's V for each pair of variables
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                cramers_v_matrix.loc[col1, col2] = 1.0  # Perfect correlation with itself
            else:
                cramers_v_matrix.loc[col1, col2] = cramers_v(data[col1], data[col2])
    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(cramers_v_matrix, annot=True, cmap='coolwarm', center=0, vmin=0, vmax=1)
    plt.title('Cramér\'s V Heatmap of Categorical Variables', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return cramers_v_matrix


def select_features_anova(X, y, weighted=False):
    # Initialize SelectKBest with F-test
    if weighted:
        class_weights = {label: 1.0 / count for label, count in zip(*np.unique(y, return_counts=True))}
        sample_weights = y.map(class_weights) if isinstance(y, pd.Series) else np.vectorize(class_weights.get)(y)
        fs = SelectKBest(score_func=lambda X, y: f_classif(X, y, sample_weight=sample_weights), k='all')
    else:
        fs = SelectKBest(score_func=f_classif, k='all')
    
    # Fit to the data
    fs.fit(X, y)

    # Get F-scores as a Pandas Series
    f_scores = pd.Series(fs.scores_, index=X.columns)
    
    # Calculate the minimum score threshold
    min_score = f_scores.max() * (1/len(f_scores))
    
    # Select features with scores above the threshold
    selected_features = f_scores[f_scores > min_score].index.tolist()
    
    return selected_features


def chi_squared_feature_selection(X, y, weighted=False):
    # Compute Chi-Square scores and p-values
    if weighted:
        class_weights = {label: 1.0 / count for label, count in zip(*np.unique(y, return_counts=True))}
        sample_weights = y.map(class_weights) if isinstance(y, pd.Series) else np.vectorize(class_weights.get)(y)
        weighted_X = X.mul(sample_weights, axis=0)
        chi2_scores, p_values = chi2(weighted_X, y)
    else:
        chi2_scores, p_values = chi2(X, y)
    
    # Create a DataFrame to summarize the results
    chi2_results = pd.DataFrame({
        'Feature': X.columns,
        'Chi2 Score': chi2_scores,
        'P-Value': p_values
    }).sort_values(by='Chi2 Score', ascending=False)
    
    # Filter features with p-value < 0.05
    selected_features = chi2_results[chi2_results['P-Value'] < 0.05]['Feature']
    
    # Return the selected features as a list
    return selected_features.tolist()


def select_features_elasticnet(X, y, l1_ratio=0.5, random_state=42, weighted=False):
    # Initialize Logistic Regression with ElasticNet regularization
    if weighted:
        log_reg = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            random_state=random_state,
            multi_class='multinomial',
            class_weight='balanced'
        )
    else:
        log_reg = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=l1_ratio,
            random_state=random_state,
            multi_class='multinomial'
        )

    # Fit the model to the data
    log_reg.fit(X, y)

    # Compute absolute mean of coefficients for feature importance
    feature_importances = np.abs(log_reg.coef_).mean(axis=0)

    # Convert feature importances to a Pandas Series
    feature_importances = pd.Series(feature_importances, index=X.columns)

    # Calculate the minimum importance threshold
    min_importance = feature_importances.max() * (1/len(feature_importances))

    # Select features with importance above the threshold
    selected_features = feature_importances[feature_importances > min_importance].index.tolist()

    return selected_features


def select_features_lasso(X, y, threshold_factor=0.01, random_state=42, weighted=False):
    # Initialize Logistic Regression with L1 regularization
    if weighted:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',  # Suitable for L1 regularization
            random_state=random_state,
            class_weight='balanced'
        )
    else:
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',  # Suitable for L1 regularization
            random_state=random_state
        )
    
    # Fit the model to the data
    model.fit(X, y)
    
    # Get absolute coefficients for feature importance
    coefficients = np.abs(model.coef_[0])
    
    # Convert to a Pandas Series with feature names
    feature_importances = pd.Series(coefficients, index=X.columns)
    
    # Calculate the minimum coefficient threshold
    min_coefficient = feature_importances.max() * (1/len(feature_importances))
    
    # Select features with coefficients above the threshold
    selected_features = feature_importances[feature_importances > min_coefficient].index.tolist()
    
    return selected_features


def select_features_random_forest(X, y, random_state=42, weighted=False):
    # Initialize Random Forest Classifier
    if weighted:
        model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    else:
        model = RandomForestClassifier(random_state=random_state)
    
    # Fit the model to the data
    model.fit(X, y)
    
    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    
    # Calculate the minimum importance threshold
    min_importance = importances.max() * (1/len(importances))
    
    # Select features with importance above the threshold
    selected_features = importances[importances > min_importance].index.tolist()
    
    return selected_features


def feature_selection(X_train, y_train, weighted=False):
    new_X_train = X_train.copy()
    
    binary_features = [col for col in new_X_train.columns if new_X_train[col].nunique() == 2]

    numerical_features = [col for col in new_X_train.columns 
                        if col not in binary_features and (col.startswith('ce_') or (col in new_X_train.select_dtypes(include=['int64', 'float64']).columns and not col.endswith('Code')))]


    date_features = [col for col in new_X_train.columns if col.endswith('Date')]

    columns_to_drop = date_features + [col for col in new_X_train.columns if col.endswith('Code') and col != 'Zip Code'  and col != 'ce_Zip Code']
    new_X_train.drop(columns=columns_to_drop, inplace=True)
    numerical_features = [col for col in numerical_features if col not in columns_to_drop]

    new_X_train_num = new_X_train[numerical_features]
    new_X_train_bin = new_X_train[binary_features]
    new_X_train_num_bin = pd.concat([new_X_train[numerical_features], new_X_train[binary_features]], axis=1)

    for feature in new_X_train_num.columns: 
        if new_X_train_num[feature].var() == 0: 
            new_X_train.drop(columns=feature, inplace=True)
            new_X_train_num.drop(columns=feature, inplace=True)  
    
    selected_feats_anova = select_features_anova(new_X_train_num_bin, y_train, weighted)
    selected_feats_chi = chi_squared_feature_selection(new_X_train_bin, y_train, weighted)
    selected_feats_elastic = select_features_elasticnet(new_X_train_num_bin, y_train, weighted)
    selected_feats_lasso = select_features_lasso(new_X_train_num_bin, y_train, weighted)
    selected_feats_rf = select_features_random_forest(new_X_train_num_bin, y_train, weighted)
    
    selected_feature_sets = [
    set(selected_feats_anova),
    set(selected_feats_chi),
    set(selected_feats_elastic),
    set(selected_feats_lasso),
    set(selected_feats_rf)
    ]

    # Union of all features (unique)
    all_features = set.union(*selected_feature_sets)

    # Counting of how many times a feature appears
    feature_counts = {feature: sum(feature in s for s in selected_feature_sets) for feature in all_features}

    # Selecting features selected by the majority of the models
    majority_threshold = (len(selected_feature_sets)-1) / 2

    majority_features = [feature for feature, count in feature_counts.items() if count >= majority_threshold]

    new_X_train = new_X_train[majority_features]

    numerical_majority_features = [feature for feature in majority_features if feature in numerical_features]

    print('All feature selection methods applied.')

    print(f'Numerical majority features: {numerical_majority_features}')

    # Calculate Spearman correlation matrix for numerical features
    numerical_corr = new_X_train[numerical_majority_features].corr(method='spearman').abs()

    # Find pairs of highly correlated features
    to_drop = set()
    for i in range(len(numerical_corr.columns)):
        for j in range(i + 1, len(numerical_corr.columns)):
            if numerical_corr.iloc[i, j] > 0.8:
                col_to_drop = numerical_corr.columns[j]  # Drop the second feature
                to_drop.add(col_to_drop)
    print('Features to drop:', to_drop)

    # Drop highly correlated features
    new_X_train.drop(columns=to_drop, axis=1, inplace=True)

    return new_X_train


### MODEL -----------------------------------------------------------------

def cross_val(X, y, models_and_params, scaler=None, n_splits=5, n_iter_rs=50, metrics=None, balancing=True, weighted=False, selected_features=None):
    """
    Perform custom cross-validation with preprocessing, hyperparameter tuning, and train/validation scoring.
    """
    if metrics is None:
        metrics = {
            'accuracy': accuracy_score,
            'macro_f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
        }

    # Initialize KFold
    kf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    # Dictionary to store results for all models
    overall_results = {}
    parameter_scores = {} 
    all_selected_features = []

    # Reset indices for X and y
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):

        # Split the data into training and validation folds
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index].copy(), y.iloc[val_index].copy()

        # Preprocessing
        X_train, X_val, y_train = preprocessing(X_train, X_val, y_train, scaler)
        print(f"Preprocessing done for fold {fold + 1}")

        # Feature selection
        if selected_features is None:
            # Perform feature selection
            X_train = feature_selection(X_train, y_train, weighted)
            X_val = X_val[X_train.columns]  # Align validation set columns with selected features
            print(f"Feature selection done for fold {fold + 1}: Selected features: {list(X_train.columns)}")

            all_selected_features.append(list(X_train.columns))
        else:
            # Use pre-defined features
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            print(f"Using predefined features for fold {fold + 1}.")


        if balancing:
            # Balancing with TomekLinks
            tomek = TomekLinks()
            X_train, y_train = tomek.fit_resample(X_train, y_train)
            print(f'Balancing done for fold {fold + 1}')


        # Combine train and validation datasets for GridSearchCV
        X_combined = np.concatenate([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])

        # Create PredefinedSplit for train/validation split
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold=test_fold)

        # Evaluate each model
        for model_name, (model, param_grid) in models_and_params.items():
            print(f"Evaluating model: {model_name} on fold {fold + 1}")

            if model_name not in overall_results:
                overall_results[model_name] = {
                    'train_scores': [],
                    'val_scores': [],
                    'best_params': [],
                    'training_times': []
                }
            
            # Check if random search is needed
            if param_grid is not None and any(len(values) > 1 for values in param_grid.values()):
                print("Performing RandomizedSearchCV...")
                # RandomSearchCV
                random_search = RandomizedSearchCV(
                                estimator=model,
                                param_distributions=param_grid,  # Use param_distributions instead of param_grid
                                scoring="f1_macro",
                                cv=ps,                           # PredefinedSplit or your cross-validation strategy
                                n_iter=n_iter_rs,                       # Number of random combinations to try
                                random_state=42,                 # For reproducibility
                                verbose=1,                       # Optional: Increase verbosity to see progress
                                n_jobs=-1,                        # Use all available processors
                                refit=False
                            )

                # Train and time the process
                start_time = time.perf_counter()
                random_search.fit(X_combined, y_combined)
                elapsed_time = time.perf_counter() - start_time

                # Get best model and parameters
                best_params = random_search.best_params_
                best_model = model.set_params(**best_params) 
                
                best_model.fit(X_train, y_train)

                print(f"Best Parameters: {best_params}")
                for params, score in zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score']):
                    # Ensure model_name exists in the parameter_scores dictionary
                    if model_name not in parameter_scores:
                        parameter_scores[model_name] = {}  # Initialize as a dictionary for this model

                    # Create a hashable tuple of the sorted parameters
                    params_tuple = tuple(sorted(params.items()))  # Make params hashable

                    # Check if the parameter set (as tuple) exists, if not initialize as a list
                    if params_tuple not in parameter_scores[model_name]:
                        parameter_scores[model_name][params_tuple] = []  # Initialize an empty list for scores

                    # Append the score to the list for this parameter set
                    parameter_scores[model_name][params_tuple].append(score)

            else:
                print("Skipping RandomizedSearchCV, fitting model with default parameters...")
                # Fit model directly
                start_time = time.perf_counter()
                model.fit(X_combined, y_combined)
                elapsed_time = time.perf_counter() - start_time

                best_model = model
                best_params = param_grid  # or an empty dictionary if no search is performed

            # Predict on train and validation sets
            y_train_pred = best_model.predict(X_train)
            y_val_pred = best_model.predict(X_val)
            

            # Compute scores
            train_score = f1_score(y_train, y_train_pred, average="macro")
            val_score = f1_score(y_val, y_val_pred, average="macro")

            # Store results
            overall_results[model_name]['train_scores'].append(train_score)
            overall_results[model_name]['val_scores'].append(val_score)
            overall_results[model_name]['best_params'].append(best_params)
            overall_results[model_name]['training_times'].append(elapsed_time)
            print(f"  Fold {fold + 1} | Train F1: {train_score:.3f} | Val F1: {val_score:.3f} | Time: {elapsed_time:.2f}s")


    if selected_features is None:
        # Count occurrences of each feature across all folds
        feature_counts = {}
        for features in all_selected_features:
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Determine threshold for feature inclusion (e.g., at least 50% of folds)
        threshold = n_splits / 2
        
        # Filter features that meet the threshold
        final_features = [feature for feature, count in feature_counts.items() if count >= threshold]
        print(f"\nFinal selected features (appear in at least {threshold} folds): {final_features}")
    else:
        # Use the predefined selected features
        final_features = selected_features

    best_params_by_model = {}
    for model_name, param_scores in parameter_scores.items():
        avg_scores = {params: np.mean(scores) for params, scores in param_scores.items()}
        best_params = max(avg_scores, key=avg_scores.get)
        best_params_by_model[model_name] = (best_params, avg_scores[best_params])
        print(f"\nModel: {model_name}")
        print(f"  Best Parameters (average val score): {dict(best_params)}")
        print(f"  Best Average Val Score: {avg_scores[best_params]:.3f}")
        print("-" * 50)

    # Summarize results across all folds
    for model_name, result in overall_results.items():
        result['avg_train_score'] = np.mean(result['train_scores'])
        result['avg_val_score'] = np.mean(result['val_scores'])
        result['avg_training_time'] = np.mean(result['training_times'])

        print(f"\nModel: {model_name}")
        print(f"  Average Train F1: {result['avg_train_score']:.3f}")
        print(f"  Average Val F1: {result['avg_val_score']:.3f}")
        print(f"  Average Training Time: {result['avg_training_time']:.2f}s")
        print("-" * 50)


    return overall_results, best_params_by_model, final_features



def avg_score(model, features, X, y, scaler, balancing, n_splits):
    skf = StratifiedKFold(n_splits)
    macro_f1_train = []
    macro_f1_test = []
    timer = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        # Preprocessing
        try:
            X_train, X_val, y_train = preprocessing(X_train, X_val, y_train, scaler=scaler)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            continue

        X_train = X_train[features].apply(pd.to_numeric, errors='coerce')
        X_val = X_val[features].apply(pd.to_numeric, errors='coerce')
        y_train = y_train.apply(pd.to_numeric, errors='coerce')

        # Balancing
        if balancing:
            try:
                tomek = TomekLinks()
                X_train, y_train = tomek.fit_resample(X_train, y_train)
                adasyn = ADASYN(random_state=42)
                X_train, y_train = adasyn.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"Balancing failed: {e}, skipping fold.")
                continue

        if X_train.empty or y_train.empty or X_val.empty or y_val.empty:
            print("Empty data encountered, skipping fold...")
            continue

        try:
            # Training the model
            begin = time.perf_counter()
            model.fit(X_train, y_train)
            end = time.perf_counter()

            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Calculate macro F1-scores
            f1_train = f1_score(y_train, y_train_pred, average='macro')
            f1_test = f1_score(y_val, y_val_pred, average='macro')

            macro_f1_train.append(f1_train)
            macro_f1_test.append(f1_test)
            timer.append(end - begin)
        except Exception as e:
            print(f"Model training/scoring error: {e}")
            continue

    # Calculate averages and standard deviations
    avg_time = round(np.mean(timer), 3) if timer else np.nan
    avg_train = round(np.mean(macro_f1_train), 3) if macro_f1_train else np.nan
    avg_test = round(np.mean(macro_f1_test), 3) if macro_f1_test else np.nan
    std_time = round(np.std(timer), 2) if timer else np.nan
    std_train = round(np.std(macro_f1_train), 2) if macro_f1_train else np.nan
    std_test = round(np.std(macro_f1_test), 2) if macro_f1_test else np.nan

    return (
        f"{avg_time} +/- {std_time}",
        f"{avg_train} +/- {std_train}",
        f"{avg_test} +/- {std_test}",
    )



def show_results(df, X, y, features, scaler, balancing=True, *args, n_splits=5):
    """
    Receive an empty dataframe and the different models and call the function avg_score
    """
    count = 0
    # for each model passed as argument
    #scale = scale
    for arg in args:
        # obtain the results provided by avg_score
        time, avg_train, avg_test = avg_score(arg, features, X, y, scaler, balancing, n_splits)
        # store the results in the right row
        df.iloc[count] = time, avg_train, avg_test
        count+=1
    return df




### OPEN ENDED -----------------------------------------------------------------


# class_weights is a dictionary {0: 1.0, 1: 2.0, ...}, so class_label : class_weight
class WeightedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None, class_weights=None):
        self.base_model = base_model or LogisticRegression()
        self.class_weights = class_weights

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = self.base_model.predict_proba(X)
        if self.class_weights:
            adjusted_probas = probas.copy()
            for cls, weight in self.class_weights.items():
                adjusted_probas[:, cls] *= weight
            adjusted_probas /= adjusted_probas.sum(axis=1, keepdims=True)
            return adjusted_probas
        return probas

    def predict(self, X):
        adjusted_probas = self.predict_proba(X)
        return np.argmax(adjusted_probas, axis=1)

def random_search(probas, y_true, class_weights_list, n_iter=25, random_seed=42):
    np.random.seed(random_seed)
    n_classes = len(class_weights_list)
    sampled_weights = []

    # Random sampling of weights
    for _ in range(n_iter):
        weights = [np.random.choice(class_weights_list[class_idx]) for class_idx in range(n_classes)]
        sampled_weights.append(weights)

    f1_scores = []
    for weights in sampled_weights:
        adjusted_probas = probas * weights
        predictions = np.argmax(adjusted_probas, axis=1)
        f1 = f1_score(y_true, predictions, average="macro")
        f1_scores.append(f1)
    
    return sampled_weights, f1_scores

def cross_val_weights(X, y, class_weights_list, n_classes = 8, n_splits = 4, n_iter = 1000):

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)

    # DataFrame to store results
    results_df = pd.DataFrame()

    # Perform Cross Validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold_idx + 1}/{n_splits}...")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Preprocessing
        X_train, X_val, y_train = preprocessing(X_train, X_val, y_train, scaler)
        print(f"Preprocessing done for fold {fold_idx + 1}")
        
        # Feature selection
        X_train = X_train[features]
        X_val = X_val[features]
        X_train = X_train.apply(pd.to_numeric)
        X_val = X_val.apply(pd.to_numeric)

        # Train LogisticRegression model with multi_class='ovr'
        print("Training LogisticRegression model...")
        model = LogisticRegression(random_state=42, max_iter=100, multi_class='ovr')
        model.fit(X_train, y_train)

        # Get class probabilities for train and validation sets
        probas_train = model.predict_proba(X_train)
        probas_val = model.predict_proba(X_val)

        # Random Search on Train and Validation Probabilities
        print("Performing random search for weights...")
        train_weights, train_f1 = random_search(probas_train, y_train.values, class_weights_list, n_iter, random_seed=42)
        val_weights, val_f1 = random_search(probas_val, y_val.values, class_weights_list, n_iter, random_seed=42)

        # Combine results into DataFrame
        fold_results = pd.DataFrame({
            f"fold{fold_idx+1}_train_f1": train_f1,
            f"fold{fold_idx+1}_val_f1": val_f1
        }, index=[str(w) for w in train_weights])

        results_df = pd.concat([results_df, fold_results], axis=1)

    # Calculate Mean Scores Across Folds
    results_df['mean_train_f1'] = results_df[[col for col in results_df.columns if 'train_f1' in col]].mean(axis=1)
    results_df['mean_val_f1'] = results_df[[col for col in results_df.columns if 'val_f1' in col]].mean(axis=1)

    # Sort Results by Mean Validation F1 Score
    results_df = results_df.sort_values(by='mean_val_f1', ascending=False)

    return results_df


def holdout_random_search(X_train, X_val, y_train, y_val, models_and_params, n_iter_rs=50, metrics=None):
    """
    Perform holdout-based random search hyperparameter tuning for multiple models.

    Parameters:
        X_train (DataFrame): Training features.
        X_val (DataFrame): Validation features.
        y_train (Series): Training target labels.
        y_val (Series): Validation target labels.
        models_and_params (dict): Dictionary of models and their parameter distributions.
        n_iter_rs (int): Number of random combinations to test per model.
        metrics (dict): Dictionary of metrics to evaluate the models.

    Returns:
        results_df (DataFrame): DataFrame summarizing all model performances.
    """
    if metrics is None:
        metrics = {
            'macro_f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'accuracy': accuracy_score
        }
    
    # To store results for all models
    results = []

    # Iterate over models and their parameter grids
    for model_name, (model, param_grid) in models_and_params.items():
        print(f"Running RandomizedSearchCV for: {model_name}")
        
        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter_rs,
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        # Measure training time
        start_time = time.time()
        random_search.fit(X_train, y_train)
        end_time = time.time()

        # Get the best estimator and parameters
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        # Predict on validation set
        y_val_pred = best_model.predict(X_val)

        # Evaluate metrics
        val_scores = {metric_name: metric(y_val, y_val_pred) for metric_name, metric in metrics.items()}

        # Store the results
        results.append({
            'Model': model_name,
            'Best Parameters': best_params,
            'Random Search best_score_': random_search.best_score_,
            'Validation Macro F1': val_scores['macro_f1'],
            'Training Time (s)': round(end_time - start_time, 2)
        })

        print(f"Finished {model_name}: best_score_ (RandomSearch) = {random_search.best_score_:.4f}, "
              f"Final Val F1 = {val_scores['macro_f1']:.4f}")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results).sort_values(by='Validation Macro F1', ascending=False)
    
    # Display sorted results
    print("\nFinal Results:")
    display(results_df)
    
    return results_df

# FUNCTIONS FOR GUIDING THE DECISION OF WEIGHT INTERVALS

def train_binary_models(X_train, y_train_bin):
    models = {}
    for class_idx in range(y_train_bin.shape[1]):
        clf = LogisticRegression(max_iter=100, solver='lbfgs', random_state=42)
        clf.fit(X_train, y_train_bin[:, class_idx])
        models[class_idx] = clf
    return models

def generate_weighted_predictions(models, X, weights):
    n_classes = len(models)
    weighted_probas = np.zeros((X.shape[0], n_classes))  # Store weighted probabilities
    
    for class_idx, model in models.items():
        proba = model.predict_proba(X)[:, 1]  # Probabilities for the positive class
        weighted_probas[:, class_idx] = proba * weights[class_idx]  # Apply weights
    
    final_predictions = np.argmax(weighted_probas, axis=1)  # Predict class with max weighted probability
    return final_predictions, weighted_probas

def evaluate_model(X_train, X_val, y_train, y_val, models, weights):
    # Train predictions
    y_train_pred, _ = generate_weighted_predictions(models, X_train, weights)
    # Validation predictions
    y_val_pred, _ = generate_weighted_predictions(models, X_val, weights)

    # Generate classification reports
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)

    return train_report, val_report


def generate_class_weight_range(n_classes, weight_ranges):
    weights_dict = {}
    for class_idx in range(n_classes):
        min_weight, max_weight = weight_ranges[class_idx]
        weights_dict[class_idx] = np.linspace(min_weight, max_weight, 20)
    return weights_dict

def evaluate_f1_for_class_weight(X_train, X_val, y_train, y_val, y_train_bin, class_idx, weights_dict, base_weight=1.0):
    models = train_binary_models(X_train, y_train_bin)  # Train logistic regression models
    weight_values = weights_dict[class_idx]  # Get weight range for this class
    f1_scores = []

    for weight in weight_values:
        # Create weights for all classes, with the current class using `weight`
        weights = np.full(len(weights_dict), base_weight)
        weights[class_idx] = weight

        # Evaluate F1 score
        _, val_report = evaluate_model(X_train, X_val, y_train, y_val, models, weights)
        f1_scores.append(val_report["macro avg"]["f1-score"])

    return weight_values, f1_scores

def plot_f1_variation(X_train, X_val, y_train, y_val, y_train_bin, weight_ranges, base_weight=1.0):
    """
    Generate plots for F1 score variation across weights for each class.
    Args:
        X_train, X_val, y_train, y_val, y_train_bin: Training and validation data.
        weight_ranges: List of tuples specifying the (min, max) range for each class.
        base_weight: The default weight for all classes except the one being varied.
    """
    n_classes = len(weight_ranges)
    weights_dict = generate_class_weight_range(n_classes, weight_ranges)

    # Define the grid layout (4 rows x 2 columns)
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for class_idx in range(n_classes):
        weight_values, f1_scores = evaluate_f1_for_class_weight(
            X_train, X_val, y_train, y_val, y_train_bin, class_idx, weights_dict, base_weight
        )

        # Plot the F1 score variation for the current class
        ax = axes[class_idx]
        ax.plot(weight_values, f1_scores, marker="o", label=f"Class {class_idx}")
        ax.set_title(f"F1 Score Variation for Class {class_idx}")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Validation F1 Score")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
