import config
import pickle
from config import *
from datetime import datetime
from utils import *
import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
from scipy.stats import kurtosis, iqr, skew
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pickle

class preprocess_installments_payments:
    '''
    Preprocess the installments_payments table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        3. preprocess_installments_payment_cleaning_add_feature method
        4. preprocess_installments_payment_aggregate method
        5. main method
    '''

    def __init__(self, file_directory='', verbose=True, dump_to_pickle=False):
        '''
        This function is used to initialize the class members 

        Inputs:
            self
            file_directory: Path, str, default = ''
                The path where the file exists. Include a '/' at the end of the path in input
            verbose: bool, default = True
                Whether to enable verbosity or not
            dump_to_pickle: bool, default = False
                Whether to pickle the final preprocessed table or not

        Returns:
            None
        '''

        self.file_directory = file_directory
        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle

    def load_dataframe(self):
        '''
        Function to load the installments_payments.csv DataFrame.

        Inputs:
            self

        Returns:
            None
        '''

        if self.verbose:
            self.start = datetime.now()
            print('##########################################################')
            print('#        Pre-processing installments_payments.csv        #')
            print('##########################################################')
            print("\nLoading the DataFrame, installments_payments.csv, into memory...")

        self.installments_payments = pd.read_csv(
            self.file_directory + 'dseb63_installments_payments.csv')
        self.initial_shape = self.installments_payments.shape

        if self.verbose:
            print("Loaded previous_application.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def preprocess_installments_payment_cleaning_add_feature(self):
        '''
        Function for pre-processing and feature engineering for installments_payments DataFrame.

        Inputs:
            self

        Returns:
            None
        '''
        if self.verbose:
            start = datetime.now()
            print("\nStarting Data Pre-processing and Feature Engineering...")

        # Aggregate payment information and create new features
        self.installments_payments = do_sum(self.installments_payments, [
                                            'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')

        # Creating new features related to payments
        self.installments_payments['PAYMENT_DIFFERENCE'] = self.installments_payments['AMT_INSTALMENT'] - \
            self.installments_payments['AMT_PAYMENT_GROUPED']
        self.installments_payments['PAYMENT_RATIO'] = self.installments_payments['AMT_INSTALMENT'] / \
            self.installments_payments['AMT_PAYMENT_GROUPED']
        self.installments_payments['PAID_OVER_AMOUNT'] = self.installments_payments['AMT_PAYMENT'] - \
            self.installments_payments['AMT_INSTALMENT']
        self.installments_payments['PAID_OVER'] = (
            self.installments_payments['PAID_OVER_AMOUNT'] > 0).astype(int)

        # Payment Entry Metrics: Days Past Due and Days Before Due
        self.installments_payments['DPD'] = self.installments_payments['DAYS_ENTRY_PAYMENT'] - \
            self.installments_payments['DAYS_INSTALMENT']
        self.installments_payments['DPD'] = self.installments_payments['DPD'].apply(
            lambda x: 0 if x <= 0 else x)
        self.installments_payments['DBD'] = self.installments_payments['DAYS_INSTALMENT'] - \
            self.installments_payments['DAYS_ENTRY_PAYMENT']
        self.installments_payments['DBD'] = self.installments_payments['DBD'].apply(
            lambda x: 0 if x <= 0 else x)
        self.installments_payments['LATE_PAYMENT'] = self.installments_payments['DBD'].apply(
            lambda x: 1 if x > 0 else 0)
        self.installments_payments['DPD_diff'] = self.installments_payments['DAYS_ENTRY_PAYMENT'] - \
            self.installments_payments['DAYS_INSTALMENT']
        self.installments_payments['DBD_diff'] = self.installments_payments['DAYS_INSTALMENT'] - \
            self.installments_payments['DAYS_ENTRY_PAYMENT']

        # Percentage Metrics: Payments, Late Payment Ratio, and Significant Late Payments
        self.installments_payments['INSTALMENT_PAYMENT_RATIO'] = self.installments_payments['AMT_PAYMENT'] / \
            self.installments_payments['AMT_INSTALMENT']
        self.installments_payments['LATE_PAYMENT_RATIO'] = self.installments_payments.apply(
            lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)
        self.installments_payments['SIGNIFICANT_LATE_PAYMENT'] = self.installments_payments['LATE_PAYMENT_RATIO'].apply(
            lambda x: 1 if x > 0.05 else 0)

        # Threshold Metrics: DPD Thresholds, Payment Percentage, and Payment Differences
        self.installments_payments['DPD_7'] = self.installments_payments['DPD'].apply(
            lambda x: 1 if x >= 7 else 0)
        self.installments_payments['DPD_15'] = self.installments_payments['DPD'].apply(
            lambda x: 1 if x >= 15 else 0)
        self.installments_payments['PAYMENT_PERC'] = self.installments_payments['AMT_PAYMENT'] / \
            self.installments_payments['AMT_INSTALMENT']
        self.installments_payments['PAYMENT_DIFF'] = self.installments_payments['AMT_INSTALMENT'] - \
            self.installments_payments['AMT_PAYMENT']

        # Days Past Due Categories
        self.installments_payments['INS_IS_DPD_UNDER_120'] = self.installments_payments['DPD'].apply(
            lambda x: 1 if (x > 0) & (x < 120) else 0)
        self.installments_payments['INS_IS_DPD_OVER_120'] = self.installments_payments['DPD'].apply(
            lambda x: 1 if (x >= 120) else 0)

        # Days Payment Ratio Metrics
        self.installments_payments['DAYS_PAYMENT_RATIO'] = self.installments_payments['DAYS_INSTALMENT'] / \
            self.installments_payments['DAYS_ENTRY_PAYMENT']
        self.installments_payments['DAYS_PAYMENT_DIFF'] = self.installments_payments['DAYS_INSTALMENT'] - \
            self.installments_payments['DAYS_ENTRY_PAYMENT']
        self.installments_payments['AMT_PAYMENT_DIFF'] = self.installments_payments['AMT_INSTALMENT'] - \
            self.installments_payments['AMT_PAYMENT']

        # sorting by SK_ID_PREV and NUM_INSTALMENT_NUMBER and Exponential Weighted Moving Average (EWMA) Metrics
        # The alpha parameter in the EWMA calculation determines the weight given to more recent observations
        self.installments_payments = self.installments_payments.sort_values(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], ascending=True)
        self.installments_payments['EXP_DAYS_PAYMENT_RATIO'] = self.installments_payments['DAYS_PAYMENT_RATIO'].transform(
            lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_DAYS_PAYMENT_DIFF'] = self.installments_payments['DAYS_PAYMENT_DIFF'].transform(
            lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_AMT_PAYMENT_RATIO'] = self.installments_payments['INSTALMENT_PAYMENT_RATIO'].transform(
            lambda x: x.ewm(alpha=0.5).mean())
        self.installments_payments['EXP_AMT_PAYMENT_DIFF'] = self.installments_payments['AMT_PAYMENT_DIFF'].transform(
            lambda x: x.ewm(alpha=0.5).mean())

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

    def aggregate_installments_general(self, cat_cols):
        """
        Aggregate installments data for general features.
        Args:
            cat_cols (list): List of categorical columns to aggregate.
        Returns:
            pandas.DataFrame: Aggregated installments data.
        """
        # Define aggregation dictionary
        INSTALLMENTS_GEN_AGG = {'NUM_INSTALMENT_VERSION': ['nunique']}
        # Add mean aggregation for each categorical column
        for cat in cat_cols:
            INSTALLMENTS_GEN_AGG[cat] = ['mean']
        # Aggregate installments data by SK_ID_CURR
        ins_agg = self.installments_payments.groupby('SK_ID_CURR').agg(INSTALLMENTS_GEN_AGG)
        # Rename columns with 'INSTAL_' prefix
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Calculate the number of installments for each SK_ID_CURR
        ins_agg['INSTAL_COUNT'] = self.installments_payments.groupby(
            'SK_ID_CURR').size()
        return ins_agg

    def aggregate_installments_year(self):
        """
        Aggregate and group the installments_payments data by SK_ID_CURR for the entries made within the last year.

        Returns:
            ins_d365_agg (DataFrame): The aggregated installments data grouped by SK_ID_CURR for the entries made within the last year.
        """
        # Filter the installments_payments data to include only entries made within the last year
        cond_day = self.installments_payments['DAYS_ENTRY_PAYMENT'] >= -365
        # Group the filtered data by SK_ID_CURR
        ins_d365_grp = self.installments_payments[cond_day].groupby(
            'SK_ID_CURR')
        # Aggregate the grouped data using the INSTALLMENTS_YEAR_AGG dictionary
        ins_d365_agg = ins_d365_grp.agg(INSTALLMENTS_YEAR_AGG)
        # Rename the columns of the aggregated data
        ins_d365_agg.columns = [
            'INS_D365' + ('_').join(column).upper() for column in ins_d365_agg.columns.ravel()]
        # Return the aggregated data
        return ins_d365_agg

    def aggregate_installments(self, pay_agg, INSTALLMENTS_AGG):
        """
        Aggregate the installments payments based on the defined columns.
        Args:
            pay_agg (DataFrame): The dataframe containing the installments payments.
            INSTALLMENTS_AGG (list): The list of columns to aggregate on.
        Returns:
            DataFrame: The aggregated dataframe.
        """
        # Group the installments payments based on the specified columns
        pay_agg = group(self.installments_payments, 'INS_', INSTALLMENTS_AGG)

        return pay_agg

    def aggregate_installments_time(self, pay_agg, INSTALLMENTS_TIME_AGG):
        """
        Aggregates installment payments based on different time periods.
        Args:
            pay_agg (pd.DataFrame): DataFrame holding the aggregated payment data.
            INSTALLMENTS_TIME_AGG (list): List of time periods (in months) to aggregate the payments.
        Returns:
            pd.DataFrame: DataFrame with the aggregated installment payment data.
        """
        for months in [12, 18, 36, 60]:
            # Get the unique SK_ID_PREV values for recent installment payments
            recent_prev_id = self.installments_payments[self.installments_payments[
                'DAYS_INSTALMENT'] >= -30*months]['SK_ID_PREV'].unique()
            # Filter installment payments for the recent SK_ID_PREV values
            pay_recent = self.installments_payments[self.installments_payments['SK_ID_PREV'].isin(
                recent_prev_id)]
            # Generate prefix for column names
            prefix = 'INS_{}M_'.format(months)
            # Group and merge the recent installment payments with the aggregated payment data
            pay_agg = group_and_merge(
                pay_recent, pay_agg, prefix, INSTALLMENTS_TIME_AGG)
        return pay_agg

    def aggregate_installments_last_loan(self, pay_agg, INSTALLMENTS_LAST_K_TREND_PERIODS):
        """
        Aggregate installment payment features for the last k trend periods.
        Parameters:
            pay_agg (DataFrame): DataFrame containing payment aggregation data.
            INSTALLMENTS_LAST_K_TREND_PERIODS (int): Number of periods to consider for trend calculation.
        Returns:
            pay_agg (DataFrame): Updated DataFrame with aggregated payment features.
            gp (DataFrame): Grouped DataFrame by 'SK_ID_CURR'.
        """
        # Define the group features
        group_features = ['SK_ID_CURR', 'SK_ID_PREV', 'DPD', 'LATE_PAYMENT',
                          'PAID_OVER_AMOUNT', 'PAID_OVER', 'DAYS_INSTALMENT']

        # Group the installment payments DataFrame by 'SK_ID_CURR'
        gp = self.installments_payments[group_features].groupby('SK_ID_CURR')

        # Define the function to apply to each group
        func = partial(trend_in_last_k_instalment_features,
                       periods=INSTALLMENTS_LAST_K_TREND_PERIODS)

        # Apply the function to each group in parallel
        g = parallel_apply(gp, func, index_name='SK_ID_CURR',
                           chunk_size=10000).reset_index()

        # Merge the results with pay_agg DataFrame
        pay_agg = pay_agg.merge(g, on='SK_ID_CURR', how='left')
        return pay_agg, gp

    def aggregate_last_loan_features(self, pay_agg, gp):
        """
        Aggregates the last loan features for each applicant.
        Args:
            pay_agg (DataFrame): DataFrame containing payment information.
            gp (DataFrameGroupBy): GroupBy object with the applicant ID as the grouping key.
        Returns:
            DataFrame: DataFrame with the aggregated last loan features merged with the payment information.
        """
        # Calculate the last loan features using parallel processing
        g = parallel_apply(gp, installments_last_loan_features,
                           index_name='SK_ID_CURR', chunk_size=10000).reset_index()
        # Merge the last loan features with the payment information
        pay_agg = pay_agg.merge(g, on='SK_ID_CURR', how='left')
        return pay_agg

    def preprocess_installments_payment_aggregate(self):
        '''
        Function for aggregating installments on previous loans over SK_ID_PREV

        Args:
            self: The object instance

        Returns:
            DataFrame: The installments_payments table aggregated over previous loans
        '''
        # Check if verbose flag is set
        if self.verbose:
            start = datetime.now()
            print("\nPerforming Aggregations over SK_ID_PREV...")

        # One-hot encode categorical columns and get the list of categorical columns
        self.installments_payments, cat_cols = one_hot_encoder(
            self.installments_payments, nan_as_category=False)

        # Aggregate the data over SK_ID_PREV, i.e. for each previous loan
        ins_agg = self.aggregate_installments_general(cat_cols)
        ins_d365_agg = self.aggregate_installments_year()
        pay_agg = self.aggregate_installments(ins_agg, INSTALLMENTS_AGG)
        pay_agg = self.aggregate_installments_time(
            pay_agg, INSTALLMENTS_TIME_AGG)
        pay_agg, gp = self.aggregate_installments_last_loan(
            pay_agg, INSTALLMENTS_LAST_K_TREND_PERIODS)
        pay_agg = self.aggregate_last_loan_features(pay_agg,  gp)
        pay_agg = pay_agg.merge(ins_agg, on='SK_ID_CURR', how='left')
        # Check if verbose flag is set
        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")
        return pay_agg

    def main(self):
        '''
        Function to be called for complete preprocessing and aggregation of installments_payments table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated installments_payments table.
        '''

        # loading the dataframe
        self.load_dataframe()
        # doing pre-processing and feature engineering
        self.preprocess_installments_payment_cleaning_add_feature()
        # First aggregating the data for each SK_ID_PREV
        installments_payments_aggregated = self.preprocess_installments_payment_aggregate()

        if self.verbose:
            print('\nDone preprocessing installments_payments.')
            print(
                f"\nInitial Size of installments_payments: {self.initial_shape}")
            print(
                f'Size of installments_payments after Pre-Processing, Feature Engineering and Aggregation: {installments_payments_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print(
                    '\nPickling pre-processed installments_payments to installments_payments_preprocessed.pkl')
            with open(self.file_directory + 'installments_payments_preprocessed.pkl', 'wb') as f:
                pickle.dump(installments_payments_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-'*100)

        return installments_payments_aggregated


def trend_in_last_k_instalment_features(gr, periods):
    """
    Calculate trend features based on the last k installments.
    Parameters:
    - gr: DataFrame, the input DataFrame
    - periods: list of integers, the periods to consider
    Returns:
    - features: dict, the calculated trend features
    """
    # Create a copy of the input DataFrame
    gr_ = gr.copy()
    # Sort the DataFrame by 'DAYS_INSTALMENT' column in descending order
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    # Initialize an empty dictionary to store the trend features
    features = {}
    # Iterate over each period
    for period in periods:
        # Select the first 'period' rows from the DataFrame
        gr_period = gr_.iloc[:period]
        # Calculate and add the trend feature based on 'DPD' column
        features = add_trend_feature(features, gr_period, 'DPD',
                                     '{}_TREND_'.format(period))
        # Calculate and add the trend feature based on 'PAID_OVER_AMOUNT' column
        features = add_trend_feature(features, gr_period, 'PAID_OVER_AMOUNT',
                                     '{}_TREND_'.format(period))
    # Return the calculated trend features
    return features


def installments_last_loan_features(gr):
    """
    Calculates features for the last loan installment.
    Parameters:
    - gr: DataFrame, the group of installment data
    Returns:
    - features: dict, the calculated features
    """
    # Make a copy of the group dataframe
    gr_ = gr.copy()
    # Sort the dataframe by 'DAYS_INSTALMENT' column in descending order
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    # Get the ID of the last installment
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    # Filter the dataframe to keep only the rows with the last installment ID
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]
    # Initialize an empty dictionary to store the calculated features
    features = {}
    # Calculate features related to 'DPD' column
    features = add_features_in_group(features, gr_, 'DPD',
                                     ['sum', 'mean', 'max', 'std'],
                                     'LAST_LOAN_')
    # Calculate features related to 'LATE_PAYMENT' column
    features = add_features_in_group(features, gr_, 'LATE_PAYMENT',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    # Calculate features related to 'PAID_OVER_AMOUNT' column
    features = add_features_in_group(features, gr_, 'PAID_OVER_AMOUNT',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'LAST_LOAN_')
    # Calculate features related to 'PAID_OVER' column
    features = add_features_in_group(features, gr_, 'PAID_OVER',
                                     ['count', 'mean'],
                                     'LAST_LOAN_')
    return features
