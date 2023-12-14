import gc
import pickle
from config import *
from datetime import datetime
from utils import *
import os
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


class preprocess_previous_application:
    '''
    Preprocess the previous_application table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        3. preprocess_previous_application_cleaning_add_feature method
        4. preprocess_previous_application_aggregate_main method
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
        Function to load the previous_application.csv DataFrame.

        Inputs:
            self

        Returns:
            None
        '''

        if self.verbose:
            self.start = datetime.now()
            print('########################################################')
            print('#        Pre-processing previous_application.csv        #')
            print('########################################################')
            print("\nLoading the DataFrame, previous_application.csv, into memory...")

        # loading the DataFrame into memory
        self.previous_application = pd.read_csv(
            self.file_directory + 'dseb63_previous_application.csv')
        self.installments_payments = pd.read_csv(
            self.file_directory + 'dseb63_installments_payments.csv')
        self.initial_shape = self.previous_application.shape
        if self.verbose:
            print("Loaded previous_application.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def preprocess_previous_application_cleaning_add_feature(self):
        '''
        Function to clean the data. Removes erroneous points, fills categorical NaNs with 'XNA'.

        Inputs:
            self

        Returns:
            None
        '''

        if self.verbose:
            start = datetime.now()
            print('\nStarting Data Cleaning...')

        # Replace 365243 with NaN in date columns
        columns_to_replace_nan = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                                  'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
        self.previous_application[columns_to_replace_nan].replace(
            365243, np.nan, inplace=True)

        # Calculate application and credit difference and ratio
        self.previous_application['APPLICATION_CREDIT_DIFF'] = self.previous_application['AMT_APPLICATION'] - \
            self.previous_application['AMT_CREDIT']
        self.previous_application['APPLICATION_CREDIT_RATIO'] = self.previous_application['AMT_APPLICATION'] / \
            self.previous_application['AMT_CREDIT']

        # Create binary feature based on application credit ratio
        self.previous_application["NEW_APP_CREDIT_RATE_RATIO"] = self.previous_application["APPLICATION_CREDIT_RATIO"].apply(
            lambda x: 1 if (x <= 1) else 0)

        # Calculate credit to goods price ratio
        self.previous_application['NEW_CREDIT_GOODS_RATE'] = self.previous_application['AMT_CREDIT'] / \
            self.previous_application['AMT_GOODS_PRICE']

        # Categorize number of payments into short, middle, and long
        self.previous_application["NEW_CNT_PAYMENT"] = pd.cut(x=self.previous_application['CNT_PAYMENT'], bins=[
                                                              0, 12, 60, 120], labels=["Short", "Middle", "Long"])

        # Convert categorical features to object type
        self.previous_application['NFLAG_LAST_APPL_IN_DAY'] = self.previous_application['NFLAG_LAST_APPL_IN_DAY'].astype(
            "O")
        self.previous_application['FLAG_LAST_APPL_PER_CONTRACT'] = self.previous_application['FLAG_LAST_APPL_PER_CONTRACT'].astype(
            "O")
        self.previous_application["NEW_CNT_PAYMENT"] = self.previous_application['NEW_CNT_PAYMENT'].astype(
            "O")
        self.previous_application['NEW_APP_CREDIT_RATE_RATIO'] = self.previous_application['NEW_APP_CREDIT_RATE_RATIO'].astype(
            'O')
        self.previous_application['NEW_APP_CREDIT_RATE_RATIO'] = self.previous_application['NEW_APP_CREDIT_RATE_RATIO'].replace({
                                                                                                                                "0": "Yes", "1": "No"})

        # Calculate return day, termination day difference, due day difference, and end difference
        self.previous_application['NEW_RETURN_DAY'] = self.previous_application['DAYS_DECISION'] + \
            self.previous_application['CNT_PAYMENT'] * 30
        self.previous_application['NEW_DAYS_TERMINATION_DIFF'] = self.previous_application['DAYS_TERMINATION'] - \
            self.previous_application['NEW_RETURN_DAY']
        self.previous_application['NEW_DAYS_DUE_DIFF'] = self.previous_application[
            'DAYS_LAST_DUE_1ST_VERSION'] - self.previous_application['DAYS_FIRST_DUE']
        self.previous_application["NEW_END_DIFF"] = self.previous_application["DAYS_TERMINATION"] - \
            self.previous_application["DAYS_LAST_DUE"]

        # Calculate ratios: credit to annuity, down payment to credit
        self.previous_application['CREDIT_TO_ANNUITY_RATIO'] = self.previous_application['AMT_CREDIT'] / \
            self.previous_application['AMT_ANNUITY']
        self.previous_application['DOWN_PAYMENT_TO_CREDIT'] = self.previous_application['AMT_DOWN_PAYMENT'] / \
            self.previous_application['AMT_CREDIT']
        # Calculate simplified interests, amount of interest, interest share, and interest rate
        total_payment = self.previous_application['AMT_ANNUITY'] * \
            self.previous_application['CNT_PAYMENT']
        self.previous_application['SIMPLE_INTERESTS'] = (
            total_payment/self.previous_application['AMT_CREDIT'] - 1)/self.previous_application['CNT_PAYMENT']
        self.previous_application['AMT_INTEREST'] = self.previous_application['CNT_PAYMENT'] * self.previous_application[
            'AMT_ANNUITY'] - self.previous_application['AMT_CREDIT']
        self.previous_application['INTEREST_SHARE'] = self.previous_application['AMT_INTEREST'] / \
            self.previous_application['AMT_CREDIT']
        self.previous_application['INTEREST_RATE'] = 2 * 12 * self.previous_application['AMT_INTEREST'] / (self.previous_application[
            'AMT_CREDIT'] * (self.previous_application['CNT_PAYMENT'] + 1))
        self.previous_application['DAYS_LAST_DUE_DIFF'] = self.previous_application[
            'DAYS_LAST_DUE_1ST_VERSION'] - self.previous_application['DAYS_LAST_DUE']

        if self.verbose:
            print("Done.")
            print(f"Time taken = {datetime.now() - start}")

    def aggregate_payment_info(self, active_df):
        """
        Aggregate payment information for active loans.
        Args:
            active_df (pandas.DataFrame): DataFrame containing active loan information.
        Returns:
            pandas.DataFrame: DataFrame with aggregated payment information for active applications.
        """
        # Find how much was already paid in active loans (using installments csv)
        active_pay = self.installments_payments[self.installments_payments['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]
        active_pay_agg = active_pay.groupby(
            'SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
        active_pay_agg.reset_index(inplace=True)

        # Active loans: difference of what was paid and installments
        active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']

        # Merge with active_df
        active_df = active_df.merge(
            active_pay_agg, on='SK_ID_PREV', how='left')
        active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']
        active_df['REPAYMENT_RATIO'] = active_df['AMT_PAYMENT'] / active_df['AMT_CREDIT']

        # Perform aggregations for active applications
        active_agg_df = group(active_df, 'PREV_ACTIVE_', PREVIOUS_ACTIVE_AGG)
        active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['PREV_ACTIVE_AMT_PAYMENT_SUM'] / \
            active_agg_df['PREV_ACTIVE_AMT_CREDIT_SUM']

        del active_pay, active_pay_agg, active_df
        gc.collect()

        return active_agg_df

    def process_approved_loans(self):
        """
        Process approved and active loans.
        Returns:
            Tuple: active_agg_df (DataFrame): aggregated payment information for active loans,
                approved (DataFrame): approved loans with the difference in days between versions of the last due date
        """
        # Filter and process approved and active loans
        # Filters applications with status "Approved" and DAYS_LAST_DUE equal to 365243 (potentially active loans)
        approved = self.previous_application[self.previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
        active_df = approved[approved['DAYS_LAST_DUE'] == 365243]
        # Aggregate payment information for active loans
        active_agg_df = self.aggregate_payment_info(active_df)
        # Calculate the difference in days between versions of the last due date for approved loans
        approved['DAYS_LAST_DUE_DIFF'] = approved['DAYS_LAST_DUE_1ST_VERSION'] - approved['DAYS_LAST_DUE']
        return active_agg_df, approved

    def perform_general_aggregations(self, categorical_cols, active_agg_df):
        """
        Perform general aggregations on previous applications data and merge it with active loans data.
        Args:
            categorical_cols (list): List of categorical columns to aggregate.
            active_agg_df (DataFrame): DataFrame containing aggregated active loans data.
        Returns:
            DataFrame: Aggregated previous applications data merged with active loans data.
        """
        categorical_agg = {key: ['mean'] for key in categorical_cols}
        # Perform general aggregations
        agg_prev = group(self.previous_application, 'PREV_',{**PREVIOUS_AGG, **categorical_agg})
        # Merge active loans dataframe on agg_prev
        agg_prev = agg_prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
        del active_agg_df
        gc.collect()
        return agg_prev

    def perform_approved_refused_aggregations(self, agg_prev):
        """
        Perform aggregations for approved and refused loans.
        Args:
            agg_prev (pandas.DataFrame): The previous aggregation data.
        Returns:
            pandas.DataFrame: The updated aggregation data.
        """
        # Filter rows for approved loans
        approved = self.previous_application[self.previous_application['NAME_CONTRACT_STATUS_Approved'] == 1]
        # Perform aggregation for approved loans
        agg_prev = group_and_merge(approved, agg_prev, 'APPROVED_', PREVIOUS_APPROVED_AGG)
        # Filter rows for refused loans
        refused = self.previous_application[self.previous_application['NAME_CONTRACT_STATUS_Refused'] == 1]
        # Perform aggregation for refused loans
        agg_prev = group_and_merge(refused, agg_prev, 'REFUSED_', PREVIOUS_REFUSED_AGG)
        del approved, refused
        gc.collect()
        return agg_prev

    def perform_loan_type_aggregations(self, agg_prev):
        """
        Perform loan type aggregations on previous application data.
        Args:
            agg_prev (pd.DataFrame): DataFrame containing previous application data.
        Returns:
            pd.DataFrame: DataFrame with loan type aggregations.
        """
        # Aggregations for Consumer loans and Cash loans
        for loan_type in ['Consumer loans', 'Cash loans']:
            type_df = self.previous_application[self.previous_application['NAME_CONTRACT_TYPE_{}'.format(
                loan_type)] == 1]
            prefix = 'PREV_' + loan_type.split(" ")[0] + '_'
            agg_prev = group_and_merge(type_df, agg_prev, prefix, PREVIOUS_LOAN_TYPE_AGG)
            del type_df
            gc.collect()
        return agg_prev

    def perform_late_payments_aggregations(self, agg_prev):
        """
        Perform aggregations for loans with late payments.
        
        Args:
            agg_prev (pd.DataFrame): Previous application data with previous aggregations.
        Returns:
            pd.DataFrame: Updated previous application data with new aggregations.
        """
        # Get the SK_ID_PREV for loans with late payments (days past due)
        self.installments_payments['LATE_PAYMENT'] = self.installments_payments['DAYS_ENTRY_PAYMENT'] - \
            self.installments_payments['DAYS_INSTALMENT']
        self.installments_payments['LATE_PAYMENT'] = self.installments_payments['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
        dpd_id = self.installments_payments[self.installments_payments['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()

        # Aggregations for loans with late payments
        agg_dpd = group_and_merge(self.previous_application[self.previous_application['SK_ID_PREV'].isin(dpd_id)], agg_prev,
                                  'PREV_LATE_', PREVIOUS_LATE_PAYMENTS_AGG)
        del agg_dpd, dpd_id
        gc.collect()
        return agg_prev

    def perform_last_months_aggregations(self, agg_prev):
        """
        Perform aggregations for loans in the last x months.
        Args:
            agg_prev (pandas.DataFrame): DataFrame containing previous aggregations.
        Returns:
            pandas.DataFrame: DataFrame with updated aggregations.
        """
        # Aggregations for loans in the last x months
        for time_frame in [6, 12, 24]:
            time_frame_df = self.previous_application[self.previous_application['DAYS_DECISION']>= -30 * time_frame]
            prefix = 'PREV_LAST{}M_'.format(time_frame)
            agg_prev = group_and_merge(time_frame_df, agg_prev, prefix, PREVIOUS_TIME_AGG)
            del time_frame_df
            gc.collect()
        del self.previous_application
        gc.collect()
        return agg_prev

    def preprocess_previous_application_aggregate_main(self):
        '''
        Function to do preprocessing such as categorical encoding and feature engineering.

        Inputs: 
            self

        Returns:
            agg_prev: The preprocessed and aggregated dataframe
        '''

        if self.verbose:
            start = datetime.now()
            print("\nPerforming Preprocessing and Feature Engineering...")
        # One-hot encode several categorical columns in the 'previous_application' dataframe
        ohe_columns = ['NAME_CONTRACT_STATUS', 'NAME_CONTRACT_TYPE', 'CHANNEL_TYPE',
                    'NAME_TYPE_SUITE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',
                    'NAME_PRODUCT_TYPE', 'NAME_CLIENT_TYPE']
        self.previous_application, categorical_cols = one_hot_encoder(self.previous_application, ohe_columns, nan_as_category=False)
        # Process approved loans
        active_agg_df, approved = self.process_approved_loans()
        # Perform general aggregations
        agg_prev = self.perform_general_aggregations(categorical_cols, active_agg_df)
        # Perform aggregations for approved and refused loans
        agg_prev = self.perform_approved_refused_aggregations(agg_prev)
        # Perform aggregations based on loan type
        agg_prev = self.perform_loan_type_aggregations(agg_prev)
        # Perform aggregations for late payments
        agg_prev = self.perform_late_payments_aggregations(agg_prev)
        # Perform aggregations for last months
        agg_prev = self.perform_last_months_aggregations(agg_prev)
        # Clean up memory
        del active_agg_df, approved
        gc.collect()
        # Check if verbose mode is enabled
        if self.verbose:
            print("Done.")
            print(f"Time taken = {datetime.now() - start}")
        return agg_prev

    def main(self):
        '''
        Function to be called for complete preprocessing and aggregation of previous_application table.

        Inputs:
            self

        Returns:
            Final pre=processed and aggregated previous_application table.
        '''

        # loading the DataFrame
        self.load_dataframe()

        # cleaning and add features to the data
        self.preprocess_previous_application_cleaning_add_feature()

        # aggregating data over SK_ID_CURR
        previous_application_aggregated = self.preprocess_previous_application_aggregate_main()

        if self.verbose:
            print('Done aggregations.')
            print(f"\nInitial Size of previous_application: {self.initial_shape}")
            print(f'Size of previous_application after Pre-Processing, Feature Engineering and Aggregation: {previous_application_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed previous_application to previous_application_preprocessed.pkl')
            with open(self.file_directory + 'previous_application_preprocessed.pkl', 'wb') as f:
                pickle.dump(previous_application_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-'*100)
        return previous_application_aggregated
