import os
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from utils import *
from datetime import datetime
import gc 
from config import *
import pickle

class preprocess_POS_CASH_balance:
    '''
    Preprocess the POS_CASH_balance table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        3. data_preprocessing_and_feature_engineering method
        4. preprocess_pos_cash_aggregate method
        5. main method
    '''

    def __init__(self, file_directory = '', verbose = True, dump_to_pickle = False):
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
        Function to load the POS_CASH_balance.csv DataFrame.
        
        Inputs:
            self
            
        Returns:
            None
        '''
        
        if self.verbose:
            self.start = datetime.now()
            print('#########################################################')
            print('#          Pre-processing POS_CASH_balance.csv          #')
            print('#########################################################')
            print("\nLoading the DataFrame, POS_CASH_balance.csv, into memory...")

        self.pos_cash = pd.read_csv(self.file_directory + 'dseb63_POS_CASH_balance.csv')
        self.initial_size = self.pos_cash.shape

        if self.verbose:
            print("Loaded POS_CASH_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")
            
    def preprocess_pos_cash_cleaning_add_feature(self):
        '''
        Function to preprocess the table and create new features.
        
        Inputs:
            self
        
        Returns:
            None
        '''
        
        if self.verbose:
            start = datetime.now()
            print("\nStarting Data Cleaning and Feature Engineering...")

        # Making the MONTHS_BALANCE Positive
        self.pos_cash['MONTHS_BALANCE'] = np.abs(self.pos_cash['MONTHS_BALANCE'])

        # Sorting the DataFrame according to the month of status from oldest to latest, for rolling computations
        self.pos_cash = self.pos_cash.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=False)

        # Computing Exponential Moving Average (EMA) for some features based on MONTHS_BALANCE
        columns_for_ema = ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
        exp_columns = ['EXP_'+ele for ele in columns_for_ema]
        self.pos_cash[exp_columns] = self.pos_cash.groupby('SK_ID_PREV')[columns_for_ema].transform(lambda x: x.ewm(alpha=0.6).mean())

        # Creating new features based on Domain Knowledge
        self.pos_cash['SK_DPD_RATIO'] = self.pos_cash['SK_DPD'] / self.pos_cash['SK_DPD_DEF']
        self.pos_cash['TOTAL_TERM'] = self.pos_cash['CNT_INSTALMENT'] + self.pos_cash['CNT_INSTALMENT_FUTURE']

        # Flag months with late payment
        self.pos_cash['POS_IS_DPD_UNDER_120'] = self.pos_cash['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
        self.pos_cash['POS_IS_DPD_OVER_120'] = self.pos_cash['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)
        # Flag months with late payment        
        self.pos_cash['LATE_PAYMENT'] = self.pos_cash['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")

    def aggregate_pos_cash(self, categorical_cols):
        """
        Aggregate the pos_cash dataframe by SK_ID_CURR.
        Args:
            categorical_cols (list): List of categorical columns to aggregate.
        Returns:
            DataFrame: Aggregated pos_cash dataframe.
        """
        # Define the aggregation methods for categorical columns
        categorical_agg = {key: ['mean'] for key in categorical_cols}
        # Aggregate the pos_cash dataframe
        pos_agg = group(self.pos_cash, 'POS_', {**POS_CASH_AGG, **categorical_agg})
        return pos_agg

    def compute_loan_metrics(self):
        """
        Compute loan metrics for each customer.
        Returns:
        df_gp (pandas.DataFrame): DataFrame containing loan metrics grouped by SK_ID_CURR.
        """
        # Sort and group by SK_ID_PREV
        sort_pos = self.pos_cash.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
        gp = sort_pos.groupby('SK_ID_PREV')
        df = pd.DataFrame()
        df['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
        df['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()

        # Percentage of previous loans completed and completed before the initial term
        df['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
        df['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
        df['POS_COMPLETED_BEFORE_MEAN'] = df.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0
                                                    and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)

        # Number of remaining installments (future installments) and percentage from the total
        df['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
        df['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last() / gp['CNT_INSTALMENT'].last()

        # Group by SK_ID_CURR and merge
        df_gp = df.groupby('SK_ID_CURR').sum().reset_index()
        df_gp.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace=True)
        return df_gp

    def compute_late_payment_percentage(self):
        """
        Computes the percentage of late payments for the 3 most recent applications.
        Returns:
            DataFrame: A DataFrame with the SK_ID_CURR and LATE_PAYMENT_SUM columns.
        """
        # Percentage of late payments for the 3 most recent applications
        self.pos_cash = do_sum(self.pos_cash, ['SK_ID_PREV'], 'LATE_PAYMENT', 'LATE_PAYMENT_SUM')
        last_month_df = self.pos_cash.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()

        # Most recent applications (last 3)
        sort_pos = self.pos_cash.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
        gp = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)
        gp_mean = gp.groupby('SK_ID_CURR').mean().reset_index()
        return gp_mean[['SK_ID_CURR', 'LATE_PAYMENT_SUM']]

    def preprocess_pos_cash_aggregate(self):
        """
        Aggregates the POS_CASH_balance rows over SK_ID_PREV.
        Returns:
            Aggregated POS_CASH_balance table over SK_ID_PREV.
        """
        # Check if verbose mode is enabled
        if self.verbose:
            start = datetime.now()
            print("\nAggregations over SK_ID_PREV...")
        # Perform one-hot encoding on pos_cash table
        self.pos_cash, categorical_cols = one_hot_encoder(self.pos_cash, nan_as_category=False)
        
        # Aggregate pos_cash table using categorical_cols
        pos_cash_aggregated = self.aggregate_pos_cash(categorical_cols)
        
        # Compute loan metrics
        loan_metrics = self.compute_loan_metrics()
        
        # Compute late payment percentage
        late_payment_percentage = self.compute_late_payment_percentage()
        # Merge pos_cash_aggregated with loan_metrics and late_payment_percentage on SK_ID_CURR
        pos_cash_aggregated = pd.merge(pos_cash_aggregated, loan_metrics, on='SK_ID_CURR', how='left')
        pos_cash_aggregated = pd.merge(pos_cash_aggregated, late_payment_percentage, on='SK_ID_CURR', how='left')
        pos_cash_aggregated.drop(['POS_NAME_CONTRACT_STATUS_Canceled_MEAN', 'POS_NAME_CONTRACT_STATUS_Amortized debt_MEAN',
                                'POS_NAME_CONTRACT_STATUS_XNA_MEAN'], axis=1, inplace=True)
        
        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")
        return pos_cash_aggregated

    
    
    def main(self):
        '''
        Function to be called for complete preprocessing and aggregation of POS_CASH_balance table.
        
        Inputs:
            self
            
        Returns:
            Final pre=processed and aggregated POS_CASH_balance table.
        '''
        
        # Loading the dataframe
        self.load_dataframe()
        # Performing the data pre-processing and feature engineering
        self.preprocess_pos_cash_cleaning_add_feature()
        # Performing aggregations over SK_ID_PREV
        pos_cash_aggregated = self.preprocess_pos_cash_aggregate()

        if self.verbose:
            print('\nDone preprocessing POS_CASH_balance.')
            print(f"\nInitial Size of POS_CASH_balance: {self.initial_size}")
            print(f'Size of POS_CASH_balance after Pre-Processing, Feature Engineering and Aggregation: {pos_cash_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')
        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed POS_CASH_balance to POS_CASH_balance_preprocessed.pkl')
            with open(self.file_directory + 'POS_CASH_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(pos_cash_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-'*100)
        return pos_cash_aggregated