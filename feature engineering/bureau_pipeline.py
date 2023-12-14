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

class preprocess_bureau_balance_and_bureau:
    '''
    Preprocess the tables bureau_balance and bureau.
    Contains 5 main functions:
        1. init method
        2. preprocess_bureau_balance method
        3. preprocess_bureau_add_features method
        4. preprocess_bureau_aggregate_main method
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
        self.start = datetime.now()
        
    def preprocess_bureau_balance(self):
        '''
        Function to preprocess bureau_balance table.
        This function first loads the table into memory, does one hot encoding, and finally
        aggregates the data over SK_ID_BUREAU and merge
        
        Inputs:
            self
            
        Returns:
            preprocessed and aggregated bureau_balance table.
        '''
        
        if self.verbose:
            print('#######################################################')
            print('#          Pre-processing bureau_balance.csv          #')
            print('#######################################################')
            print("\nLoading the DataFrame, bureau_balance.csv, into memory...")

        bureau_balance = pd.read_csv(self.file_directory + 'dseb63_bureau_balance.csv')

        if self.verbose:
            print("Loaded bureau_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")
            print("\nStarting Data Cleaning and Feature Engineering...")

        bureau_balance, categorical_columns = one_hot_encoder(bureau_balance, nan_as_category= False)
        # Calculate rate for each category with decay
        aggregated_bureau_balance = bureau_balance.groupby('SK_ID_BUREAU')[categorical_columns].mean().reset_index()
        # Define aggregations for 'MONTHS_BALANCE' column
        aggrerations_mb = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size', 'var']}
        aggregated_bureau_balance = group_and_merge(bureau_balance, aggregated_bureau_balance, '', aggrerations_mb, 'SK_ID_BUREAU')
        

        if self.verbose:
            print('Done preprocessing bureau_balance.')
            print(f"\nInitial Size of bureau_balance: {bureau_balance.shape}")
            print(f'Size of bureau_balance after Pre-Processing, Feature Engineering and Aggregation: {aggregated_bureau_balance.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')
        del bureau_balance
        gc.collect()
        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed bureau_balance to bureau_balance_preprocessed.pkl')
            with open(self.file_directory + 'bureau_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(aggregated_bureau_balance, f)
            if self.verbose:
                print('Done.')     
        
        return aggregated_bureau_balance
    
    def preprocess_bureau_add_feature(self):
        '''
        Function to preprocess the bureau table and merge it with the aggregated bureau_balance table.
        Finally aggregates the data over SK_ID_CURR for it to be merged with application_train table.
        
        Inputs:
            self
            aggregated_bureau_balance: DataFrame of aggregated bureau_balance table
        
        Returns:
            Final preprocessed, merged and aggregated bureau table
        '''
        
        if self.verbose:
            start2 = datetime.now()
            print('\n##############################################')
            print('#          Pre-processing bureau.csv         #')
            print('##############################################')
            print("\nLoading the DataFrame, bureau.csv, into memory...")
            

        bureau = pd.read_csv(self.file_directory + 'dseb63_bureau.csv')

        if self.verbose:
            print("Loaded bureau.csv")
            print(f"\nInitial Size of bureau: {bureau.shape}")
            print(f"Time Taken to load = {datetime.now() - start2}")
            print("\nStarting Data Cleaning and Feature Engineering...")

        # SECTION 1: HANDLING MISSING VALUES Replaces values with NaN if they exceed a certain threshold (-50 years).
        bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] > -50*365] = np.nan
        bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] > -50*365] = np.nan
        bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] > -50*365] = np.nan

        # SECTION 2: ADDING MORE FEATURES BASED ON DOMAIN KNOWLEDGE
        # 2.1 Calculating Loan Duration and Related Features:
        bureau['CREDIT_DURATION'] = np.abs(bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE'])
        bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
        bureau['CREDIT_ENDDATE_UPDATE_DIFF'] = np.abs(bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE'])

        # 2.2 Credit-to-Debt Analysis
        bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
        bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
        bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
        
        # 2.3 Overdue Ratio Features
        # FLAG_OVERDUE_RECENT: Binary flag indicating recent overdue credit (1 if overdue, 0 otherwise). 
        bureau['FLAG_OVERDUE_RECENT'] = [0 if ele == 0 else 1 for ele in bureau['CREDIT_DAY_OVERDUE']]
        bureau['MAX_AMT_OVERDUE_DURATION_RATIO'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['CREDIT_DURATION']
        bureau['CURRENT_AMT_OVERDUE_DURATION_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['CREDIT_DURATION']
        bureau['AMT_OVERDUE_DURATION_LEFT_RATIO'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['DAYS_CREDIT_ENDDATE']

        # 2.4 Loan Usage and Prolongation
        bureau['CNT_PROLONGED_MAX_OVERDUE_MUL'] = bureau['CNT_CREDIT_PROLONG'] * bureau['AMT_CREDIT_MAX_OVERDUE']
        bureau['CNT_PROLONGED_DURATION_RATIO'] = bureau['CNT_CREDIT_PROLONG'] / bureau['CREDIT_DURATION']
        
        if self.verbose:
            print('Done adding more features to bureau_balance.')
            print(f'Size of bureau after adding new domain knowledge features: {bureau.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')
        return bureau

    def preprocess_bureau_aggregate_general(self, bureau): 
        """
        Preprocesses the given `bureau` data by aggregating it based on the column 'BUREAU_' using the function `BUREAU_AGG`.
        Args:
            bureau (DataFrame): The input data containing the bureau information.
        Returns:
            DataFrame: The aggregated bureau data based on the column 'BUREAU_'.
        """
        aggregated_bureau = group(bureau, 'BUREAU_', BUREAU_AGG) 
        return aggregated_bureau

    def preprocess_bureau_aggregate_duration(self, bureau): 
        """
        Preprocesses the bureau dataframe by calculating the mean for each feature based on the number of months in balance.
        Args:
            bureau (pandas.DataFrame): The bureau dataframe.
        Returns:
            pandas.DataFrame: The preprocessed dataframe with aggregated duration.
        """
        # Define the loan length features
        loan_length_features = [
            'AMT_CREDIT_MAX_OVERDUE', 
            'AMT_CREDIT_SUM_OVERDUE', 
            'AMT_CREDIT_SUM',
            'AMT_CREDIT_SUM_DEBT', 
            'DEBT_PERCENTAGE', 
            'DEBT_CREDIT_DIFF', 
            'STATUS_0', 
            'STATUS_SUM'
        ]
        # Group by months balance size and calculate the mean for each loan length feature
        aggregated_duration = bureau.groupby('MONTHS_BALANCE_SIZE')[loan_length_features].mean().reset_index()
        # Rename the loan length features with a prefix 'LL_'
        aggregated_duration.rename({feat: 'LL_' + feat for feat in loan_length_features}, axis=1, inplace=True)
        return aggregated_duration

    def preprocess_bureau_aggregate_active_loan(self, bureau, aggregated_bureau): 
        """
        Preprocesses the bureau data by aggregating the active loans.
        
        Args:
            bureau (DataFrame): The bureau data.
            aggregated_bureau (DataFrame): The aggregated bureau data.
            
        Returns:
            DataFrame: The aggregated bureau data after including the active loans.
        """
        # Select only the active loans
        active_loan = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        # Group and merge the active loans with the aggregated bureau data
        aggregated_bureau = group_and_merge(active_loan, aggregated_bureau, 'BUREAU_ACTIVE_', BUREAU_ACTIVE_AGG)
        # Clean up the memory
        del active_loan
        gc.collect()
        return aggregated_bureau

    def preprocess_bureau_aggregate_closed_loan(self, bureau, aggregated_bureau):
        """
        Preprocesses the bureau data by aggregating loan types.
        Args:
            bureau (DataFrame): The bureau data.
            aggregated_bureau (DataFrame): The aggregated bureau data.
        Returns:
            DataFrame: The preprocessed aggregated bureau data.
        """
        # Iterate over the list of credit types
        for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
            # Filter the bureau data based on the credit type
            type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
            # Generate the prefix for the aggregated bureau data based on the credit type
            prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
            # Group and merge the filtered bureau data with the aggregated bureau data
            aggregated_bureau = group_and_merge(type_df, aggregated_bureau, prefix, BUREAU_LOAN_TYPES_AGG)
            # Clean up the filtered bureau data
            del type_df
            gc.collect()
        # Return the preprocessed aggregated bureau data
        return aggregated_bureau

    def preprocess_bureau_aggregate_loan_types(self, bureau, aggregated_bureau):
        """
        Preprocesses the bureau dataframe by aggregating loan types.
        Args:
            bureau (DataFrame): The bureau dataframe.
            aggregated_bureau (DataFrame): The aggregated bureau dataframe.
        Returns:
            DataFrame: The aggregated bureau dataframe.
        """
        # Iterate over the list of credit types
        for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan', 'Microloan']:
            # Filter the bureau dataframe based on the current credit type
            type_df = bureau[bureau['CREDIT_TYPE_' + credit_type] == 1]
            # Generate the prefix for the aggregated columns
            prefix = 'BUREAU_' + credit_type.split(' ')[0].upper() + '_'
            # Group and merge the filtered dataframe with the aggregated dataframe
            aggregated_bureau = group_and_merge(type_df, aggregated_bureau, prefix, BUREAU_LOAN_TYPES_AGG)
            # Clean up the filtered dataframe
            del type_df
            gc.collect()
        return aggregated_bureau

    def preprocess_bureau_aggregate_time(self, bureau, aggregated_bureau):
        """
        Preprocesses the bureau data by aggregating it based on different time frames.
        Args:
            bureau (DataFrame): The bureau data.
            aggregated_bureau (DataFrame): The aggregated bureau data.
        Returns:
            DataFrame: The preprocessed aggregated bureau data.
        """
        # Iterate over the time frames
        for time_frame in [12, 24]:
            # Filter the bureau data based on the time frame
            time_frame_df = bureau[bureau['DAYS_CREDIT'] >= -30*time_frame]
            # Generate the prefix for the aggregated columns
            prefix = 'BUREAU_LAST{}M_'.format(time_frame)
            # Group and merge the time frame data with the aggregated data
            aggregated_bureau = group_and_merge(time_frame_df, aggregated_bureau, prefix, BUREAU_TIME_AGG)
            # Clean up the temporary time frame data
            del time_frame_df
            gc.collect()
        return aggregated_bureau

    def preprocess_bureau_aggregate_main(self, bureau, aggregated_bureau_balance):
        """
        Preprocesses the bureau.csv file and performs feature engineering and aggregation.
        Args:
            bureau: DataFrame containing the bureau.csv data.
            aggregated_bureau_balance: DataFrame containing the aggregated bureau_balance data.
        Returns:
            aggregated_bureau: DataFrame containing the preprocessed and aggregated bureau data.
        """
        if self.verbose:
            start3 = datetime.now()
            print('\n##############################################')
            print('#          Pre-processing bureau.csv         #')
            print('##############################################')
            print("\nLoading the DataFrame, bureau.csv, into memory...")
        # Perform one-hot encoding on categorical columns
        bureau, categorical_cols = one_hot_encoder(bureau, nan_as_category=False)
        # Merge bureau balance features
        bureau = bureau.merge(aggregated_bureau_balance, how='left', on='SK_ID_BUREAU')
        # Flag months with late payments (days past due)
        bureau['STATUS_SUM'] = bureau[[f'STATUS_{i}' for i in range(1, 6)]].sum(axis=1)
        # Perform duration-based aggregation
        aggregated_duration = self.preprocess_bureau_aggregate_duration(bureau)
        bureau = bureau.merge(aggregated_duration, how='left', on='MONTHS_BALANCE_SIZE')
        del aggregated_duration
        # Perform general loans aggregations
        aggregated_bureau = self.preprocess_bureau_aggregate_general(bureau)
        # Perform active and closed loans aggregations
        aggregated_bureau = self.preprocess_bureau_aggregate_active_loan(bureau, aggregated_bureau)
        aggregated_bureau = self.preprocess_bureau_aggregate_closed_loan(bureau, aggregated_bureau)
        # Perform aggregations for the main loan types
        # Perform time-based aggregations: last x months
        aggregated_bureau = self.preprocess_bureau_aggregate_time(bureau, aggregated_bureau)
        # Perform last loan max overdue aggregation
        sort_bureau = bureau.sort_values(by=['DAYS_CREDIT'])
        gr = sort_bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].last().reset_index().rename({'AMT_CREDIT_MAX_OVERDUE': 'BUREAU_LAST_LOAN_MAX_OVERDUE'})
        aggregated_bureau = aggregated_bureau.merge(gr, on='SK_ID_CURR', how='left')
        # Perform additional feature engineering
        # Calculate ratios: total debt/total credit and active loans debt/active loans credit
        aggregated_bureau['BUREAU_DEBT_OVER_CREDIT'] = aggregated_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / aggregated_bureau['BUREAU_AMT_CREDIT_SUM_SUM']
        aggregated_bureau['BUREAU_ACTIVE_DEBT_OVER_CREDIT'] = aggregated_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'] / aggregated_bureau['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM']
        # Calculate overdue over debt ratio: The fraction of total Debt that is overdue per customer, A high value could indicate a potential DEFAULT
        aggregated_bureau['OVERDUE_DEBT_RATIO'] = aggregated_bureau['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] / aggregated_bureau['BUREAU_AMT_CREDIT_SUM_DEBT_SUM']
        if self.verbose:
            print('Done preprocessing bureau and bureau_balance.')
            print(f'Size of bureau and bureau_balance after Merging, Pre-Processing, Feature Engineering and Aggregation: {aggregated_bureau.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed bureau and bureau_balance to bureau_merged_preprocessed.pkl')
            with open(self.file_directory + 'bureau_merged_preprocessed.pkl', 'wb') as f:
                pickle.dump(aggregated_bureau, f)
            if self.verbose:
                print('Done.')  
        if self.verbose:
            print('-'*100)

        return aggregated_bureau
        
    def main(self):
        '''
        Function to be called for complete preprocessing and aggregation of the bureau and bureau_balance tables.
        
        Inputs:
            self
            
        Returns:
            Final pre=processed and merged bureau and burea_balance tables
        '''
        #preprocessing the bureau_balance first
        aggregated_bureau_balance = self.preprocess_bureau_balance()
        bureau_add_feature = self.preprocess_bureau_add_feature()
        #preprocessing the bureau table next, by combining it with the aggregated bureau_balance
        bureau_merged_aggregated = self.preprocess_bureau_aggregate_main(bureau_add_feature, aggregated_bureau_balance)
        return bureau_merged_aggregated