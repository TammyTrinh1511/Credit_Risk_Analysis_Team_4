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

class preprocess_credit_card_balance:
    '''
    Preprocess the credit_card_balance table.
    Contains 5 member functions:
        1. init method
        2. load_dataframe method
        3. preprocess_credit_card_add_feature method
        4. preprocess_credit_card_aggregate method
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
        Function to load the credit_card_balance.csv DataFrame.
        
        Inputs:
            self
            
        Returns:
            None
        '''
        
        if self.verbose:
            self.start = datetime.now()
            print('#########################################################')
            print('#        Pre-processing credit_card_balance.csv         #')
            print('#########################################################')
            print("\nLoading the DataFrame, credit_card_balance.csv, into memory...")

        self.cc_balance = pd.read_csv(self.file_directory + 'dseb63_credit_card_balance.csv')
        self.initial_size = self.cc_balance.shape

        if self.verbose:
            print("Loaded credit_card_balance.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")
            
    def preprocess_credit_card_add_feature(self):
        '''
        Function to preprocess the table, by removing erroneous points, and then creating new domain based features.
        
        Inputs:
            self
            
        Returns:
            None
        '''
        
        if self.verbose:
            start = datetime.now()
            print("\nStarting Preprocessing and Feature Engineering...")
        # Rename a column for consistency
        self.cc_balance.rename(columns={'AMT_RECIVABLE': 'AMT_RECEIVABLE'}, inplace=True)
        # Feature: Amount used from credit limit
        self.cc_balance['LIMIT_USE'] = self.cc_balance['AMT_BALANCE'] / self.cc_balance['AMT_CREDIT_LIMIT_ACTUAL']
        # Feature: Current payment / Minimum payment
        self.cc_balance['PAYMENT_DIV_MIN'] = self.cc_balance['AMT_PAYMENT_CURRENT'] / self.cc_balance['AMT_INST_MIN_REGULARITY']
        # Feature: Late payment indicator
        self.cc_balance['LATE_PAYMENT'] = self.cc_balance['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        # Feature: Drawing limit ratio
        self.cc_balance['DRAWING_LIMIT_RATIO'] = self.cc_balance['AMT_DRAWINGS_ATM_CURRENT'] / self.cc_balance['AMT_CREDIT_LIMIT_ACTUAL']
        # Additional features
        self.cc_balance['AMT_DRAWING_SUM'] = self.cc_balance[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
        self.cc_balance['BALANCE_LIMIT_RATIO'] = self.cc_balance['AMT_BALANCE'] / self.cc_balance['AMT_CREDIT_LIMIT_ACTUAL']
        self.cc_balance['CNT_DRAWING_SUM'] = self.cc_balance[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']].sum(axis=1)
        self.cc_balance['MIN_PAYMENT_RATIO'] = self.cc_balance['AMT_PAYMENT_CURRENT'] / self.cc_balance['AMT_INST_MIN_REGULARITY']
        self.cc_balance['PAYMENT_MIN_DIFF'] = self.cc_balance['AMT_PAYMENT_CURRENT'] - self.cc_balance['AMT_INST_MIN_REGULARITY']
        self.cc_balance['MIN_PAYMENT_TOTAL_RATIO'] = self.cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] / self.cc_balance['AMT_INST_MIN_REGULARITY']
        self.cc_balance['PAYMENT_MIN_DIFF_TOTAL'] = self.cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] - self.cc_balance['AMT_INST_MIN_REGULARITY']
        self.cc_balance['AMT_INTEREST_RECEIVABLE'] = self.cc_balance['AMT_TOTAL_RECEIVABLE'] - self.cc_balance['AMT_RECEIVABLE_PRINCIPAL']
        self.cc_balance['SK_DPD_RATIO'] = self.cc_balance['SK_DPD'] / self.cc_balance['SK_DPD_DEF']

        # Feature: Rate of payback of loans - Number of installments paid by customer per loan
        grp = self.cc_balance.groupby(by = ['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index().rename(index = str, columns = {'SK_ID_PREV': 'NO_LOANS'})
        self.cc_balance = self.cc_balance.merge(grp, on = ['SK_ID_CURR'], how = 'left')
        grp = self.cc_balance.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].max().reset_index().rename(index=str, columns={'CNT_INSTALMENT_MATURE_CUM': 'NO_INSTALMENTS'})
        grp1 = grp.groupby(by=['SK_ID_CURR'])['NO_INSTALMENTS'].sum().reset_index().rename(index=str, columns={'NO_INSTALMENTS': 'TOTAL_INSTALMENTS'})
        self.cc_balance = self.cc_balance.merge(grp1, on=['SK_ID_CURR'], how='left')
        self.cc_balance['INSTALLMENTS_PER_LOAN'] = (self.cc_balance['TOTAL_INSTALMENTS'] / self.cc_balance['NO_LOANS']).astype('uint32')
        del self.cc_balance['TOTAL_INSTALMENTS'], self.cc_balance['NO_LOANS'], grp, grp1
        gc.collect()

        # Feature: Average number of times days past due has occurred per customer
        def count_non_zero_dpd(DPD):
            return sum(DPD != 0)

        grp = self.cc_balance.groupby(by=['SK_ID_CURR', 'SK_ID_PREV']).apply(lambda x: count_non_zero_dpd(x.SK_DPD)).reset_index().rename(index=str, columns={0: 'NO_DPD'})
        grp1 = grp.groupby(by=['SK_ID_CURR'])['NO_DPD'].mean().reset_index().rename(index=str, columns={'NO_DPD': 'DPD_COUNT'})
        self.cc_balance = self.cc_balance.merge(grp1, on=['SK_ID_CURR'], how='left')

        # Feature: Percentage of minimum payments missed
        def percentage_missed_payments(min_pay, total_pay):
            return (100 * sum(total_pay < min_pay)) / len(min_pay)

        grp = self.cc_balance.groupby(by=['SK_ID_CURR']).apply(lambda x: percentage_missed_payments(x.AMT_INST_MIN_REGULARITY, x.AMT_PAYMENT_CURRENT)).reset_index().rename(index=str, columns={0: 'PERCENTAGE_MISSED_PAYMENTS'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')

        # Feature: Ratio of cash vs. card swipes
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index().rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'DRAWINGS_ATM'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'DRAWINGS_TOTAL'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        self.cc_balance['CASH_CARD_RATIO1'] = (self.cc_balance['DRAWINGS_ATM'] / self.cc_balance['DRAWINGS_TOTAL']) * 100
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['CASH_CARD_RATIO1'].mean().reset_index().rename(index=str, columns={'CASH_CARD_RATIO1': 'CASH_CARD_RATIO'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        del self.cc_balance['CASH_CARD_RATIO1'], self.cc_balance['DRAWINGS_ATM'], self.cc_balance['DRAWINGS_TOTAL']
        gc.collect()

        # Feature: Average drawing per customer
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'TOTAL_DRAWINGS'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index().rename(index=str, columns={'CNT_DRAWINGS_CURRENT': 'NO_DRAWINGS'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        self.cc_balance['DRAWINGS_RATIO1'] = (self.cc_balance['TOTAL_DRAWINGS'] / self.cc_balance['NO_DRAWINGS']) * 100
        grp = self.cc_balance.groupby(by=['SK_ID_CURR'])['DRAWINGS_RATIO1'].mean().reset_index().rename(index=str, columns={'DRAWINGS_RATIO1': 'DRAWINGS_RATIO'})
        self.cc_balance = self.cc_balance.merge(grp, on=['SK_ID_CURR'], how='left')
        del self.cc_balance['TOTAL_DRAWINGS'], self.cc_balance['NO_DRAWINGS'], self.cc_balance['DRAWINGS_RATIO1'], grp

        # Calculating the rolling Exponential Weighted Moving Average over months for certain features
        rolling_columns = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECEIVABLE', 'AMT_TOTAL_RECEIVABLE', 'AMT_DRAWING_SUM', 'BALANCE_LIMIT_RATIO', 'CNT_DRAWING_SUM', 'MIN_PAYMENT_RATIO', 'PAYMENT_MIN_DIFF', 'MIN_PAYMENT_TOTAL_RATIO', 'AMT_INTEREST_RECEIVABLE', 'SK_DPD_RATIO']
        self.cc_balance_balance = self.cc_balance.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[1, 0])

        exp_weighted_columns = ['EXP_' + ele for ele in rolling_columns]
        self.cc_balance[exp_weighted_columns] = self.cc_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])[rolling_columns].transform(lambda x: x.ewm(alpha=0.7).mean())

        if self.verbose:
            print("Done.")
            print(f"Time Taken = {datetime.now() - start}")
            
    def preprocess_credit_card_aggregate(self):
        '''
        Function to perform aggregations of rows of the credit_card_balance table.
        
        Inputs:
            self
        
        Returns:
            aggregated credit_card_balance table.
        '''

        if self.verbose:
            print("\nAggregating the DataFrame")

        # Performing aggregations over SK_ID_PREV
        cc_agg = self.cc_balance.groupby('SK_ID_CURR').agg(CREDIT_CARD_AGG)
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        cc_agg.reset_index(inplace= True)

        # Last month balance of each credit card application
        last_ids = self.cc_balance.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
        last_months_df = self.cc_balance[self.cc_balance.index.isin(last_ids)]
        cc_agg = group_and_merge(last_months_df,cc_agg,'CC_LAST_', {'AMT_BALANCE': ['mean', 'max']})

        # Aggregations for last x months
        for months in [12, 24, 48]:
            cc_prev_id = self.cc_balance[self.cc_balance['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
            cc_recent = self.cc_balance[self.cc_balance['SK_ID_PREV'].isin(cc_prev_id)]
            prefix = 'INS_{}M_'.format(months)
            cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)
                        
        return cc_agg
                    
    def main(self):
        '''
        Function to be called for complete preprocessing and aggregation of credit_card_balance table.
        
        Inputs:
            self
            
        Returns:
            Final pre=processed and aggregated credit_card_balance table.
        '''
        
        #loading the dataframe 
        self.load_dataframe()
        #preprocessing and performing Feature Engineering
        self.data_preprocessing_and_feature_engineering()
        #aggregating over SK_ID_PREV and SK_ID_CURR
        cc_aggregated = self.aggregations()

        if self.verbose:
            print('\nDone preprocessing credit_card_balance.')
            print(f"\nInitial Size of credit_card_balance: {self.initial_size}")
            print(f'Size of credit_card_balance after Pre-Processing, Feature Engineering and Aggregation: {cc_aggregated.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed credit_card_balance to credit_card_balance_preprocessed.pkl')
            with open(self.file_directory + 'credit_card_balance_preprocessed.pkl', 'wb') as f:
                pickle.dump(cc_aggregated, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-'*100)
                    
        return cc_aggregated