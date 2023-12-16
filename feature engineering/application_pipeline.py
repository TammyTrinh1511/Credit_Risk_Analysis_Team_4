import gc
import pickle
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


class preprocess_application_train_test:
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

        self.verbose = verbose
        self.dump_to_pickle = dump_to_pickle
        self.file_directory = file_directory

    def load_dataframes(self):
        '''
        Function to load the application_train.csv and application_test.csv DataFrames.

        Inputs:
            self

        Returns:
            None
        '''

        if self.verbose:
            self.start = datetime.now()
            print('#######################################################')
            print('#        Pre-processing application_train.csv         #')
            print('#        Pre-processing application_test.csv          #')
            print('#######################################################')
            print("\nLoading the DataFrame, credit_card_balance.csv, into memory...")

        application_train = pd.read_csv(
            self.file_directory + 'dseb63_application_train.csv')
        application_test = pd.read_csv(
            self.file_directory + 'dseb63_application_test.csv')
        self.initial_shape = application_train.shape
        self.application = application_train.append(application_test)
        # del application_train, application_train; gc.collect()
        if self.verbose:
            print("Loaded application_train.csv and application_test.csv")
            print(f"Time Taken to load = {datetime.now() - self.start}")

    def replace_missing_values(self):
        self.application['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        self.application['DAYS_LAST_PHONE_CHANGE'].replace(
            0, np.nan, inplace=True)
        self.application['DAYS_BIRTH'] = self.application['DAYS_BIRTH'] * -1 / 365
        self.application['OBS_30_CNT_SOCIAL_CIRCLE'][self.application['OBS_30_CNT_SOCIAL_CIRCLE'] > 30] == np.nan
        self.application['OBS_60_CNT_SOCIAL_CIRCLE'][self.application['OBS_60_CNT_SOCIAL_CIRCLE'] > 30] == np.nan
        categorical_columns = self.application.select_dtypes(
            include='object').columns.tolist()
        self.application[categorical_columns] = self.application[categorical_columns].fillna(
            'XNA')

    def preprocess_installments_payment_cleaning_add_feature(self):
        '''
        Function to clean the tables, by removing erroneous rows/entries.

        Inputs:
            self

        Returns:
            None
        '''

        if self.verbose:
            print("\nPerforming Data Cleaning...")

        self.replace_missing_values()
        # Flag_document features - count and kurtosis
        docs = [f for f in self.application.columns if 'FLAG_DOC' in f]
        self.application['DOCUMENT_COUNT'] = self.application[docs].sum(axis=1)
        self.application['NEW_DOC_KURT'] = self.application[docs].kurtosis(axis=1)
        # Categorical age - based on target=1 plot
        self.application['AGE_RANGE'] = self.application['DAYS_BIRTH'].apply(lambda x: get_age_label(x, [27, 40, 50, 65, 99]))
        # New features based on External sources
        self.application['EXT_SOURCES_PROD'] = self.application['EXT_SOURCE_1'] * self.application['EXT_SOURCE_2'] * self.application['EXT_SOURCE_3']
        self.application['EXT_SOURCES_WEIGHTED'] = self.application.EXT_SOURCE_1 * 2 + self.application.EXT_SOURCE_2 * 1 + self.application.EXT_SOURCE_3 * 3
        np.warnings.filterwarnings(
            'ignore', r'All-NaN (slice|axis) encountered')
        for function_name in ['min', 'max', 'mean', 'median', 'var']:
            feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
            self.application[feature_name] = eval('np.{}'.format(function_name))(
                self.application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
        self.application['EXT_SOURCE_1^2'] = self.application['EXT_SOURCE_1']**2
        self.application['EXT_SOURCE_2^2'] = self.application['EXT_SOURCE_2']**2
        self.application['EXT_SOURCE_3^2'] = self.application['EXT_SOURCE_3']**2
        self.application['EXT_SOURCE_1 EXT_SOURCE_2'] = self.application['EXT_SOURCE_1'] * self.application['EXT_SOURCE_2']
        self.application['EXT_SOURCE_1 EXT_SOURCE_3'] = self.application['EXT_SOURCE_1'] * self.application['EXT_SOURCE_3']
        self.application['EXT_SOURCE_2 EXT_SOURCE_3'] = self.application['EXT_SOURCE_2'] * self.application['EXT_SOURCE_3']
        self.application['PHONE_TO_EMPLOY_RATIO'] = self.application['DAYS_LAST_PHONE_CHANGE'] / self.application['DAYS_EMPLOYED']
        self.application['APP_SCORE1_TO_FAM_CNT_RATIO'] = self.application['EXT_SOURCE_1'] / self.application['CNT_FAM_MEMBERS']
        self.application['APP_SCORE1_TO_GOODS_RATIO'] = self.application['EXT_SOURCE_1'] / self.application['AMT_GOODS_PRICE']
        self.application['APP_SCORE1_TO_CREDIT_RATIO'] = self.application['EXT_SOURCE_1'] / self.application['AMT_CREDIT']
        self.application['APP_SCORE1_TO_SCORE2_RATIO'] = self.application['EXT_SOURCE_1'] / self.application['EXT_SOURCE_2']
        self.application['APP_SCORE1_TO_SCORE3_RATIO'] = self.application['EXT_SOURCE_1'] / self.application['EXT_SOURCE_3']
        self.application['APP_SCORE2_TO_CREDIT_RATIO'] = self.application['EXT_SOURCE_2'] / self.application['AMT_CREDIT']
        self.application['APP_SCORE2_TO_CITY_RATING_RATIO'] = self.application['EXT_SOURCE_2'] / self.application['REGION_RATING_CLIENT_W_CITY']
        self.application['APP_SCORE2_TO_POP_RATIO'] = self.application['EXT_SOURCE_2'] / self.application['REGION_POPULATION_RELATIVE']
        self.application['APP_SCORE2_TO_PHONE_CHANGE_RATIO'] = self.application['EXT_SOURCE_2'] / self.application['DAYS_LAST_PHONE_CHANGE']

        # Credit ratios
        self.application['CREDIT_TO_ANNUITY_RATIO'] = self.application['AMT_CREDIT'] / \
            self.application['AMT_ANNUITY']
        self.application['CREDIT_TO_GOODS_RATIO'] = self.application['AMT_CREDIT'] / \
            self.application['AMT_GOODS_PRICE']
        self.application['GOODS_INCOME_RATIO'] = self.application['AMT_GOODS_PRICE'] / \
            self.application['AMT_INCOME_TOTAL']
        
        # Income features
        inc_by_org = self.application[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
        self.application['NEW_INC_BY_ORG'] = self.application['ORGANIZATION_TYPE'].map(inc_by_org)
        self.application['ANNUITY_TO_INCOME_RATIO'] = self.application['AMT_ANNUITY'] / self.application['AMT_INCOME_TOTAL']
        self.application['CREDIT_TO_INCOME_RATIO'] = self.application['AMT_CREDIT'] / self.application['AMT_INCOME_TOTAL']
        self.application['INCOME_TO_EMPLOYED_RATIO'] = self.application['AMT_INCOME_TOTAL'] / self.application['DAYS_EMPLOYED']
        self.application['INCOME_TO_BIRTH_RATIO'] = self.application['AMT_INCOME_TOTAL'] / self.application['DAYS_BIRTH']
        self.application['INCOME_ANNUITY_DIFF'] = self.application['AMT_INCOME_TOTAL'] - self.application['AMT_ANNUITY']
        self.application['INCOME_EXT_RATIO'] = self.application['AMT_INCOME_TOTAL'] / self.application['EXT_SOURCE_3']
        self.application['CREDIT_EXT_RATIO'] = self.application['AMT_CREDIT'] / self.application['EXT_SOURCE_3']
        self.application['INCOME_APARTMENT_AVG_MUL'] = self.application['APARTMENTS_SUM_AVG'] * self.application['AMT_INCOME_TOTAL']
        self.application['INCOME_APARTMENT_MODE_MUL'] = self.application['APARTMENTS_SUM_MODE'] * self.application['AMT_INCOME_TOTAL']
        self.application['INCOME_APARTMENT_MEDI_MUL'] = self.application['APARTMENTS_SUM_MEDI'] * self.application['AMT_INCOME_TOTAL']
        self.application['INCOME_PER_CHILD'] = self.application['AMT_INCOME_TOTAL'] / (1 + self.application['CNT_CHILDREN'])
        self.application['INCOME_PER_PERSON'] = self.application['AMT_INCOME_TOTAL'] / self.application['CNT_FAM_MEMBERS']
        self.application['INCOME_CREDIT_PERCENTAGE'] = self.application['AMT_INCOME_TOTAL'] / self.application['AMT_CREDIT']
        
        # Time ratios
        self.application['EMPLOYED_TO_BIRTH_RATIO'] = self.application['DAYS_EMPLOYED'] / self.application['DAYS_BIRTH']
        self.application['ID_TO_BIRTH_RATIO'] = self.application['DAYS_ID_PUBLISH'] / self.application['DAYS_BIRTH']
        self.application['CAR_TO_BIRTH_RATIO'] = self.application['OWN_CAR_AGE'] / self.application['DAYS_BIRTH']
        self.application['CAR_TO_EMPLOYED_RATIO'] = self.application['OWN_CAR_AGE'] / self.application['DAYS_EMPLOYED']
        self.application['PHONE_TO_BIRTH_RATIO'] = self.application['DAYS_LAST_PHONE_CHANGE'] / self.application['DAYS_BIRTH']
        
        # apartment scores
        self.application['APARTMENTS_SUM_AVG'] = self.application['APARTMENTS_AVG'] + self.application['BASEMENTAREA_AVG'] + self.application['YEARS_BEGINEXPLUATATION_AVG'] + self.application[
            'YEARS_BUILD_AVG'] + self.application['COMMONAREA_AVG'] + self.application['ELEVATORS_AVG'] + self.application['ENTRANCES_AVG'] + self.application[
            'FLOORSMAX_AVG'] + self.application['FLOORSMIN_AVG'] + self.application['LANDAREA_AVG'] + self.application['LIVINGAPARTMENTS_AVG'] + self.application[
            'LIVINGAREA_AVG'] + self.application['NONLIVINGAPARTMENTS_AVG'] + self.application['NONLIVINGAREA_AVG']

        self.application['APARTMENTS_SUM_MODE'] = self.application['APARTMENTS_MODE'] + self.application['BASEMENTAREA_MODE'] + self.application['YEARS_BEGINEXPLUATATION_MODE'] + self.application[
            'YEARS_BUILD_MODE'] + self.application['COMMONAREA_MODE'] + self.application['ELEVATORS_MODE'] + self.application['ENTRANCES_MODE'] + self.application[
            'FLOORSMAX_MODE'] + self.application['FLOORSMIN_MODE'] + self.application['LANDAREA_MODE'] + self.application['LIVINGAPARTMENTS_MODE'] + self.application[
            'LIVINGAREA_MODE'] + self.application['NONLIVINGAPARTMENTS_MODE'] + self.application['NONLIVINGAREA_MODE'] + self.application['TOTALAREA_MODE']

        self.application['APARTMENTS_SUM_MEDI'] = self.application['APARTMENTS_MEDI'] + self.application['BASEMENTAREA_MEDI'] + self.application['YEARS_BEGINEXPLUATATION_MEDI'] + self.application[
            'YEARS_BUILD_MEDI'] + self.application['COMMONAREA_MEDI'] + self.application['ELEVATORS_MEDI'] + self.application['ENTRANCES_MEDI'] + self.application[
            'FLOORSMAX_MEDI'] + self.application['FLOORSMIN_MEDI'] + self.application['LANDAREA_MEDI'] + self.application['LIVINGAPARTMENTS_MEDI'] + self.application[
            'LIVINGAREA_MEDI'] + self.application['NONLIVINGAPARTMENTS_MEDI'] + self.application['NONLIVINGAREA_MEDI']
        
        # features eng
        self.application['CHILDRE_RATIO'] = self.application['CNT_CHILDREN'] / self.application['CNT_FAM_MEMBERS']
        
        self.application['PAYMENT_RATE'] = self.application['AMT_ANNUITY'] / self.application['AMT_CREDIT']

        # OBS And DEF
        self.application['OBS_30_60_SUM'] = self.application['OBS_30_CNT_SOCIAL_CIRCLE'] + self.application['OBS_60_CNT_SOCIAL_CIRCLE']
        self.application['DEF_30_60_SUM'] = self.application['DEF_30_CNT_SOCIAL_CIRCLE'] + self.application['DEF_60_CNT_SOCIAL_CIRCLE']
        self.application['OBS_DEF_30_MUL'] = self.application['OBS_30_CNT_SOCIAL_CIRCLE'] * self.application['DEF_30_CNT_SOCIAL_CIRCLE']
        self.application['OBS_DEF_60_MUL'] = self.application['OBS_60_CNT_SOCIAL_CIRCLE'] * self.application['DEF_60_CNT_SOCIAL_CIRCLE']
        self.application['SUM_OBS_DEF_ALL'] = self.application['OBS_30_CNT_SOCIAL_CIRCLE'] + self.application['DEF_30_CNT_SOCIAL_CIRCLE'] + self.application[
            'OBS_60_CNT_SOCIAL_CIRCLE'] + self.application['DEF_60_CNT_SOCIAL_CIRCLE']
        self.application['OBS_30_CREDIT_RATIO'] = self.application['AMT_CREDIT'] / self.application['OBS_30_CNT_SOCIAL_CIRCLE']
        self.application['OBS_60_CREDIT_RATIO'] = self.application['AMT_CREDIT'] / self.application['OBS_60_CNT_SOCIAL_CIRCLE']
        self.application['DEF_30_CREDIT_RATIO'] = self.application['AMT_CREDIT'] / self.application['DEF_30_CNT_SOCIAL_CIRCLE']
        self.application['DEF_60_CREDIT_RATIO'] = self.application['AMT_CREDIT'] / self.application['DEF_60_CNT_SOCIAL_CIRCLE']

        # Flag Documents combined
        self.application['SUM_FLAGS_DOCUMENTS'] = self.application['FLAG_DOCUMENT_3'] + self.application['FLAG_DOCUMENT_5'] + self.application['FLAG_DOCUMENT_6'] + self.application[
            'FLAG_DOCUMENT_7'] + self.application['FLAG_DOCUMENT_8'] + self.application['FLAG_DOCUMENT_9'] + self.application[
            'FLAG_DOCUMENT_11'] + self.application['FLAG_DOCUMENT_13'] + self.application['FLAG_DOCUMENT_14'] + self.application[
            'FLAG_DOCUMENT_15'] + self.application['FLAG_DOCUMENT_16'] + self.application['FLAG_DOCUMENT_17'] + self.application[
            'FLAG_DOCUMENT_18'] + self.application['FLAG_DOCUMENT_19'] + self.application['FLAG_DOCUMENT_21']

        # details change
        self.application['DAYS_DETAILS_CHANGE_MUL'] = self.application['DAYS_LAST_PHONE_CHANGE'] * self.application['DAYS_REGISTRATION'] * \
            self.application['DAYS_ID_PUBLISH']
        self.application['DAYS_DETAILS_CHANGE_SUM'] = self.application['DAYS_LAST_PHONE_CHANGE'] + self.application['DAYS_REGISTRATION'] + \
            self.application['DAYS_ID_PUBLISH']

        # enquires
        self.application['AMT_ENQ_SUM'] = self.application['AMT_REQ_CREDIT_BUREAU_HOUR'] + self.application['AMT_REQ_CREDIT_BUREAU_DAY'] + self.application['AMT_REQ_CREDIT_BUREAU_WEEK'] + self.application[
            'AMT_REQ_CREDIT_BUREAU_MON'] + self.application['AMT_REQ_CREDIT_BUREAU_QRT'] + self.application['AMT_REQ_CREDIT_BUREAU_YEAR']
        self.application['ENQ_CREDIT_RATIO'] = self.application['AMT_ENQ_SUM'] / self.application['AMT_CREDIT']
        self.application['CNT_NON_CHILD'] = self.application['CNT_FAM_MEMBERS'] - self.application['CNT_CHILDREN']
        self.application['CHILD_TO_NON_CHILD_RATIO'] = self.application['CNT_CHILDREN'] / self.application['CNT_NON_CHILD']
        self.application['INCOME_PER_NON_CHILD'] = self.application['AMT_INCOME_TOTAL'] / self.application['CNT_NON_CHILD']
        self.application['CREDIT_PER_PERSON'] = self.application['AMT_CREDIT'] / self.application['CNT_FAM_MEMBERS']
        self.application['CREDIT_PER_CHILD'] = self.application['AMT_CREDIT'] / self.application['CNT_CHILDREN']
        self.application['CREDIT_PER_NON_CHILD'] = self.application['AMT_CREDIT'] / self.application['CNT_NON_CHILD']

        # age bins
        self.application['RETIREMENT_AGE'] = (self.application['DAYS_BIRTH'] < -14000).astype(int)
        self.application['DAYS_BIRTH_QCUT'] = pd.qcut(self.application['DAYS_BIRTH'], q=5, labels=False)

        # long employemnt
        self.application['LONG_EMPLOYMENT'] = (self.application['DAYS_EMPLOYED'] < -2000).astype(int)

        bins = [0, 30000, 65000, 95000, 130000, 160000, 190880, 220000, 275000, 325000, np.inf]
        labels = range(1, 11)
        self.application['INCOME_BAND'] = pd.cut(
            self.application['AMT_INCOME_TOTAL'], bins=bins, labels=labels, right=False)
        # flag asset
        self.application['FLAG_ASSET'] = np.nan
        filter_0 = (self.application['FLAG_OWN_CAR'] == 'N') & (self.application['FLAG_OWN_REALTY'] == 'N')
        filter_1 = (self.application['FLAG_OWN_CAR'] == 'Y') & (self.application['FLAG_OWN_REALTY'] == 'N')
        filter_2 = (self.application['FLAG_OWN_CAR'] == 'N') & (self.application['FLAG_OWN_REALTY'] == 'Y')
        filter_3 = (self.application['FLAG_OWN_CAR'] == 'Y') & (self.application['FLAG_OWN_REALTY'] == 'Y')

        self.application.loc[filter_0, 'FLAG_ASSET'] = 0
        self.application.loc[filter_1, 'FLAG_ASSET'] = 1
        self.application.loc[filter_2, 'FLAG_ASSET'] = 2
        self.application.loc[filter_3, 'FLAG_ASSET'] = 3

        # Groupby: Statistics for applications in the same group
        group = ['ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE',
                 'OCCUPATION_TYPE', 'AGE_RANGE', 'CODE_GENDER']
        self.application = do_median(self.application, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_MEDIAN')
        self.application = do_std(self.application, group, 'EXT_SOURCES_MEAN', 'GROUP_EXT_SOURCES_STD')
        self.application = do_mean(self.application, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_MEAN')
        self.application = do_std(self.application, group, 'AMT_INCOME_TOTAL', 'GROUP_INCOME_STD')
        self.application = do_mean(self.application, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_MEAN')
        self.application = do_std(self.application, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP_CREDIT_TO_ANNUITY_STD')
        self.application = do_mean(self.application, group, 'AMT_CREDIT', 'GROUP_CREDIT_MEAN')
        self.application = do_mean(self.application, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_MEAN')
        self.application = do_std(self.application, group, 'AMT_ANNUITY', 'GROUP_ANNUITY_STD')

        # Groupby 2: Statistics for applications with the same credit duration, income type and education
        self.application['CREDIT_TO_ANNUITY_GROUP'] = self.application['CREDIT_TO_ANNUITY_RATIO'].apply(lambda x: _group_credit_to_annuity(x))
        group = ['CREDIT_TO_ANNUITY_GROUP',
                 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']
        self.application = do_median(self.application, group, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_MEDIAN')
        self.application = do_std(self.application, group, 'EXT_SOURCES_MEAN', 'GROUP1_EXT_SOURCES_STD')
        self.application = do_median(self.application, group, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_MEDIAN')
        self.application = do_std(self.application, group, 'AMT_INCOME_TOTAL', 'GROUP1_INCOME_STD')
        self.application = do_median(self.application, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_MEDIAN')
        self.application = do_std(self.application, group, 'CREDIT_TO_ANNUITY_RATIO', 'GROUP1_CREDIT_TO_ANNUITY_STD')
        self.application = do_median(self.application, group, 'AMT_CREDIT', 'GROUP1_CREDIT_MEDIAN')
        self.application = do_std(self.application, group, 'AMT_CREDIT', 'GROUP1_CREDIT_STD')
        self.application = do_median(self.application, group, 'AMT_ANNUITY', 'GROUP1_ANNUITY_MEDIAN')
        self.application = do_std(self.application, group, 'AMT_ANNUITY', 'GROUP1_ANNUITY_STD')

        # now we will create features based on categorical interactions
        columns_to_aggregate_on = [
            ['NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE'],
            ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE'],
            ['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE'],
            ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
            ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

        ]
        aggregations = {
            'EXT_SOURCE_1': ['mean', 'max', 'min'],
            'EXT_SOURCE_2': ['mean', 'max', 'min'],
            'EXT_SOURCE_3': ['mean', 'max', 'min'],
            'AMT_ANNUITY': ['mean', 'max', 'min'],
            'AMT_INCOME_TOTAL': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_AVG': ['mean', 'max', 'min'],
            'APARTMENTS_SUM_MEDI': ['mean', 'max', 'min'],
        }

        # extracting values
        for group in columns_to_aggregate_on:
            # grouping based on categories
            grouped_interactions = self.application.groupby(
                group).agg(aggregations)
            grouped_interactions.columns = ['_'.join(ele).upper() + '_AGG_' + '_'.join(group) for ele in grouped_interactions.columns]
            # saving the grouped interactions to pickle file
            group_name = '_'.join(group)

            self.application = self.application.join(grouped_interactions, on=group)

        # Encode categorical features (LabelEncoder)
        self.application, le_encoded_cols = label_encoder(self.application, None)
        self.application = drop_application_columns(self.application)

        if self.verbose:
            print("Done.")

    def main(self):
        '''
        Function to be called for complete preprocessing of application_train and application_test tables.

        Inputs:
            self

        Returns:
            Final pre=processed application_train and application_test tables.
        '''

        # loading the DataFrames first
        self.load_dataframes()
        # first doing Data Cleaning
        self.preprocess_installments_payment_cleaning_add_feature()


        if self.verbose:
            print('Done preprocessing appplication_train and application_test.')
            print(f"\nInitial Size of application_train: {self.initial_shape}")
            print(
                f'Size of application_train after Pre-Processing and Feature Engineering: {self.application.shape}')
            print(f'\nTotal Time Taken = {datetime.now() - self.start}')

        if self.dump_to_pickle:
            if self.verbose:
                print('\nPickling pre-processed application_train and application_test to application_train_preprocessed.pkl and application_test_preprocessed, respectively.')
            with open(self.file_directory + 'application_preprocessed.pkl', 'wb') as f:
                pickle.dump(self.application, f)
            if self.verbose:
                print('Done.')
        if self.verbose:
            print('-'*100)

        return self.application


def _group_credit_to_annuity(x):
    """ Return the credit duration group label (int). """
    if x == np.nan:
        return 0
    elif x <= 6:
        return 1
    elif x <= 12:
        return 2
    elif x <= 18:
        return 3
    elif x <= 24:
        return 4
    elif x <= 30:
        return 5
    elif x <= 36:
        return 6
    else:
        return 7


def drop_application_columns(df):
    """ Drop features based on VIF, permutation feature importance. """
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG', 'HOUSETYPE_MODE',
        'FONDKAPREMONT_MODE', 'EMERGENCYSTATE_MODE'
    ]
    # Drop most flag document columns (due to EDA)
    for doc_num in [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df


def get_age_label(days_birth, ranges):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    for label, max_age in enumerate(ranges):
        if age_years <= max_age:
            return label + 1
    else:
        return 0
