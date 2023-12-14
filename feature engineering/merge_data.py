from application_pipeline import *
from credit_card_pipeline import *
from bureau_pipeline import *
from installments_payments_pipeline import *
from pos_cash_pipeline import *
from previous_application_pipeline import *
import config
import pandas as pd 
import gc 

def merge_data(file_directory): 
    """
    This function merges multiple dataframes based on a common column 'SK_ID_CURR',
    performs some additional data preprocessing, and saves the merged dataframe to a CSV file.
    
    Args:
        file_directory (str): The directory where the input data files are located.
    
    Returns:
        None
    """
    
    # Preprocess application_train_test data
    df = preprocess_application_train_test(file_directory).main()
    
    # Preprocess bureau_balance_and_bureau data
    bureau = preprocess_bureau_balance_and_bureau(file_directory).main()
    df = pd.merge(df, bureau, on='SK_ID_CURR', how='left')
    
    # Preprocess previous_application data
    previous_application = preprocess_previous_application(file_directory).main()
    df = pd.merge(df, previous_application, on='SK_ID_CURR', how='left')
    
    # Preprocess POS_CASH_balance data
    pos_cash = preprocess_POS_CASH_balance(file_directory).main()
    df = pd.merge(df, pos_cash, on='SK_ID_CURR', how='left')
    
    # Preprocess installments_payments data
    installments_payments = preprocess_installments_payments(file_directory).main()
    df = pd.merge(df, installments_payments, on='SK_ID_CURR', how='left')
    # Preprocess credit_card_balance data
    credit_card = preprocess_credit_card_balance(file_directory).main()
    df = pd.merge(df, credit_card, on='SK_ID_CURR', how='left')
    
    # Add additional ratio features
    df = add_ratios_features(df)
    
    # Add additional groupby features
    df = add_groupby_features(df)
    
    # Delete unnecessary variables to free up memory
    del bureau, previous_application, pos_cash, installments_payments, credit_card
    
    # Reduce memory usage of the dataframe
    df = reduce_mem_usage(df)
    # Create the output directory if it doesn't exist
    if not os.path.exists("../processed_data/"):
        os.makedirs("../processed_data/")
    
    # Save the merged dataframe to a CSV file
    df.to_csv('../processed_data/dataframe_merged.csv')


def add_ratios_features(df):
    """
    This function adds various ratio features to the dataframe.
    Args:
        df (pandas.DataFrame): The dataframe to which the features will be added.
    Returns:
        pandas.DataFrame: The dataframe with the added features.
    """
    # CREDIT TO INCOME RATIO
    df['BUREAU_INCOME_CREDIT_RATIO'] = df['BUREAU_AMT_CREDIT_SUM_MEAN'] / df['AMT_INCOME_TOTAL']
    df['BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO'] = df['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM'] / df['AMT_INCOME_TOTAL']
    # PREVIOUS TO CURRENT CREDIT RATIO
    df['CURRENT_TO_APPROVED_CREDIT_MIN_RATIO'] = df['APPROVED_AMT_CREDIT_MIN'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MAX_RATIO'] = df['APPROVED_AMT_CREDIT_MAX'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO'] = df['APPROVED_AMT_CREDIT_MEAN'] / df['AMT_CREDIT']
    # PREVIOUS TO CURRENT ANNUITY RATIO
    df['CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO'] = df['APPROVED_AMT_ANNUITY_MAX'] / df['AMT_ANNUITY']
    df['CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO'] = df['APPROVED_AMT_ANNUITY_MEAN'] / df['AMT_ANNUITY']
    df['PAYMENT_MIN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MIN'] / df['AMT_ANNUITY']
    df['PAYMENT_MAX_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MAX'] / df['AMT_ANNUITY']
    df['PAYMENT_MEAN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MEAN'] / df['AMT_ANNUITY']
    # PREVIOUS TO CURRENT CREDIT TO ANNUITY RATIO
    df['CTA_CREDIT_TO_ANNUITY_MAX_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX'] / df[
        'CREDIT_TO_ANNUITY_RATIO']
    df['CTA_CREDIT_TO_ANNUITY_MEAN_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN'] / df[
        'CREDIT_TO_ANNUITY_RATIO']
    # DAYS DIFFERENCES AND RATIOS
    df['DAYS_DECISION_MEAN_TO_BIRTH'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_CREDIT_MEAN_TO_BIRTH'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_DECISION_MEAN_TO_EMPLOYED'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_EMPLOYED']
    df['DAYS_CREDIT_MEAN_TO_EMPLOYED'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_EMPLOYED']
    return df

def add_groupby_features(df):
    """Group some features by duration (credit/annuity) and extract the mean, median and std.

    Arguments:
        df: pandas DataFrame with features from all csv files

    Returns:
        df: Same DataFrame with the new features
    """
    g = 'CREDIT_TO_ANNUITY_RATIO'
    feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'BUREAU_ACTIVE_DAYS_CREDIT_MEAN',
             'APPROVED_CNT_PAYMENT_MEAN', 'EXT_SOURCES_PROD', 'CREDIT_TO_GOODS_RATIO',
             'INS_DAYS_ENTRY_PAYMENT_MAX', 'EMPLOYED_TO_BIRTH_RATIO', 'EXT_SOURCES_MEAN',
             'DAYS_REGISTRATION', 'DAYS_EMPLOYED', 'DAYS_BIRTH']
    agg = df.groupby(g)[feats].agg(['mean', 'median', 'std'])
    agg.columns = pd.Index(['CTAR_' + e[0] + '_' + e[1].upper() for e in agg.columns.tolist()])
    df = df.join(agg, how='left', on=g)
    del agg
    gc.collect()
    return df   

if __name__ == "__main__":
    merge_data(FILE_DIRECTORY)