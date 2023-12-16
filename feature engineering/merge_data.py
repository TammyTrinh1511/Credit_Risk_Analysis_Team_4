from application_pipeline import *
from credit_card_pipeline import *
from bureau_pipeline import *
from installments_payments_pipeline import *
from pos_cash_pipeline import *
from previous_application_pipeline import *
import config
import pandas as pd 
import gc 
from utils import * 

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



if __name__ == "__main__":
    merge_data(FILE_DIRECTORY)