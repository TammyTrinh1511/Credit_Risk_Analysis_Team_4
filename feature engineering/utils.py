import os
import gc
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from datetime import datetime
from functools import partial
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, iqr, skew

def reduce_mem_usage(data, verbose = True):
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-'*100)
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        print('-'*100)
    
    return data

def group(df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """
    Groups a DataFrame by a specified column and performs aggregations on the grouped data.
    Args:
        df_to_agg (pandas.DataFrame): The DataFrame to be aggregated.
        prefix (str): The prefix to be added to the aggregated column names.
        aggregations (dict): A dictionary specifying the columns to be aggregated and the corresponding aggregation functions.
        aggregate_by (str, optional): The column to group the data by. Defaults to 'SK_ID_CURR'.
    Returns:
        pandas.DataFrame: The aggregated DataFrame.
    """
    # Group the DataFrame by the specified column and perform the aggregations
    agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
    agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                               for e in agg_df.columns.tolist()])
    return agg_df.reset_index()


def group_and_merge (df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
    """
    Group the 'df_to_agg' DataFrame by 'aggregate_by' column and apply the specified aggregations.
    Merge the resulting aggregated DataFrame with 'df_to_merge' DataFrame.
    
    Args:
        df_to_agg (pandas.DataFrame): The DataFrame to be aggregated.
        df_to_merge (pandas.DataFrame): The DataFrame to be merged with the aggregated DataFrame.
        prefix (str): The prefix to be added to the column names of the aggregated DataFrame.
        aggregations (dict): A dictionary specifying the aggregations to be applied.
        aggregate_by (str): The column used for grouping and aggregating.
        
    Returns:
        pandas.DataFrame: The merged DataFrame.
    """
    agg_df = group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
    return df_to_merge.merge(agg_df, how='left', on= aggregate_by)



def do_mean(df, group_cols, counted, agg_name):
    """
    Calculate the mean value of a column based on groupings in another column.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        group_cols (list): The column(s) to group by.
        counted (str): The column to calculate the mean of.
        agg_name (str): The name of the new column to store the mean value.
    Returns:
        pandas.DataFrame: The DataFrame with the calculated mean values added.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_median(df, group_cols, counted, agg_name):
    """
    Calculate the median of a column in a DataFrame grouped by specified columns.
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    group_cols (list): The columns to group by.
    counted (str): The column to calculate the median of.
    agg_name (str): The name of the resulting column.
    Returns:
    DataFrame: The DataFrame with the median column added.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].median().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_std(df, group_cols, counted, agg_name):
    """
    Calculate the standard deviation of a column for each group in the dataframe and merge the result back to the dataframe.
    Args:
        df (pandas.DataFrame): The input dataframe.
        group_cols (list): List of column names to group by.
        counted (str): Name of the column to calculate the standard deviation on.
        agg_name (str): Name of the column to store the standard deviation result.
    Returns:
        pandas.DataFrame: The input dataframe with the standard deviation column merged.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].std().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    return df


def do_sum(df, group_cols, counted, agg_name):
    """
    Performs a sum aggregation on a DataFrame based on specified group columns.
    Args:
        df (pandas.DataFrame): The input DataFrame.
        group_cols (list): The columns to group by.
        counted (str): The column to sum.
        agg_name (str): The name for the aggregated column.
    Returns:
        pandas.DataFrame: The DataFrame with the aggregated column added.
    """
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    import gc 
    gc.collect()
    return df


def one_hot_encoder(df, categorical_columns=None, nan_as_category=True):
    """
    Create a new column for each categorical value in categorical columns.
    
    Args:
        df (pandas.DataFrame): The dataframe to encode.
        categorical_columns (list): List of column names to encode. If None, all object dtype columns will be encoded.
        nan_as_category (bool): Whether to encode NaN values as a separate category.
    
    Returns:
        encoded_df (pandas.DataFrame): The encoded dataframe.
        new_columns (list): List of new column names created during encoding.
    """
    original_columns = list(df.columns)
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    categorical_columns = [c for c in df.columns if c not in original_columns]
    return df, categorical_columns


def label_encoder(df, categorical_columns=None):
    """
    Encode categorical values as integers using pandas.factorize.
    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_columns (Optional[List[str]]): The columns to encode. If None, all object columns will be encoded.
    Returns:
        Tuple[pd.DataFrame, List[str]]: The encoded DataFrame and the list of encoded columns.
    """
    if not categorical_columns:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
    return df, categorical_columns


def add_features(feature_name, aggs, features, feature_names, groupby):
    """
    Add new features to the existing features dataframe.
    Args:
        feature_name (str): Name of the feature.
        aggs (list): List of aggregation functions to apply.
        features (DataFrame): Existing features dataframe.
        feature_names (list): List of feature names.
        groupby (DataFrame): Groupby object.
    Returns:
        features (DataFrame): Updated features dataframe.
        feature_names (list): Updated list of feature names.
    """
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])

    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg

        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                     columns={feature_name: '{}_{}'.format(feature_name,agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    """
    Add aggregated features to a group.
    
    Parameters:
        features (dict): Dictionary of features.
        gr_ (DataFrame): Grouped DataFrame.
        feature_name (str): Name of the feature to aggregate.
        aggs (list): List of aggregation methods.
        prefix (str): Prefix to add to the aggregated feature names.
        
    Returns:
        dict: Updated dictionary of features.
    """
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features


def add_trend_feature(features, gr, feature_name, prefix):
    """
    Add a trend feature to the given features dictionary.
    Args:
        features (dict): The dictionary of features.
        gr (pandas.DataFrame): The groupby object.
        feature_name (str): The name of the feature.
        prefix (str): The prefix to be added to the feature name.
    Returns:
        dict: The updated features dictionary.
    """
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    # Add the trend feature to the features dictionary
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def parallel_apply(groups, func, index_name='Index', num_workers=0, chunk_size=100000):
    """
    Apply a function to groups of data in parallel and return the results in a DataFrame.
    Args:
        groups (iterable): The groups of data to apply the function to.
        func (callable): The function to apply to each group of data.
        index_name (str, optional): The name of the index in the resulting DataFrame. Defaults to 'Index'.
        num_workers (int, optional): The number of worker processes to use. If 0, defaults to 4. Defaults to 0.
        chunk_size (int, optional): The size of each chunk of data to process in parallel. Defaults to 100000.
    Returns:
        pd.DataFrame: The DataFrame containing the results of applying the function to the groups of data.
    """
    if num_workers <= 0: num_workers = 4
    indeces, features = [], []
    for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
        with mp.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def chunk_groups(groupby_object, chunk_size):
    """
    Yield chunks of groups from a pandas GroupBy object.
    Args:
        groupby_object (pandas.core.groupby.GroupBy): The GroupBy object to chunk.
        chunk_size (int): The size of each chunk.
    Yields:
        tuple: A tuple containing the group indices and corresponding groups in each chunk.
    """
    # Get the total number of groups
    n_groups = groupby_object.ngroups
    # Initialize empty lists to store groups and indices
    group_chunk, index_chunk = [], []
    # Iterate over each group in the GroupBy object
    for i, (index, df) in enumerate(groupby_object):
        # Append the group and index to the corresponding lists
        group_chunk.append(df)
        index_chunk.append(index)
        # Check if the chunk size is reached or if it's the last group
        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            # Create copies of the group and index lists
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            # Reset the group and index lists
            group_chunk, index_chunk = [], []
            # Yield the chunk of groups and indices
            yield index_chunk_, group_chunk_

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