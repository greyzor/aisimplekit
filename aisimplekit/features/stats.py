"""
Some statistics extraction helpers.
"""
import gc

def do_count(df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    """Count occurences on groups of columns.

    :param df: input dataframe.
    :type df: pd.DataFrame
    :param group_cols: the columns we want to group on.
    :type group_cols: list
    :param agg_name: aggregation column name
    :type agg_name: str
    :param agg_type: type of new aggregated column
    :type agg_type: str
    :param show_max: debug option to show max value of aggregated column
    :type show_max: bool
    :param show_agg: debug option to show info on aggregation
    :type show_agg: bool

    :returns: the dataframe with a the new aggregated column
    :rtype: pd.DataFrame
    """
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )

    prev_idx = df.index
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left', left_index=True)
    df.index = prev_idx
    del(gp)

    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return df

def do_countuniq(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    """Count unique occurences of a column after grouping by other columns.

    :param df: input dataframe.
    :type df: pd.DataFrame
    :param group_cols: the columns we want to group on.
    :type group_cols: list
    :param counted: the column being uniquely counted
    :type counted: str
    :param agg_name: aggregation column name
    :type agg_name: str
    :param agg_type: type of new aggregated column
    :type agg_type: str
    :param show_max: debug option to show max value of aggregated column
    :type show_max: bool
    :param show_agg: debug option to show info on aggregation
    :type show_agg: bool

    :returns: the dataframe with a the new aggregated column
    :rtype: pd.DataFrame
    """
    if show_agg:
        print( "Counting unique ", counted, " by ", group_cols , '...' )

    prev_idx = df.index
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', left_index=True)
    df.index = prev_idx
    del(gp)

    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return df
    
def do_cumcount(df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True):
    """Cumulatively count a column after grouping by other columns.

    :param df: input dataframe.
    :type df: pd.DataFrame
    :param group_cols: the columns we want to group on.
    :type group_cols: list
    :param counted: the column being cumulatively counted
    :type counted: str
    :param agg_name: aggregation column name
    :type agg_name: str
    :param agg_type: type of new aggregated column
    :type agg_type: str
    :param show_max: debug option to show max value of aggregated column
    :type show_max: bool
    :param show_agg: debug option to show info on aggregation
    :type show_agg: bool

    :returns: the dataframe with a the new aggregated column
    :rtype: pd.DataFrame
    """
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )

    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del(gp)

    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return df

def do_mean(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    """Compute mean of a column values after grouping by other columns.

    :param df: input dataframe.
    :type df: pd.DataFrame
    :param group_cols: the columns we want to group on.
    :type group_cols: list
    :param counted: the column for which we want to compute the average of values
    :type counted: str
    :param agg_name: aggregation column name
    :type agg_name: str
    :param agg_type: type of new aggregated column
    :type agg_type: str
    :param show_max: debug option to show max value of aggregated column
    :type show_max: bool
    :param show_agg: debug option to show info on aggregation
    :type show_agg: bool

    :returns: the dataframe with a the new aggregated column
    :rtype: pd.DataFrame
    """
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )

    prev_idx = df.index
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', left_index=True)
    df.index = prev_idx
    del(gp)

    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return df

def do_var(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    """Compute variance of a column values after grouping by other columns.

    :param df: input dataframe.
    :type df: pd.DataFrame
    :param group_cols: the columns we want to group on.
    :type group_cols: list
    :param counted: the column for which we want to compute the variance of values
    :type counted: str
    :param agg_name: aggregation column name
    :type agg_name: str
    :param agg_type: type of new aggregated column
    :type agg_type: str
    :param show_max: debug option to show max value of aggregated column
    :type show_max: bool
    :param show_agg: debug option to show info on aggregation
    :type show_agg: bool

    :returns: the dataframe with a the new aggregated column
    :rtype: pd.DataFrame
    """
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )

    prev_idx = df.index
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', left_index=True)
    df.index = prev_idx
    del(gp)

    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return df

# List of exposed symbols
__all__ = [
    'do_count',
    'do_countuniq',
    'do_cumcount',
    'do_mean',
    'do_var'
]