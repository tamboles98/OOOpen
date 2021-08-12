from pandas import DataFrame

def rfi(data: DataFrame, nd: int = 15, open_col: str = "open", close_col: str = "close") -> DataFrame:
    """Add a new column to data with the rfi oscilator

    Parameters
    ----------
    data : DataFrame
        A pandas DataFrame with stock information, in need to have columns with open
        a close values for each day and be ordered from older to newer values
    nd : int, optional
        Time frame (number of days) to use when computing the rfi, by default 15
    open_col : str, optional
        Name of the column to use as open values, by default "open"
    close_col : str, optional
        Name of the column to use as close values, by default "close"

    Returns
    -------
    DataFrame
        The original DataFrame (data) with the new rfi column
    """
    dif = data[open_col] - data[close_col]
    w = dif.clip(lower=0).rolling(nd).sum()
    l = -dif.clip(upper=0).rolling(nd).sum()
    return data.assign(**{"rfi{}".format(nd): 100 - 100/(1 + w/l)})