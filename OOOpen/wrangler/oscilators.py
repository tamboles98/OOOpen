def rsi(data, nd: int = 14):
    """Add a new column to data with the rsi oscilator

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame with stock information, in need to have columns with open
        a close values for each day and be ordered from older to newer values
    nd : int, optional
        Time frame (number of days) to use when computing the rsi, by default 15

    Returns
    -------
    pandas.DataFrame
        The original DataFrame (data) with the new rsi column
    """
    dif = data[open_col] - data[close_col]
    w = dif.clip(lower=0).rolling(nd).sum()
    l = -dif.clip(upper=0).rolling(nd).sum()
    return data.assign(**{"rsi{}".format(nd): 100 - 100/(1 + w/l)})