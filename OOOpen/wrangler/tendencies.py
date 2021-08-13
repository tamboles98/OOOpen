def moving_average(data, window: int = 15, value_col: str = "close"):
    """Calculate the moving average of a column in a DataFrame

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame with the data
    window : int, optional
        The width of the window for the moving average, by default 15
    value_col : str, optional
        The name of the column to use to calculate the moving average, by default "close"

    Returns
    -------
    pandas.DataFrame
        An identical DataFrame to the original with a new column named ma{window}
        with the values of the moving average
    """
    return data.assign(**{"ma{}".format(window): data[value_col].rolling(window).mean()})