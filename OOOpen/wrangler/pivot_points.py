from pandas import DataFrame

def pivot_points(data: DataFrame) -> DataFrame:
    """For each day in data calculate the pivot points and their associated
    supports and resistances

    Parameters
    ----------
    data : DataFrame
        A pandas dataFrame with the data, must be ordered time-like (older to newer)
        and have the columns "close", "high" and "low"

    Returns
    -------
    DataFrame
        A copy of the original dataFrame with 5 new columns
            pivots: values of the pivot points
            r1: first resistance
            r2: second resitance
            s1: first support
            s2: second support
    """
    yesterday = data[["close", "high", "low"]].shift(1)
    pivots = yesterday.sum(axis=1)/3
    r1 = 2*pivots - yesterday["low"]
    s1 = 2*pivots - yesterday["high"]
    r2 = pivots + (r1 - s1)
    s2 = pivots - (r1 - s1)
    return data.assign(pivots = pivots, r1 = r1, r2 = r2, s1 = s1, s2 = s2)