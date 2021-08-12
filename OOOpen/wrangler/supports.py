from pandas import DataFrame, Series, to_timedelta
from numpy import subtract, logical_and, logical_or, invert

def get_sup_func(x):
    return x.iloc[0] > x.iloc[1] < x.iloc[2]

def get_res_func(x):
    return x.iloc[0] < x.iloc[1] > x.iloc[2]

def get_supports(data, value_col: str = "low", date_col: str = "date", thresh: int = 0.01,
                resistances: bool = False) -> DataFrame:
    """Get all support days in data with the number of days that each support lasted

    Parameters
    ----------
    data : DataFrame
        A DataFrame with the historical data
    value_col : str, optional
        The name of the column to use as a price, by default "low"
    date_col : str, optional
        The name of the column to use as a date for the price, by default "date"
    thresh : int, optional
        How much (in proportion) can a support be breached until its considered
        broken, by default 0.01
    resistances : bool, optional
        Calculate resistances instead

    Returns
    -------
    tuple:
        A tuple with two elements:
        A DataFrame with the info for all the supports, the same fields as for a normal
        day but with a new "age" column specifying for how many days the support
        survived before breaking.
        A boolean DataFrame that indicates which supports where close to a previous one.
    """
    data = data.sort_values(date_col)
    #Get all the days that are supports (lower than the two adjacent days in value_col)
    func = get_res_func if resistances else get_sup_func
    mins = data[value_col] \
        .rolling(3, center=True) \
        .agg(func) \
        .fillna(0) \
        .astype(bool)
    sups = data[mins]
    val_vec = (-1)**resistances*sups[value_col].to_numpy()
    refer_vec = (-1)**resistances*data[value_col].to_numpy()
    difs = subtract.outer(val_vec, refer_vec) / abs(val_vec[:, None])
    #Get matrix with which days before an after the supports
    difdates = subtract.outer(sups[date_col].astype("int").to_numpy(),
        data[date_col].astype("int").to_numpy())
    predates = difdates >= 0
    postdates = invert(predates)
    #Get matrix with which days have a higher value_col than each support
    # days the support survives
    days_higher = difs < thresh
    days_surviving = logical_or(predates, days_higher).cumprod(axis=1)
    days_surviving = logical_and(postdates, days_surviving)
    #Finally get the days in which supports almost got broken but survived
    difs = DataFrame(difs, columns= data.index) \
        .loc[:, sups.index] \
        .to_numpy()
    difdates = DataFrame(difdates, columns= data.index) \
        .loc[:, sups.index] \
        .to_numpy()
    days_alive = DataFrame(days_surviving, columns= data.index) \
        .loc[:, sups.index] \
        .to_numpy()
    days_close = abs(difs) < thresh
    days_close = logical_and(days_close, days_alive)
    ret = sups \
        .assign(
            age = Series(days_surviving.sum(axis=1), index = sups.index),
            near = Series(days_close.sum(axis=1), index = sups.index)) \
        .sort_values("date")
    ret = ret.assign(
        end = ret["date"] + to_timedelta(ret["age"], unit = "D")
    )
    return ret, DataFrame(days_close, index = ret.index, columns = ret.index)

def get_resistances(data, value_col: str = "high", date_col: str = "date", thresh: int = 0.01) -> DataFrame:
    """Get all resitance days in data with the number of days until each resistance
    was broken

    Parameters
    ----------
    data : DataFrame
        A DataFrame with the historical data
    value_col : str, optional
        The name of the column to use as a price, by default "high"
    date_col : str, optional
        The name of the column to use as a date for the price, by default "date"
    thresh : int, optional
        How much (in proportion) can a resistance be supased until its considered
        broken, by default 0.01

    Returns
    -------
    DataFrame
        A DataFrame with the info for all the resistances, the same fields as for a normal
        day but with a new "age" column specifying for how many days the resistance
        survived before breaking.
    """
    return get_supports(data, value_col, date_col, thresh, resistances= True)