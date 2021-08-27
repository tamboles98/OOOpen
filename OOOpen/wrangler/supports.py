from numpy.lib.type_check import real_if_close
from pandas import DataFrame, Series, to_timedelta, read_pickle
from numpy import subtract, logical_and, logical_or, invert, abs
from pandas.core.reshape.concat import concat
from .utils import id_to_date
import pathlib

def _get_sup_func(x):
    return x.iloc[0] > x.iloc[1] < x.iloc[2]

def _get_res_func(x):
    return x.iloc[0] < x.iloc[1] > x.iloc[2]

def _get_days(x: DataFrame, value_col: str, resistances: bool = False) -> DataFrame:
    """Filter stock data leaving only days that were supports (or resistances)

    Parameters
    ----------
    x : DataFrame
        Data
    value_col : str
        column to use as price for calculations
    resistances : bool, optional
        If true filter resistances instead or supports, by default False

    Returns
    -------
    DataFrame
        The filtered data
    """
    func = _get_res_func if resistances else _get_sup_func
    mins = x[value_col] \
        .rolling(3, center=True) \
        .agg(func) \
        .fillna(0) \
        .astype(bool)
    return x[mins]

def _get_compatible_dates(x1: DataFrame, x2: DataFrame, date_col: str = "date") -> DataFrame:
    """Return a boolean DataFrame with compatible days, in ceil (x, y) True means
    that day y is later than day x

    Parameters
    ----------
    x1 : DataFrame
        The data for the days to use as rows (x days in the above example)
    x2 : DataFrame
        The data for the days to use as colums (y days in the above example)
    date_col : str
        The column with the dates in each DataFrame

    Returns
    -------
    DataFrame
        A boolean DataFrame with index = x1.index and columns = x2.index
    """
    dates1, dates2 = x1[date_col].to_numpy("int"), x2[date_col].to_numpy("int")
    difdates = subtract.outer(dates1, dates2)
    return DataFrame(difdates < 0, index = x1.index, columns= x2.index)

def surviving_days (sups: DataFrame, data: DataFrame, thresh: float = 0.01,
    date_col: str = "date", ref_col: str = "low", compare_col: str = "close",
    resistances = False) -> tuple:
    compa = _get_compatible_dates(sups, data, date_col = date_col)
    ref_val = sups[ref_col].to_numpy()
    comp_val = data[compare_col].to_numpy()
    difval = subtract.outer(ref_val, comp_val)
    nordifval = difval / ref_val[:, None]
    booldif = ((-1)**resistances)*nordifval < thresh
    alive = logical_and(compa, booldif)
    alive = logical_or(invert(compa), alive).cumprod(axis = 1).astype(bool)
    alive = logical_and(compa, alive)
    breakers = logical_and(invert(alive), compa)
    breakers = breakers[breakers.any(axis = 1)].idxmax(axis = 1)
    return alive, breakers


def close_days(data: DataFrame, compare: DataFrame, surviving_days: DataFrame, thresh: float = 0.01,
    reference_col: str = "low", compare_col: str = "close", date_col: str = "date",
    resistances: bool = False) -> DataFrame:
    """Return a matrix with which days that were close to be breached but finally
    survived

    Parameters
    ----------
    data : DataFrame
        The supports/resistances you want to check
    compare : DataFrame
        The days to compare the supports/resistances with
    surviving_days : DataFrame
        Days that each support/resistance was alive. The index must match data's
        index
    thresh : float, optional
        Closeness (in euclidean distance) to consider that a support/resistance
        was almost breached, by default 0.01
    reference_col : str, optional
        Column to use as values for the supports/resistances, by default "low"
    compare_col : str, optional
        column to use as values for the days in compare, by default "close"
    date_col : str, optional
        Column to use for date info, by default "date"
    resistances : bool, optional
        Wheter you are working with supports(false) or resistances(true), by default False

    Returns
    -------
    DataFrame:
        The index are the supports/resistances, the columns the days you are compareing
        against (but only those that are maximums/minimums depending if you are working
        with resitances/supports). Each cell [x, y] is a boolean that says whether or not
        day y was a close day for the support/resistance x

    Raises
    ------
    ValueError
        If surviving_days.index is not equal to data.index
    """
    if not data.index.equals(surviving_days.index):
        raise ValueError("surviving_days index doesn't match data.index")
    pbreachers = _get_days(compare, compare_col, resistances)[[date_col, compare_col]]
    dates = _get_compatible_dates(data, pbreachers, date_col = date_col)
    ref_val = data[reference_col].to_numpy()
    comp_val = compare[compare_col].to_numpy()
    valdif = abs(subtract.outer(ref_val, comp_val))
    valdif = valdif/ref_val[:, None]
    valdif = DataFrame(valdif, index=data.index, columns=compare.index)
    return surviving_days[compare.index] & (valdif < thresh)


def get_supports_resistances(data, thresh: int = 0.01, date_col: str = "date") -> DataFrame:
    """Get all the information about supports and resistances in data

    When broken supports become resistances and viceversa

    Parameters
    ----------
    data : DataFrame
        A DataFrame with the historical data
    date_col : str, optional
        The name of the column to use as a date for the price, by default "date"
    thresh : int, optional
        How much (in proportion) can a support be breached until its considered
        broken, by default 0.01

    Returns
    -------
    tuple:
        A tuple with 6 elements
            * DataFrame with general information about supports
            * Matrix of days that supports were active
            * Matrix of close days for supports (days that the support was almost
            broken but the finally ended bouncing back)
            * DataFrame with general information about resistances
            * Matrix of days that resistances were active
            * Matrix of close days for supports
    """
    data = data.sort_values(date_col)
    sup = _get_days(data, value_col= "low")[["symbol", "date", "low", "volume"]]
    res = _get_days(data, value_col= "high", resistances= True)[["symbol", "date", "high", "volume"]]
    #Calculate the days the support and resistances survived and the id of the day
    # that finally breached them
    sdays, sbreakers = surviving_days(sup, data, thresh, date_col)
    rdays, rbreakers = surviving_days(res, data, thresh, date_col, "high", resistances= True)
    #Enrich the supports/resistances with the above information
    sup = sup.assign(age = sdays.sum(axis = 1), enday = id_to_date(sbreakers),
        ended = sbreakers.notna(), original = True)
    res = res.assign(age = rdays.sum(axis = 1), enday = id_to_date(rbreakers),
        ended = rbreakers.notna(), original = True)
    #Days that broke a resistance become support and vice versa
    #Select the days that break any support or ressintance
    auxsbreakers, auxrbreakers = sbreakers.dropna(), rbreakers.dropna()
    #A day can break multiple supports or resistances in that case we focus only in
    # the lowest support (or higher resistance)
    #Some breakers can be already supports or resistances themselves, in that case
    # we ignore them
    auxsbreakers = sup \
        .assign(breakers = sbreakers) \
        .groupby("breakers")["low"] \
        .agg("idxmin") \
        .drop(res.index, errors = "ignore")
    #The days that break support will become resistances, we drop those days that
    # are already resistances
    auxrbreakers = res \
        .assign(breakers = rbreakers) \
        .groupby("breakers")["high"] \
        .agg("idxmax") \
        .drop(sup.index, errors = "ignore") #Drop days that are already supports
    #Get the data for those breakers
    second_resistances = data.loc[auxsbreakers.index, ["symbol", "date", "volume"]]
    auxsbreakers = Series(auxsbreakers.index, index = auxsbreakers.values)
    second_resistances = second_resistances \
        .assign(high = sup.loc[auxsbreakers.index, "low"].rename(index = auxsbreakers))
    second_supports = data.loc[auxrbreakers.index, ["symbol", "date", "volume"]]
    auxrbreakers = Series(auxrbreakers.index, index = auxrbreakers.values)
    second_supports = second_supports \
        .assign(low = res.loc[auxrbreakers.index, "high"].rename(index = auxrbreakers))
    del(auxsbreakers, auxrbreakers)
    #Enrich those breakers
    sdays2, sbreakers2 = surviving_days(second_supports, data, thresh, date_col)
    rdays2, rbreakers2 = surviving_days(second_resistances, data, thresh, date_col, "high", resistances=True)
    second_supports = second_supports \
        .assign(age = sdays2.sum(axis = 1), enday = id_to_date(sbreakers2),
            ended = sbreakers2.notna(), original = False)
    second_resistances = second_resistances.assign(age = rdays2.sum(axis = 1), enday = id_to_date(rbreakers2),
        ended = rbreakers2.notna(), original = False)

    #Start preparing the final data
    sup = sup.append(second_supports).sort_values("date")
    sdays = sdays.append(sdays2).sort_index()
    res = res.append(second_resistances).sort_values("date")
    rdays = rdays.append(rdays2).sort_index()


    #Finally compute days that the supports/resistances almost got breached but
    # survived
    sclose = close_days(sup, data, sdays, thresh=thresh, reference_col="low",
        compare_col= "close", date_col="date", resistances=False)
    rclose = close_days(res, data, rdays, thresh=thresh, reference_col="high",
        compare_col="close", date_col="date", resistances=True)
    
    sup = sup.assign(close = sclose.sum(axis = 1))
    res = res.assign(close = rclose.sum(axis = 1))

    sclose.columns = id_to_date(sclose.columns)
    rclose.columns = id_to_date(rclose.columns)

    return (sup.sort_values("date"), sdays, sclose, res.sort_values("date"), rdays, rclose)


def get_supports(data, reference_col: str = "low", compare_col: str = "close",
                date_col: str = "date", thresh: int = 0.01, resistances: bool = False) -> DataFrame:
    """Get all support days in data with the number of days that each support lasted

    Parameters
    ----------
    data : DataFrame
        A DataFrame with the historical data
    reference_col : str, optional
        The name of the column to use as a price to determine when a support
        is created, by default "low"
    compare_col : str, optional
        The name of the column to use as a price to determine when a support is
        broken, by default "close
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
    sups = _get_days(data, value_col=reference_col, resistances=resistances)
    val_vec = (-1)**resistances*sups[reference_col].to_numpy()
    refer_vec = (-1)**resistances*data[compare_col].to_numpy()
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

def get_resistances(data, reference_col: str = "high", compare_col: str = "close",
                    date_col: str = "date", thresh: int = 0.01) -> DataFrame:
    """Get all resitance days in data with the number of days until each resistance
    was broken

    Parameters
    ----------
    data : DataFrame
        A DataFrame with the historical data
    reference_col : str, optional
        The name of the column to use as a price to determine when a resistance
        is created, by default "high"
    compare_col : str, optional
        The name of the column to use as a price to determine when a support is
        broken, by default "close"
    date_col : str, optional
        The name of the column to use as a date for the price, by default "date"
    thresh : int, optional
        How much (in proportion) can a resistance be surpased until its considered
        broken, by default 0.01

    Returns
    -------
    DataFrame
        A DataFrame with the info for all the resistances, the same fields as for a normal
        day but with a new "age" column specifying for how many days the resistance
        survived before breaking.
    """
    return get_supports(data, reference_col, compare_col, date_col, thresh, resistances= True)


# ---------------------------------------

def get_closest_support(x: Series, info: DataFrame, comp: DataFrame, close: DataFrame) -> dict:
    candidates = info[comp[x.name]]
    if candidates.empty:
        return {"sup_val": 0, "sup_vol": -1, "sup_age": 0, "sup_close": 0}
    else:
        id_ = ((candidates["low"] - x["close"])/x["close"]).abs().idxmin()
        age = comp.loc[id_, :x.name].shape[0]
        ncl = close.loc[id_, :x["date"]].sum()
        return {"sup_val": info.loc[id_, "low"], "sup_vol": info.loc[id_, "volume"],
            "sup_age": age, "sup_close": ncl}

def add_supports(data: DataFrame, info: DataFrame, comp: DataFrame, close: DataFrame) -> DataFrame:
    """Add columns regarding the closest supports to each day in a dataFrame with
    historical data about a stock

    Parameters
    ----------
    data : DataFrame
        The historical data of the stock
    info : DataFrame
        A dataFrame with general info about supports (see the returned values of
        the get_supports_resistances method in this same module)
    comp : DataFrame
        A boolean matrix that tells which supports were active each day (see the returned values of
        the get_supports_resistances method in this same module)
    close : DataFrame
        A boolean matrix with the close days for each support (see the returned values of
        the get_supports_resistances method in this same module)

    Returns
    -------
    DataFrame
        The original data dataFrame with new columns with the supports info:
         * sup_val: the price of the support
         * sup_vol: the volume of transactions the day of the support
         * sup_age: the age of the support the relevant day
         * sup_close: the number of close days the support had experienced before
            the relevant day (days that the support was almost breached but ultimately
            the price bounced back)
    """
    new_cols = data \
        .apply(get_closest_support, axis = 1, info = info, comp = comp, close = close) \
        .apply(Series)
    return concat([data, new_cols], axis = 1)


def get_closest_resitances(x: Series, info: DataFrame, comp: DataFrame, close: DataFrame) -> dict:
    candidates = info[comp[x.name]]
    if candidates.empty:
        return {"res_val": 2*x["close"], "res_vol": -1, "res_age": 0, "res_close": 0}
    else:
        id_ = ((candidates["high"] - x["close"])/x["close"]).abs().idxmin()
        age = comp.loc[id_, :x.name].shape[0]
        ncl = close.loc[id_, :x["date"]].sum()
        return {"res_val": info.loc[id_, "high"], "res_vol": info.loc[id_, "volume"],
            "res_age": age, "res_close": ncl}

def add_resistances(data: DataFrame, info: DataFrame, comp: DataFrame, close: DataFrame) -> DataFrame:
    """Add columns regarding the closest resistances to each day in a dataFrame with
    historical data about a stock

    Parameters
    ----------
    data : DataFrame
        The historical data of the stock
    info : DataFrame
        A dataFrame with general info about resistances (see the returned values of
        the get_resistances_resistances method in this same module)
    comp : DataFrame
        A boolean matrix that tells which resistances were active each day (see the returned values of
        the get_resistances_resistances method in this same module)
    close : DataFrame
        A boolean matrix with the close days for each resistance (see the returned values of
        the get_resistances_resistances method in this same module)

    Returns
    -------
    DataFrame
        The original data dataFrame with new columns with the resistances info:
         * sup_val: the price of the resistance
         * sup_vol: the volume of transactions the day of the resistance
         * sup_age: the age of the resistance the relevant day
         * sup_close: the number of close days the resistance had experienced before
            the relevant day (days that the resistance was almost breached but ultimately
            the price bounced back)
    """
    new_cols = data \
        .apply(get_closest_resitances, axis = 1, info = info, comp = comp, close = close) \
        .apply(Series)
    return concat([data, new_cols], axis = 1)


def process_supports_resistances(data: DataFrame, artifacts_dir: str,
    thresh: float = 0.01) -> DataFrame:
    needed = ["sup_info.pickle", "sup_days.pickle", "sup_close.pickle",
            "res_info.pickle", "res_days.pickle", "res_close.pickle"]
    symb = data.iloc[0]["symbol"]
    artidir = pathlib.Path(artifacts_dir) / symb
    needed = [artidir / n for n in needed]
    if all(f.is_file() for f in needed):
        supinfo, supdays, supclose, resinfo, resdays, resclose = tuple(
            read_pickle(p) for p in needed)
    else:
        elements = get_supports_resistances(
            data = data, thresh=thresh, date_col = "date"
        )
        artidir.mkdir(parents=True, exist_ok=True)
        for n, e in zip(needed, elements): e.to_pickle(n)
        supinfo, supdays, supclose, resinfo, resdays, resclose = elements
    data = add_resistances(data, info=resinfo, comp=resdays, close=resclose)
    return add_supports(data, info=supinfo, comp=supdays, close=supclose)

    