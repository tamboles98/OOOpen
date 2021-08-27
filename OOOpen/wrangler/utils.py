from pandas import DataFrame, to_datetime, concat

def id_to_date(ids):
    return to_datetime(ids.str.slice(start = -8), format = "%Y%m%d")

def apply_to_multiple_symbols(data: DataFrame, function, **kwargs):
    return data.groupby("symbol").apply(function, **kwargs).droplevel("symbol")

def shift_data(data: DataFrame, previous: int) -> DataFrame:
    cols = ["open", "high", "low", "close", "volume"]
    shifted = data[cols].shift(previous)
    renames = dict(zip(cols, (cn + "-{}".format(previous) for cn in cols)))
    shifted = shifted.rename(columns= renames)
    return shifted


def add_previous_days(data: DataFrame, previous: int = 4) -> DataFrame:
    shifted = [shift_data(data, i) for i in range(1, previous + 1)]
    con = [data]
    con.extend(shifted)
    return concat(con, axis=1)

def normalize_volume(data: DataFrame, window: int = 64):
    mean_volume = data["volume"].rolling(64, center = False).mean()
    return data.assign(volume = (data["volume"]-mean_volume)/mean_volume)

        