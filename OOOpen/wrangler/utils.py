from pandas import to_datetime as _to_datetime

def id_to_date(ids):
    return _to_datetime(ids.str.slice(start = -8), format = "%Y%m%d")