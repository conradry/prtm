import datetime


def to_date(s: str):
    return datetime.datetime(year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10]))