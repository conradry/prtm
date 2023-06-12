import os
import sqlite3
import hashlib


DB_NAME = "proteome_queries.db"
TABLE_NAME = "queries"


def _get_db_path():
    """Gets the path to the sqlite databse"""
    db_dir = os.path.join(os.path.expanduser("~"), ".proteome")
    os.makedirs(db_dir, exist_ok=True)
    
    db_path = os.path.join(db_dir, DB_NAME)
    return db_path


def _get_query_db_connect():
    return sqlite3.connect(_get_db_path())


def _make_table():
    cur = _get_query_db_connect().cursor()
    res = cur.execute(f"SELECT name FROM sqlite_master WHERE name='{TABLE_NAME}'")
    if res.fetchone() is None:
        cur.execute(f"CREATE TABLE {TABLE_NAME}(query_hash, result)")

def hash_args(*args):
    """Hashes a list of arguments, the arguments must be string 
    type or convertable to strings.
    """
    hash = hashlib.sha512()
    for arg in args:
        hash.update(str(arg).encode("utf-8"))

    return hash


def get_cached_query_result(hash):
    _make_table()
    
    # Check if hash is in the table
    cur = _get_query_db_connect().cursor()
    cached_result = cur.execute(
        f"SELECT result FROM {TABLE_NAME} WHERE query_hash='{hash}'"
    ).fetchone()
    if cached_result is not None:
        return eval(cached_result[0])
    else:
        return cached_result


def insert_result_in_cache(hash, result):
    _make_table()
    con = _get_query_db_connect()
    cur = con.cursor()
    cur.execute(f"INSERT INTO {TABLE_NAME} VALUES(?, ?)", (hash, str(result)))
    con.commit()