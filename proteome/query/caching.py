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


def cache_query(hash_func_kwargs, hash_class_attrs):
    """Class method decorator for query utilities"""
    def wrapped_function(func):
        def get_cached(*args, **kwargs):
            """
            Only execute the function if its value isn't stored
            in cache already.
            """
            # First of args is self with given hash_class_attrs
            class_attrs = args[0].__dict__
            hash_class_args = [class_attrs[k] for k in hash_class_attrs]
            hash_func_args = []
            for v in args[1:]:
                if isinstance(v, str):
                    hash_func_args.append(v)
                elif v and isinstance(v, list):
                    if all(isinstance(el, str) for el in v):
                        hash_func_args.append(v)
                        
            for func_kw in hash_func_kwargs:
                if func_kw in kwargs:
                    hash_func_args.append(kwargs[func_kw])
                
            hash = hash_args(*(hash_class_args + hash_func_args)).hexdigest()
            cached_result = get_cached_query_result(hash)
            if cached_result is None:
                result = func(*args, **kwargs)
                insert_result_in_cache(hash, result)
                cached_result = result

            return cached_result
        return get_cached
    return wrapped_function