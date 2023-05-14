import sys
import os
import re
import gc
import importlib
from collections import defaultdict
from functools import reduce
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd


### Logging ###

def log(*message, **kwargs):
    
    global _log_file
    
    end = kwargs.get('end', '\n')
    
    if len(message) == 1:
        message, = message
    
    full_message = '[%s] %s' % (format_now(), message)
    
    print(full_message, end = end)
    sys.stdout.flush()
    
    if log_file_open():
        _log_file.write(full_message + end)
        _log_file.flush()

def start_log(log_dir, log_file_base_name):
    
    global _log_file
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file_name = '%s__%d__%s.txt' % (log_file_base_name, os.getpid(), format_now())
    
    if not log_file_open():
        print('Creating log file: %s' % log_file_name)
        _log_file = open(os.path.join(log_dir, log_file_name), 'w')
        
def close_log():
    
    global _log_file
    
    if log_file_open():
        _log_file.close()
        del _log_file
    
def restart_log():
    close_log()
    start_log()
    
def log_file_open():
    global _log_file
    return '_log_file' in globals()
    
def create_time_measure_if_verbose(opening_statement, verbose):
    if verbose:
        return TimeMeasure(opening_statement)
    else:
        return DummyContext()


### General ###

def get_nullable(value, default_value):
    if pd.isnull(value):
        return default_value
    else:
        return value
        
        
### Reflection ###

def load_object(full_object_name):
    name_parts = full_object_name.split('.')
    object_name = name_parts[-1]
    module_name = '.'.join(name_parts[:-1])
    module = importlib.import_module(module_name)
    return getattr(module, object_name)
    
    
### Strings ###

def trim(string, max_length, trim_suffix = '...'):
    if len(string) <= max_length:
        return string
    else:
        return string[:(max_length - len(trim_suffix))] + trim_suffix
        
def break_to_lines(text, max_line_len):
    
    lines = ['']
    
    for word in text.split():
        
        if len(lines[-1]) + len(word) > max_line_len:
            lines.append('')
            
        if lines[-1] != '':
            lines[-1] += ' '
            
        lines[-1] += word
        
    return '\n'.join(lines)
    
    
### IO ###

def safe_symlink(src, dst, post_creation_hook = lambda created_symlink: None):
    if os.path.exists(dst):
        log('%s: already exists.' % dst)
    else:
        try:
            os.symlink(src, dst)
            post_creation_hook(dst)
            log('Created link: %s -> %s' % (src, dst))
        except OSError as e:
            if e.errno == 17:
                log('%s: already exists after all.' % dst)
            else:
                raise e
        
def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as e:
        assert 'File exists' in str(e), str(e)
        
def format_size_in_bytes(size):
    
    UNIT_RATIO = 1024
    UNITS = ['B', 'KB', 'MB', 'GB', 'TB']
    
    for unit_index in range(len(UNITS)):
        if size < UNIT_RATIO:
            break
        else:
            size /= UNIT_RATIO
            
    return '%.1f%s' % (size, UNITS[unit_index])
    
def get_recognized_files_in_dir(dir_path, file_parser, log_unrecognized_files = True):
    
    recognized_files = []
    unrecognized_files = []
    
    for file_name in os.listdir(dir_path):
        try:
            recognized_files.append((file_parser(file_name), file_name))
        except:
            if log_unrecognized_files:
                unrecognized_files.append(file_name)
                
    if log_unrecognized_files and len(unrecognized_files) > 0:
        log('%s: %d unrecognized files: %s' % (dir_path, len(unrecognized_files), ', '.join(unrecognized_files)))
        
    return list(sorted(recognized_files))
    
def monitor_memory(min_bytes_to_log = 1e08, max_elements_to_check = 100, collect_gc = True, del_output_variables = True, \
        list_like_types = [list, tuple, np.ndarray, pd.Series], dict_like_types = [dict, defaultdict]):
    
    already_monitored_object_ids = set()
    
    def _is_of_any_type(obj, types):
        
        for t in types:
            if isinstance(obj, t):
                return True
            
        return False
        
    def _check_len_limit(obj):
        try:
            return len(obj) <= max_elements_to_check
        except:
            return False

    def _log_object_if_needed(name, obj):

        size = sys.getsizeof(obj)

        if size >= min_bytes_to_log:
            log('%s: %s' % (name, format_size_in_bytes(size)))
            
    def _monitor_object(name, obj):
        if id(obj) not in already_monitored_object_ids:
            
            already_monitored_object_ids.add(id(obj))
            _log_object_if_needed(name, obj)
                        
            if _is_of_any_type(obj, list_like_types) and _check_len_limit(obj):
                for i, element in enumerate(obj):
                    _monitor_object('%s[%d]' % (name, i), element)

            if _is_of_any_type(obj, dict_like_types) and _check_len_limit(obj):
                for key, value in obj.items():
                    _monitor_object('%s[%s]' % (name, repr(key)), value)
            
            
    for module_name, module in sys.modules.items():
        for variable_name in dir(module):
            
            full_variable_name = variable_name if module_name == '__main__' else '%s.%s' % (module_name, variable_name)
            _monitor_object(full_variable_name, getattr(module, variable_name))

            if del_output_variables and module_name == '__main__' and re.match(r'^_[\d_]+$', variable_name):
                delattr(module, variable_name)
                
    if del_output_variables:
        sys.modules['__main__'].Out = dict()
        sys.modules['__main__']._oh = dict()

    if collect_gc:
        gc.collect()


### Date & time ###

def format_now():
    return datetime.now().strftime('%Y_%m_%d-%H:%M:%S')
    

### Iterators & collections ###

def compare_list_against_collection(input_list, collection):
    collection_set = set(collection)
    return [element for element in input_list if element in collection_set], [element for element in input_list if element not in collection_set]

def get_chunk_slice(size, n_chunks, chunk_index):
    assert size >= n_chunks
    chunk_size = size / n_chunks
    start_index = int(chunk_index * chunk_size)
    end_index = int((chunk_index + 1) * chunk_size)
    return start_index, end_index

def get_chunk_intervals(size, chunk_size):
    for start_index in range(0, size, chunk_size):
        end_index = min(start_index + chunk_size, size)
        yield start_index, end_index
        
def to_chunks(iterable, chunk_size):
    
    chunk = []
    
    for element in iterable:
        
        chunk.append(element)
        
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
            
    if len(chunk) > 0:
        yield chunk
        
def get_job_and_subjob_indices(n_jobs, n_tasks, task_index):
    
    '''
    For example, if there are 170 tasks for working on 50 jobs, than each job will be divided to 3-4 tasks.
    Since 170 % 50 = 20, the 20 first jobs will receive 4 tasks and the last 30 jobs will receive only 3 tasks.
    In total, the first 80 tasks will be dedicated to jobs with 4 tasks each, and the 90 last tasks will be
    dedicated to jobs with 3 tasks each. Hence, tasks 0-3 will go to job 0, tasks 4-7 will go to job 1, and so on;
    tasks 80-82 will go to job 21, tasks 83-85 will job to job 22, and so on.  
    '''
    
    assert n_tasks >= n_jobs
    n_tasks_in_unprivileged_jobs = n_tasks // n_jobs
    n_tasks_in_privileged_jobs = n_tasks_in_unprivileged_jobs + 1
    n_privileged_jobs = n_tasks % n_jobs
    n_tasks_of_privileged_jobs = n_tasks_in_privileged_jobs * n_privileged_jobs
    
    if task_index < n_tasks_of_privileged_jobs:
        job_index = task_index // n_tasks_in_privileged_jobs
        index_within_job = task_index % n_tasks_in_privileged_jobs
        n_tasks_in_job = n_tasks_in_privileged_jobs
    else:
        task_index_in_unprivileged_group = task_index - n_tasks_of_privileged_jobs
        job_index = n_privileged_jobs + task_index_in_unprivileged_group // n_tasks_in_unprivileged_jobs
        index_within_job = task_index_in_unprivileged_group % n_tasks_in_unprivileged_jobs
        n_tasks_in_job = n_tasks_in_unprivileged_jobs
        
    return job_index, index_within_job, n_tasks_in_job
    
def choose_from_cartesian_product(list_of_values, i, total = None):
    
    n = int(np.prod(list(map(len, list_of_values))))
    
    if total is not None:
        assert n == total
    
    chosen_elements = []
    
    for values in list_of_values:
        n //= len(values)
        chosen_elements.append(values[i // n])
        i %= n

    return chosen_elements

def calc_overlap_between_segments(ordered_segments1, ordered_segments2):
    
    '''
    Calculates the total overlap size between a pair of ordered and disjoint groups of segments.
    Each group of segment is given by: [(start1, end1), (start2, end2), ...]. 
    '''
    
    from interval_tree import IntervalTree
    
    if len(ordered_segments1) == 0 or len(ordered_segments2) == 0:
        return 0
    
    if len(ordered_segments1) > len(ordered_segments2):
        ordered_segments1, ordered_segments2 = ordered_segments2, ordered_segments1
    
    min_value = min(ordered_segments1[0][0], ordered_segments2[0][0])
    max_value = max(ordered_segments1[-1][1], ordered_segments2[-1][1])
    interval_tree1 = IntervalTree([segment + (segment,) for segment in ordered_segments1], min_value, max_value)
    total_overlap = 0
    
    for segment in ordered_segments2:
        for overlapping_segment in interval_tree1.find_range(segment):
            overlapping_start = max(segment[0], overlapping_segment[0])
            overlapping_end = min(segment[1], overlapping_segment[1])
            assert overlapping_start <= overlapping_end, 'Reported overlap between %d..%d to %d..%d.' % (segment + \
                    overlapping_segment)
            total_overlap += (overlapping_end - overlapping_start + 1)
            
    return total_overlap
    
def merge_lists_with_compatible_relative_order(lists):
    
    '''
    Given a list of lists with compatible relative ordering (i.e. for every two sublists, the subset of elements that exist in the two
    sublists will have the same relative order), returns a merging of these sublists into a single grand list that contains all the
    elements (each element only once), and preserves the same ordering.
    '''
    
    def merge_two_sublists(list1, list2):
        
        value_to_index = {value: float(i) for i, value in enumerate(list1)}
        unique_list2_index = {}
        last_identified_index = len(list1)
        
        for i, value in list(enumerate(list2))[::-1]:
            if value in value_to_index:
                last_identified_index = value_to_index[value]
            else:
                unique_list2_index[value] = last_identified_index - 1 + i / len(list2)
                
        value_to_index.update(unique_list2_index)
        return sorted(value_to_index.keys(), key = value_to_index.get)
    
    return reduce(merge_two_sublists, lists, [])
    
    
### argparse ###

def get_parser_bool_type(parser):

    def _bool_type(value):
        if isinstance(value, bool):
           return value
        if value.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif value.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise parser.error('"%s": unrecognized boolean value.' % value)
            
    return _bool_type

def get_parser_file_type(parser, must_exist = False):

    def _file_type(path):
    
        path = os.path.expanduser(path)
    
        if must_exist:
            if not os.path.exists(path):
                parser.error('File doesn\'t exist: %s' % path)
            elif not os.path.isfile(path):
                parser.error('Not a file: %s' % path)
            else:
                return path
        else:
        
            dir_path = os.path.dirname(path)
        
            if dir_path and not os.path.exists(dir_path):
                parser.error('Parent directory doesn\'t exist: %s' % dir_path)
            else:
                return path
    
    return _file_type

def get_parser_directory_type(parser, create_if_not_exists = False):
    
    def _directory_type(path):
    
        path = os.path.expanduser(path)
    
        if not os.path.exists(path):
            if create_if_not_exists:
            
                parent_path = os.path.dirname(path)
            
                if parent_path and not os.path.exists(parent_path):
                    parser.error('Cannot create empty directory (parent directory doesn\'t exist): %s' % path)
                else:
                    os.mkdir(path)
                    return path
            else:
                parser.error('Path doesn\'t exist: %s' % path)
        elif not os.path.isdir(path):
            parser.error('Not a directory: %s' % path)
        else:
            return path
        
    return _directory_type
    
def add_parser_task_arguments(parser):
    parser.add_argument('--task-index', dest = 'task_index', metavar = '<0,...,N_TASKS-1>', type = int, default = None, help = 'If you want to ' + \
            ' distribute this process across multiple computation resources (e.g. on a cluster) you can specify the total number of tasks ' + \
            '(--total-tasks) to split it into, and the index of the current task to run (--task-index).')
    parser.add_argument('--total-tasks', dest = 'total_tasks', metavar = '<N_TASKS>', type = int, default = None, help = 'See --task-index.')
    parser.add_argument('--task-index-env-variable', dest = 'task_index_env_variable', metavar = '<e.g. SLURM_ARRAY_TASK_ID>', type = str, default = None, \
            help = 'Instead of specifying a hardcoded --task-index, you can specify an environtment variable to take it from (e.g. SLURM_ARRAY_TASK_ID ' + \
            'if you use SLURM to distribute the jobs).')
    parser.add_argument('--total-tasks-env-variable', dest = 'total_tasks_env_variable', metavar = '<e.g. SLURM_ARRAY_TASK_COUNT>', type = str, \
            default = None, help = 'Instead of specifying a hardcoded --total-tasks, you can specify an environtment variable to take it from (e.g. ' + \
            'SLURM_ARRAY_TASK_COUNT if you use SLURM to distribute the jobs).')
            
def determine_parser_task_details(args):
    
    if args.task_index is not None and args.task_index_env_variable is not None:
        parser.error('You must choose between --task-index and --task-index-env-variable.')
    
    if args.task_index is not None:
        task_index = args.task_index
    elif args.task_index_env_variable is not None:
        task_index = int(os.getenv(args.task_index_env_variable))
    else:
        task_index = None
        
    if args.total_tasks is not None and args.total_tasks_env_variable is not None:
        parser.error('You must choose between --total-tasks and --total-tasks-env-variable.')
        
    if args.total_tasks is not None:
        total_tasks = args.total_tasks
    elif args.total_tasks_env_variable is not None:
        total_tasks = int(os.getenv(args.total_tasks_env_variable))
    else:
        total_tasks = None

    if task_index is None and total_tasks is None:
        task_index = 0
        total_tasks = 1
    elif task_index is None or total_tasks is None:
        parser.error('Task index and total tasks must either be specified or unspecified together.')
    
    if task_index < 0 or task_index >= total_tasks:
        parser.error('Task index must be in the range 0,...,(total tasks)-1.')
    
    return task_index, total_tasks

    
### Numpy ###

def normalize(x):

    if isinstance(x, list):
        x = np.array(x)

    u = np.mean(x)
    sigma = np.std(x)
    
    if sigma == 0:
        return np.ones_like(x)
    else:
        return (x - u) / sigma
    
def random_mask(size, n_trues):
    assert n_trues <= size
    mask = np.full(size, False)
    mask[:n_trues] = True
    np.random.shuffle(mask)
    return mask
    
def indices_to_masks(n, indices):
    positive_mask = np.zeros(n, dtype = bool)
    positive_mask[indices] = True
    negative_mask = np.ones(n, dtype = bool)
    negative_mask[indices] = False
    return positive_mask, negative_mask
    
def as_hot_encoding(values, value_to_index, n_values = None):

    if n_values is None:
        n_values = len(value_to_index)
        
    result = np.zeros(n_values)
    
    try:
        values = iter(values)
    except TypeError:
        values = iter([values])
        
    for value in values:
        result[value_to_index[value]] += 1
        
def is_full_rank(matrix):
    return np.linalg.matrix_rank(matrix) == min(matrix.shape)
    
def find_linearly_independent_columns(matrix):
    
    '''
    The calculation is fasciliated by the Gram Schmidt process, everytime taking the next column and removing its projections
    from all next columns, getting rid of columns which end up zero.
    '''
    
    n_rows, n_cols = matrix.shape
    
    if np.linalg.matrix_rank(matrix) == n_cols:
        return np.arange(n_cols)
    
    orthogonalized_matrix = matrix.copy().astype(float)
    independent_columns = []
    
    for i in range(n_cols):
        if not np.isclose(orthogonalized_matrix[:, i], 0).all():
            
            independent_columns.append(i)
            
            if len(independent_columns) >= n_rows:
                break
            
            orthogonalized_matrix[:, i] = orthogonalized_matrix[:, i] / np.linalg.norm(orthogonalized_matrix[:, i])
            
            if i < n_cols - 1:
                # Remove the projection of the ith column from all next columns
                orthogonalized_matrix[:, (i + 1):] -= np.dot(orthogonalized_matrix[:, i], \
                        orthogonalized_matrix[:, (i + 1):]).reshape(1, -1) * orthogonalized_matrix[:, i].reshape(-1, 1)
            
    return np.array(independent_columns)

def transpose_dataset(src, dst, max_memory_bytes, flush_func = None):
    
    n_rows, n_cols = src.shape[:2]
    entry_nbytes = src[:1, :1].nbytes
    ideal_entries_per_chunk = max_memory_bytes / entry_nbytes
    ideal_chunk_size = np.sqrt(ideal_entries_per_chunk)
    
    if n_rows <= n_cols:
        row_chunk_size = min(int(ideal_chunk_size), n_rows)
        col_chunk_size = min(int(ideal_entries_per_chunk / row_chunk_size), n_cols)
    else:
        col_chunk_size = min(int(ideal_chunk_size), n_cols)
        row_chunk_size = min(int(ideal_entries_per_chunk / col_chunk_size), n_rows)
        
    log('Will use chunks of size %dx%d to transpose a %dx%d matrix.' % (row_chunk_size, col_chunk_size, n_rows, n_cols))
    
    for row_start, row_end in get_chunk_intervals(n_rows, row_chunk_size):
        for col_start, col_end in get_chunk_intervals(n_cols, col_chunk_size):
            
            log('Transposing chunk (%d..%d)x(%d..%d)...' % (row_start, row_end - 1, col_start, col_end - 1))
            dst[col_start:col_end, row_start:row_end] = src[row_start:row_end, col_start:col_end].transpose()
            
            if flush_func is not None:
                flush_func()
                
    log('Finished transposing.')


### Pandas ###

def summarize(df, n = 5, sample = False):
    
    from IPython.display import display
    
    if sample:
        display(df.sample(n))
    else:
        display(df.head(n))
    
    print('%d records' % len(df))
    
def nullable_idxmin(series):
    
    result = series.idxmin()
    
    if pd.isnull(result):
        if len(series) == 0:
            return np.nan
        else:
            return series.index[0]
    else:
        return result
    
def get_first_value(df):
    '''
    Will return a Series with the same index. For each row the value will be that of the first column which is not null.
    '''
    col_idxs = np.argmax(pd.notnull(df).values, axis = 1)
    return pd.Series(df.values[np.arange(len(df)), col_idxs], index = df.index)
    
def slice_not_in_index(df_or_series, index_to_exclude):
    mask = pd.Series(True, index = df_or_series.index)
    mask.loc[index_to_exclude] = False
    return df_or_series.loc[mask]
    
def swap_series_index_and_value(series):
    return pd.Series(series.index, index = series.values)
    
def concat_dfs_with_partial_columns(dfs):
    columns = max([df.columns for df in dfs], key = len)
    assert all([set(df.columns) <= set(columns) for df in dfs])
    return pd.concat(dfs, sort = False)[columns]
    
def concat_dfs_with_compatible_columns(dfs):
    columns = merge_lists_with_compatible_relative_order([df.columns for df in dfs])
    return pd.concat(dfs, sort = False)[columns]

def safe_get_df_group(df_groupby, group_name):
    if group_name in df_groupby.groups:
        return df_groupby.get_group(group_name)
    else:
        _, some_group_df = next(iter(df_groupby))
        return pd.DataFrame(columns = some_group_df.columns)
        
def bin_groupby(df, series_or_col_name, n_bins):
    
    if len(df) == 0:
        return df
    
    if isinstance(series_or_col_name, str):
        series = df[series_or_col_name]
    else:
        series = series_or_col_name
        
    min_value, max_value = series.min(), series.max()
    bin_size = (max_value - min_value) / n_bins
    
    bind_ids = ((series - min_value) / bin_size).astype(int)
    bind_ids[bind_ids >= n_bins] = n_bins - 1
    
    return df.groupby(bind_ids)
    
def value_df_to_hot_encoding_df(value_df, value_headers = {}):
    
    flat_values = value_df.values.flatten()
    all_values = sorted(np.unique(flat_values[pd.notnull(flat_values)]))
    value_to_index = {value: i for i, value in enumerate(all_values)}
    hot_encoding_matrix = np.zeros((len(value_df), len(all_values)))
    
    for _, column_values in value_df.iteritems():
        row_position_to_value_index = column_values.reset_index(drop = True).dropna().map(value_to_index)
        hot_encoding_matrix[row_position_to_value_index.index.values, row_position_to_value_index.values] = 1
    
    headers = [value_headers.get(value, value) for value in all_values]
    return pd.DataFrame(hot_encoding_matrix, index = value_df.index, columns = headers)
    
def set_series_to_hot_encoding_df(set_series, value_headers = {}):

    all_values = sorted(set.union(*set_series))
    value_to_index = {value: i for i, value in enumerate(all_values)}
    hot_encoding_matrix = np.zeros((len(set_series), len(all_values)))
    
    for i, record_values in enumerate(set_series):
        hot_encoding_matrix[i, [value_to_index[value] for value in record_values]] = 1
        
    headers = [value_headers.get(value, value) for value in all_values]
    return pd.DataFrame(hot_encoding_matrix, index = set_series.index, columns = headers)
    
def resolve_dummy_variable_trap(hot_encoding_df, validate_completeness = True, inplace = False, verbose = True):

    '''
    When using one-hot-encoding in regression, there is a problem of encoding all possible variables if also using an intercept/const variable,
    because then the variables end up linearly dependent (a singular matrix is problematic with many implementations of regression). See for
    example: https://www.algosome.com/articles/dummy-variable-trap-regression.html
    To resolve this issue, this function will remove the most frequent column (to minimize the chance of any subset of the rows resulting a
    matrix which is not fully ranked).
    '''
    
    # Validate we are indeed dealing with one-hot-encoding.
    assert set(np.unique(hot_encoding_df.values).astype(float)) <= {0.0, 1.0}
    
    if validate_completeness:
        assert (hot_encoding_df.sum(axis = 1) == 1).all()
    else:
        assert (hot_encoding_df.sum(axis = 1) <= 1).all()
    
    most_frequent_variable = hot_encoding_df.sum().idxmax()
    
    if verbose:
        log('To avoid the "dummy variable trap", removing the %s column (%d matching records).' % (most_frequent_variable, \
                hot_encoding_df[most_frequent_variable].sum()))
    
    if inplace:
        del hot_encoding_df[most_frequent_variable]
    else:
        return hot_encoding_df[[column_name for column_name in hot_encoding_df.columns if column_name != most_frequent_variable]]
    
def set_constant_row(df, row_mask, row_values):
    df[row_mask] = np.tile(row_values, (row_mask.sum(), 1))
    
def construct_df_from_rows(row_repertoire, row_indexer):
    
    result = pd.DataFrame(index = row_indexer.index, columns = row_repertoire.columns)
    
    for row_index, row_values in row_repertoire.iterrows():
        set_constant_row(result, row_indexer == row_index, row_values)
        
    return result
    
def get_row_last_values(df):
    
    result = pd.Series(np.nan, index = df.index)

    for column in df.columns[::-1]:
        result = result.where(pd.notnull(result), df[column])

    return result
    
def are_close_dfs(df1, df2, rtol = 1e-05, atol = 1e-08):
    
    assert (df1.dtypes == df2.dtypes).all()
    
    for column, dtype in df1.dtypes.iteritems():
        
        if np.issubdtype(dtype, np.float):
            cmp_series = np.isclose(df1[column], df2[column], rtol = rtol, atol = atol) | (pd.isnull(df1[column]) & \
                    pd.isnull(df2[column]))
        else:
            cmp_series = (df1[column] == df2[column])
            
        if not cmp_series.all():
            return False
        
    return True
    
def append_df_to_excel(excel_writer, df, sheet_name, index = True):
    
    header_format = excel_writer.book.add_format({'bold': True})
     
    df.to_excel(excel_writer, sheet_name, index = index)
    worksheet = excel_writer.sheets[sheet_name]
    
    for column_index, column_name in enumerate(df.columns):
        worksheet.write(0, column_index + int(index), column_name, header_format)
        
    if index:
        for row_index_number, row_index_value in enumerate(df.index):
            worksheet.write(row_index_number + 1, 0, row_index_value)
        
def is_binary_series(series):

    # First validating that the type of the series is convertable to float.
    try:
        float(series.iloc[0])
    except TypeError:
        return False

    return set(series.unique().astype(float)) <= {0.0, 1.0}
    
def resolve_quasi_complete_separation_by_removing_binary_columns(X, y):
    
    '''
    When performing logistic regression of y against X, the matrix X must be of full rank; otherwise (i.e. if the columns of X are
    linearly dependent), then statsmodel's Logit model gives a singular-matrix error. It also appears that quasi-complete separation
    causes troubles, namely if the columns of X are linearly dependent conditioned on y. In other words, assuming that y is binary,
    we need that X[y, :] would still be of full rank (we assume that the vast majority of records have a negative y value, and only
    a small fraction have a positive value, so given that X is of full rank we need not worry about X[~y, :]). To resolve this problem,
    this function will remove binary columns of X until X[y, :] is of full rank. Whenever a column of X is removed, we also remove the
    corresponding records (rows of X and y) that have those values (so if a removed column represent some covariate, e.g. a certain
    batch, we also remove all the samples from this batch in order for not having any covariates not accounted for).
    @param X (pd.DataFrame): The exogenous variables (rows are records, columns are variables).
    @pram y (pd.Series): The endogenous variable (must have the same index as X).
    '''
    
    row_mask = pd.Series(True, index = X.index)
    
    if not is_binary_series(y):
        return X, y, X.columns, set(), row_mask
        
    boolean_y = y.astype(bool)
    all_kept_binary_columns = np.array([column_name for column_name in X.columns if is_binary_series(X[column_name])])
    # We sort the binary columns by how common they are, so when we start removing them, we will give priority to the more common ones
    # (i.e. remove the least frequent first).
    all_kept_binary_columns = X[all_kept_binary_columns].sum().sort_values(ascending = False).index
    all_removed_binary_columns = set()
    
    while len(all_kept_binary_columns) > 0:
    
        positive_X = X.loc[row_mask & boolean_y, all_kept_binary_columns]
        old_all_kept_binary_columns = all_kept_binary_columns
        all_kept_binary_columns = all_kept_binary_columns[find_linearly_independent_columns(positive_X.values)]
        columns_to_remove = set(old_all_kept_binary_columns) - set(all_kept_binary_columns)
        
        for column_name in columns_to_remove:
            log('Removing the columns %s (%d occurances) to avoid quasi-complete separation.' % (column_name, X[column_name].sum()))
            all_removed_binary_columns.add(column_name)
            row_mask &= (~X[column_name].astype(bool))
            
        if len(columns_to_remove) == 0:
            break

    if not row_mask.all():
        log('Overall removed %d columns occuring in %d records to avoid quasi-complete separation.' % (len(all_removed_binary_columns), \
                (~row_mask).sum()))
        
    retained_columns = [column_name for column_name in X.columns if column_name not in all_removed_binary_columns]
    X = X.loc[row_mask, retained_columns]
    y = y.loc[row_mask]
    
    return X, y, retained_columns, all_removed_binary_columns, row_mask

    
### Statistics ###

def to_normal_z_values(raw_values):

    from scipy.stats import rankdata, norm
    
    pvals = (rankdata(raw_values) - 0.5) / len(raw_values)
    normal_z_values = norm.ppf(pvals)
    
    if isinstance(raw_values, pd.Series):
        return pd.Series(normal_z_values, index = raw_values.index)
    else:
        return normal_z_values

def multipletests_with_nulls(values, method = 'fdr_bh'):

    from statsmodels.stats.multitest import multipletests
    
    significance = np.zeros(len(values), dtype = bool)
    qvals = np.nan * np.empty(len(values))
    mask = pd.notnull(values)
    
    if mask.any():
        significance[np.array(mask)], qvals[np.array(mask)], _, _ = multipletests(values[mask], method = method)
    
    return significance, qvals
    
def test_enrichment(mask1, mask2):

    from scipy.stats import fisher_exact

    assert len(mask1) == len(mask2)
    
    n1 = mask1.sum()
    n2 = mask2.sum()
    n_both = (mask1 & mask2).sum()
    n_total = len(mask1)
    n_expected = n1 * n2 / n_total
    enrichment_factor = n_both / n_expected
    
    contingency_table = np.array([
        [(mask1 & mask2).sum(), (mask1 & (~mask2)).sum()],
        [((~mask1) & mask2).sum(), ((~mask1) & (~mask2)).sum()],
    ])
    _, pval = fisher_exact(contingency_table)
    
    return n1, n2, n_both, n_total, n_expected, enrichment_factor, contingency_table, pval
    
def test_enrichment_sets(set1, set2, n_total):

    from scipy.stats import fisher_exact
    
    n1 = len(set1)
    n2 = len(set2)
    n_both = len(set1 & set2)
    n_expected = n1 * n2 / n_total
    enrichment_factor = n_both / n_expected
    
    contingency_table = np.array([
        [n_both, n1 - n_both],
        [n2 - n_both, n_total - n1 - n2 + n_both],
    ])
    _, pval = fisher_exact(contingency_table)
    
    return n1, n2, n_both, n_total, n_expected, enrichment_factor, contingency_table, pval
    
    
### h5f ###
    
def flush_h5_file(h5f):
    h5f.flush()
    os.fsync(h5f.id.get_vfd_handle())
    
def transpose_h5f_dataset(h5f, src_name, dst_name, max_memory_bytes):
    flush_func = lambda: flush_h5_file(h5f)
    src = h5f[src_name]
    nrows, ncols = src.shape[:2]
    dst = h5f.create_dataset(dst_name, shape = (ncols, nrows), dtype = src.dtype)
    transpose_dataset(src, dst, max_memory_bytes, flush_func)
    
    
### Matplotlib ###

def draw_rectangle(ax, start_x, end_x, start_y, end_y, **kwargs):
    from matplotlib import patches
    ax.add_patch(patches.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y, **kwargs))
    
def set_ax_border_color(ax, color):

    import matplotlib.pyplot as plt

    for child in ax.get_children():
        if isinstance(child, plt.matplotlib.spines.Spine):
            child.set_color(color)
    
def plot_prediction_scatter(y_pred, y_true, value = 'value'):

    import matplotlib.pyplot as plt
    
    log(pearsonr(y_pred, y_true))
    log(spearmanr(y_pred, y_true))

    fig, ax = plt.subplots(figsize = (10, 6))
    ax.scatter(y_pred, y_true)
    ax.set_xlabel('Predicted %s' % value)
    ax.set_ylabel('Actual %s' % value)
    
def draw_pvals_qq_plot(pvals, max_density = 100, min_pval = None, ax = None, figsize = (7, 7), scatter_options = {}, \
        xlabel = 'Expected p-values (-log10)', ylabel = 'Observed p-values (-log10)'):
    
    import matplotlib.pyplot as plt
    
    if 'color' not in scatter_options:
        scatter_options['color'] = '#2e75b6'
    
    pvals = np.array(pvals)
    
    if min_pval is not None:
        pvals = np.maximum(pvals, min_pval)
    
    n_total_pvals = len(pvals)
    sorted_mlog_pvals = np.sort(-np.log10(pvals))
    max_mlog_pval = sorted_mlog_pvals.max()
    
    if ax is None:
        _, ax = plt.subplots(figsize = figsize)
    
    ax.plot([0, max_mlog_pval], [0, max_mlog_pval], color = 'red', linestyle = '--', alpha = 0.5)
    ax.set_xlim((0, max_mlog_pval))
    ax.set_ylim((0, max_mlog_pval))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    for upper_limit in range(1, int(max_mlog_pval + 3)):
        
        n_remained_pvals = len(sorted_mlog_pvals)
        i = np.searchsorted(sorted_mlog_pvals, upper_limit)
        range_pvals = sorted_mlog_pvals[:i]
        sorted_mlog_pvals = sorted_mlog_pvals[i:]
        
        if len(range_pvals) > 0:
                    
            if len(range_pvals) <= max_density:
                range_chosen_indices = np.arange(len(range_pvals))
            else:
                # We want to choose the p-values uniformly in the space of their expected frequencies (i.e. sampling more towards the higher end of the
                # spectrum).
                range_min_mlog_freq = -np.log10(n_remained_pvals / n_total_pvals)
                range_max_mlog_freq = -np.log10((n_remained_pvals - len(range_pvals) + 1) / n_total_pvals)
                range_chosen_mlog_freqs = np.linspace(range_min_mlog_freq, range_max_mlog_freq, max_density)
                range_chosen_freqs = np.power(10, -range_chosen_mlog_freqs)
                # Once having the desired freqs, reverse the function to get the indices that provide them
                range_chosen_indices = np.unique((n_remained_pvals - n_total_pvals * range_chosen_freqs).astype(int))

            range_pvals = range_pvals[range_chosen_indices]
            range_freqs = (n_remained_pvals - range_chosen_indices) / n_total_pvals
            range_mlog_freqs = -np.log10(range_freqs)
            ax.scatter(range_mlog_freqs, range_pvals, **scatter_options)
            
def draw_manhattan_plot(gwas_results, significance_treshold = 5e-08, max_results_to_plot = 1e06, \
        pval_threshold_to_force_inclusion = 1e-03, min_pval = 1e-300, ax = None, figsize = (12, 6), \
        s = 1.5, chrom_to_color = None):
    
    '''
    gwas_results (pd.DataFrame): Should have the following columns:
    - chromosome (str)
    - position (int)
    - pval (float)
    '''
    
    import matplotlib.pyplot as plt
        
    CHROMS = list(map(str, range(1, 23))) + ['X', 'Y']
    CHROM_TO_COLOR = {'1': '#0100fb', '2': '#ffff00', '3': '#00ff03', '4': '#bfbfbf', '5': '#acdae9', '6': '#a020f1',
            '7': '#ffa502', '8': '#ff00fe', '9': '#fe0000', '10': '#90ee90', '11': '#a52929', '12': '#000000', 
            '13': '#ffbfcf', '14': '#4484b2', '15': '#b63063', '16': '#f8816f', '17': '#ed84f3', '18': '#006401',
            '19': '#020184', '20': '#ced000', '21': '#cd0001', '22': '#050098', 'X': '#505050', 'Y': '#ff8000'}
    
    if chrom_to_color is None:
        chrom_to_color = CHROM_TO_COLOR
    
    if len(gwas_results) > max_results_to_plot:
        mask = pd.Series(random_mask(len(gwas_results), int(max_results_to_plot)), index = gwas_results.index)
        mask[gwas_results['pval'] <= pval_threshold_to_force_inclusion] = True
        gwas_results = gwas_results[mask]
    
    max_pos_per_chrom = gwas_results.groupby('chromosome')['position'].max()
    accumulating_pos = 0
    chrom_accumulating_positions = []
    
    for chrom in CHROMS:
        if chrom in max_pos_per_chrom.index:
            chrom_accumulating_positions.append((chrom, accumulating_pos + 1, accumulating_pos + max_pos_per_chrom[chrom]))
            accumulating_pos += max_pos_per_chrom[chrom]
            
    chrom_accumulating_positions = pd.DataFrame(chrom_accumulating_positions, columns = ['chromosome', \
            'accumulating_start_position', 'accumulating_end_position']).set_index('chromosome', drop = True)
    chrom_middle_accumulating_positions = (chrom_accumulating_positions['accumulating_start_position'] + \
            chrom_accumulating_positions['accumulating_end_position']) / 2
        
    if ax is None:
        _, ax = plt.subplots(figsize = figsize)
    
    ax.set_facecolor('white')
    plt.setp(ax.spines.values(), color = '#444444')
    ax.grid(False)
    
    if significance_treshold is not None:
        ax.axhline(y = -np.log10(significance_treshold), linestyle = '--', linewidth = 1, color = 'red')
    
    gwas_results_per_chrom = gwas_results.groupby('chromosome')
    max_y = 0
    
    for chrom in chrom_accumulating_positions.index:
        chrom_gwas_results = gwas_results_per_chrom.get_group(chrom)
        chrom_gwas_accumulating_positions = chrom_accumulating_positions.loc[chrom, 'accumulating_start_position'] + \
                chrom_gwas_results['position']
        chrom_gwas_minus_log_pval = -np.log10(np.maximum(chrom_gwas_results['pval'], min_pval))
        max_y = max(max_y, chrom_gwas_minus_log_pval.max())
        ax.scatter(chrom_gwas_accumulating_positions, chrom_gwas_minus_log_pval, color = chrom_to_color[chrom], s = s)
        
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('-log10(p-value)')
    ax.set_xticks(chrom_middle_accumulating_positions)
    ax.set_xticklabels(chrom_middle_accumulating_positions.index)
    ax.set_xlim(1, accumulating_pos)
    ax.set_ylim(0, max_y + 1)
    
    return ax
    
    
### Biopython Helper Functions ###

def as_biopython_seq(seq):

    from Bio.Seq import Seq

    if isinstance(seq, Seq):
        return seq
    elif isinstance(seq, str):
        return Seq(seq)
    else:
        raise Exception('Cannot resolve type %s as Biopython Seq' % type(seq))
            
            
### Slurm ###

def get_slurm_job_array_ids(parse_total_tasks_by_max_variable = True, log_ids = True, verbose = True, task_index_remapping_json_file_path = None):

    job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))
    task_index = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    
    if 'TASK_ID_OFFSET' in os.environ:
        
        task_offset = int(os.getenv('TASK_ID_OFFSET'))
        
        if verbose:
            log('Raw task index %d with offset %d.' % (task_index, task_offset))
        
        task_index += task_offset
        
    if task_index_remapping_json_file_path is not None:
        
        with open(task_index_remapping_json_file_path, 'r') as f:
            task_index_remapping = json.load(f)
            
        remapped_task_index = task_index_remapping[task_index]
        
        if verbose:
            log('Remapped task index %d into %d.' % (task_index, remapped_task_index))
        
        task_index = remapped_task_index
    
    if 'TOTAL_TASKS' in os.environ:
        total_tasks = int(os.getenv('TOTAL_TASKS'))
    elif parse_total_tasks_by_max_variable:
        total_tasks = int(os.getenv('SLURM_ARRAY_TASK_MAX')) + 1
    else:
        total_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    
    if log_ids:
        log('Running job %s, task %d of %d.' % (job_id, task_index, total_tasks))
    
    return job_id, total_tasks, task_index 


### Liftover ###

def liftover_locus(liftover, chrom, pos):
    try:
            
        pos = int(pos)

        if not isinstance(chrom, str) or not chrom.startswith('chr'):
            chrom = 'chr%s' % chrom

        (new_chrom, new_pos, _, _), = liftover.convert_coordinate(chrom, pos)

        if new_chrom.startswith('chr'):
            new_chrom = new_chrom[3:]

        return new_chrom, new_pos
    except:
        return np.nan, np.nan

def liftover_loci_in_df(df, chrom_column = 'chromosome', pos_column = 'position', source_ref_genome = 'hg38', \
        target_ref_genome = 'hg19'):
    
    from pyliftover import LiftOver
    
    liftover = LiftOver(source_ref_genome, target_ref_genome)
    new_loci = []
    
    for _, (chrom, pos) in df[[chrom_column, pos_column]].iterrows():
        new_loci.append(liftover_locus(liftover, chrom, pos))
            
    new_chroms, new_positions = (pd.Series(list(values), index = df.index) for values in zip(*new_loci))
    return pd.concat([new_chroms.rename(chrom_column) if column == chrom_column else (new_positions.rename(pos_column) if \
            column == pos_column else df[column]) for column in df.columns], axis = 1)    
    
    
### Helper classes ###

class DummyContext(object):

    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

class TimeMeasure(object):

    def __init__(self, opening_statement):
        self.opening_statement = opening_statement

    def __enter__(self):
        self.start_time = datetime.now()
        log(self.opening_statement)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.finish_time = datetime.now()
        self.elapsed_time = self.finish_time - self.start_time
        log('Finished after %s.' % self.elapsed_time)
        
class Profiler(object):

    def __init__(self):
        self.creation_time = datetime.now()
        self.profiles = defaultdict(Profiler.Profile)
        
    def measure(self, profile_name):
        return self.profiles[profile_name].measure()
        
    def format(self, delimiter = '\n'):
        all_profiles = list(self.profiles.items()) + [('Total', Profiler.Profile(total_invokes = 1, total_time = datetime.now() - self.creation_time))]
        sorted_profiles = sorted(all_profiles, key = lambda profile_tuple: profile_tuple[1].total_time, reverse = True)
        return delimiter.join(['%s: %s' % (profile_name, profile) for profile_name, profile in sorted_profiles])
        
    def __repr__(self):
        return self.format()
        
    class Profile(object):
    
        def __init__(self, total_invokes = 0, total_time = timedelta(0)):
            self.total_invokes = total_invokes
            self.total_time = total_time
            
        def measure(self):
            return Profiler._Measurement(self)
            
        def __repr__(self):
            return '%s (%d times)' % (self.total_time, self.total_invokes)
        
    class _Measurement(object):
    
        def __init__(self, profile):
            self.profile = profile
        
        def __enter__(self):
            self.start_time = datetime.now()
            
        def __exit__(self, exc_type, exc_value, exc_traceback):
            self.profile.total_time += (datetime.now() - self.start_time)
            self.profile.total_invokes += 1
