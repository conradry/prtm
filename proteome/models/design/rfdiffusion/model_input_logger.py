import datetime
import os
import pickle
import traceback
from inspect import signature


def pickle_function_call_wrapper(func, output_dir="pickled_inputs"):
    i = 0
    os.makedirs(output_dir)

    # pickle.dump({'args': args, 'kwargs': kwargs}, fh)
    def wrapper(*args, **kwargs):
        """
        Wrap the original function call to print the arguments before
        calling the intended function
        """
        nonlocal i
        i += 1
        func_sig = signature(func)
        # Create the argument binding so we can determine what
        # parameters are given what values
        argument_binding = func_sig.bind(*args, **kwargs)
        argument_map = argument_binding.arguments

        # Perform the print so that it shows the function name
        # and arguments as a dictionary
        path = os.path.join(output_dir, f"{i:05d}.pkl")
        print(
            f"logging {func.__name__} arguments: {[k for k in argument_map]} to {path}"
        )
        argument_map["stack"] = traceback.format_stack()

        for k, v in argument_map.items():
            if hasattr(v, "detach"):
                argument_map[k] = v.cpu().detach()
        with open(path, "wb") as fh:
            pickle.dump(argument_map, fh)

        return func(*args, **kwargs)

    return wrapper


def wrap_it(wrapper, instance, method, **kwargs):
    class_method = getattr(instance, method)
    wrapped_method = wrapper(class_method, **kwargs)
    setattr(instance, method, wrapped_method)


def pickle_function_call(instance, method, subdir):
    output_dir = os.path.join(
        os.getcwd(),
        "pickled_inputs",
        subdir,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    wrap_it(pickle_function_call_wrapper, instance, method, output_dir=output_dir)
    return output_dir
