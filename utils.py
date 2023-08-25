import inspect
import re


def dbg(*args):
    """
    Print the name and value of all arguments passed to the function.

    Parameters:
    * args: The arguments passed to the function.
    """

    # Get the frame information of the caller function.
    frame = inspect.currentframe().f_back

    # Get the code context of the caller function.
    s = inspect.getframeinfo(frame).code_context[0]

    # Get the names of the arguments from the code context.
    r = re.search(r"(.*)", s).group(1)
    vnames = r.split(", ")

    # Iterate over the arguments and print their names and values.
    for i, (var, val) in enumerate(zip(vnames, args)):
        print(f"{var} = {val}")