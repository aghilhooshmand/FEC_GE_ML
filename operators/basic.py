from typing import Dict, Callable, List
import numpy as np


# Arithmetic (UGE built-ins)
def add(a, b):
    return np.add(a, b)


def sub(a, b):
    return np.subtract(a, b)


def mul(a, b):
    return np.multiply(a, b)


def pdiv(a, b):
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b == 0, np.ones_like(a), a / b)
    except ZeroDivisionError:
        return 1.0


# Mathematical functions (UGE built-ins)
def sigmoid(arr):
    if np.isscalar(arr):
        arr = np.array([arr])
    return 1 / (1 + np.exp(-arr))


def psqrt(a):
    return np.sqrt(np.abs(a))


def psin(n):
    return np.sin(n)


def pcos(n):
    return np.cos(n)


def plog(a):
    return np.log(1.0 + np.abs(a))


def exp(a):
    return np.exp(a)


def abs_(a):
    return np.abs(a)


def minimum(a, b):
    return np.minimum(a, b)


def maximum(a, b):
    return np.maximum(a, b)


def min_(a, b):
    return np.minimum(a, b)


def max_(a, b):
    return np.maximum(a, b)


def neg(a):
    return -a


# Logical (UGE built-ins)
def and_(a, b):
    return np.logical_and(a, b)


def or_(a, b):
    return np.logical_or(a, b)


def not_(a):
    return np.logical_not(a)


def nand_(a, b):
    return np.logical_not(np.logical_and(a, b))


def nor_(a, b):
    return np.logical_not(np.logical_or(a, b))


# Comparisons (UGE built-ins)
def greater_than_or_equal(a, b):
    return a >= b


def less_than_or_equal(a, b):
    return a <= b


def greater_than(a, b):
    return a > b


def less_than(a, b):
    return a < b


def equal(a, b):
    return a == b


def not_equal(a, b):
    return a != b


# Conditional (UGE built-in)
def if_(i, o0, o1):
    return np.where(i, o0, o1)


# Aliases to keep earlier simple expressions working in this project
def safe_div(a, b):
    return pdiv(a, b)


def safe_log(a):
    # Compatible with earlier usage; keep as protected log
    return plog(a)


# Operator dictionaries used by the notebook
OPERATORS: Dict[str, Callable] = {
    # arithmetic
    "add": add,
    "sub": sub,
    "mul": mul,
    "pdiv": pdiv,
    # aliases
    "div": pdiv,
    # min/max variants
    "minimum": minimum,
    "maximum": maximum,
    "min_": min_,
    "max_": max_,
    # logical and comparisons (non-unary go here)
    "and_": and_,
    "or_": or_,
    "nand_": nand_,
    "nor_": nor_,
    "greater_than_or_equal": greater_than_or_equal,
    "less_than_or_equal": less_than_or_equal,
    "greater_than": greater_than,
    "less_than": less_than,
    "equal": equal,
    "not_equal": not_equal,
    # conditional (arity 3)
    "if_": if_,
}


UNARIES: Dict[str, Callable] = {
    # math
    "sigmoid": sigmoid,
    "psqrt": psqrt,
    "psin": psin,
    "pcos": pcos,
    "plog": plog,
    "exp": exp,
    "abs_": abs_,
    "neg": neg,
    # aliases used earlier in the notebook (non-UGE names)
    "sin": psin,
    "cos": pcos,
    "log": plog,
}


def feature_names(n_features: int) -> List[str]:
    return [f"x{i}" for i in range(int(n_features))]


