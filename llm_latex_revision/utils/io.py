import json
import os
from typing import Union

FilePath = Union[str, os.PathLike]


def json_parse_constant(arg):
    c = {"-Infinity": -float("inf"), "Infinity": float("inf"), "NaN": float("nan")}
    return c[arg]


def json_dump(o, fn: FilePath):
    with open(fn, "w") as f:
        json.dump(o, f, indent=2)


def json_load(fn: FilePath):
    with open(fn, "r") as f:
        o = json.load(f, parse_constant=json_parse_constant)
    return o
