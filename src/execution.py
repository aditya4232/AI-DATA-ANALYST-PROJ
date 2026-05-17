from __future__ import annotations

from dataclasses import dataclass
import ast
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISALLOWED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "requests",
}

ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass(frozen=True)
class ExecutionOutcome:
    result: Any = None
    figures: list[Any] | None = None
    locals_map: dict[str, Any] | None = None
    code: str = ""
    result_type: str = ""


class AnalysisCodeError(RuntimeError):
    pass


def execute_analysis_code(code: str, df: pd.DataFrame) -> ExecutionOutcome:
    validate_analysis_code(code)
    plt.close("all")

    safe_globals = {
        "__builtins__": ALLOWED_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    local_vars: dict[str, Any] = {}
    compiled = compile(code, "<analysis>", "exec")
    exec(compiled, safe_globals, local_vars)

    result = local_vars.get("result")
    if result is None:
        result = local_vars.get("answer")
    if result is None and "table" in local_vars:
        result = local_vars["table"]
    if result is None and "output" in local_vars:
        result = local_vars["output"]

    figures = [plt.figure(num) for num in plt.get_fignums()]
    result_type = type(result).__name__ if result is not None else "None"
    return ExecutionOutcome(result=result, figures=figures, locals_map=local_vars, code=code, result_type=result_type)


def validate_analysis_code(code: str) -> None:
    if not code.strip():
        raise AnalysisCodeError("The model returned empty code.")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise AnalysisCodeError(f"The generated code has a syntax error: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise AnalysisCodeError("Imports are not allowed in generated code.")
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            raise AnalysisCodeError(f"Disallowed name used in generated code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise AnalysisCodeError("Dunder attribute access is not allowed.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_NAMES:
            raise AnalysisCodeError(f"Disallowed call used in generated code: {node.func.id}")
