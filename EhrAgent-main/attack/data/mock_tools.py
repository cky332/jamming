"""Mock tools that operate on the stub_db, replacing tools/tabtools.py operations.

These functions preserve the exact signatures and return semantics that EHRAgent's
prompts expect (see prompts_mimic.EHRAgent_Message_Prompt), so an agent emitting
structurally-correct DB chains gets correct answers from the stub data.

Patching: install_mock_tools() monkey-patches tools.tabtools at runtime so the
existing toolset_high.run_code path picks up the stub instead of the file-system DB.
"""

import sqlite3
from datetime import datetime, timedelta
import re
import pandas as pd
import Levenshtein

from attack.data import stub_db

_SQL_DB = None


def _build_sql_db() -> sqlite3.Connection:
    global _SQL_DB
    if _SQL_DB is not None:
        return _SQL_DB
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for name in stub_db.all_table_names():
        df = stub_db.load_table(name)
        df.to_sql(name, conn, if_exists="replace", index=False)
    _SQL_DB = conn
    return conn


def reset_sql_db():
    global _SQL_DB
    if _SQL_DB is not None:
        _SQL_DB.close()
    _SQL_DB = None


def db_loader(target_ehr: str) -> pd.DataFrame:
    return stub_db.load_table(target_ehr)


def data_filter(data: pd.DataFrame, argument: str) -> pd.DataFrame:
    backup_data = data
    commands = argument.split('||')
    for cmd in commands:
        column_name = ""
        value = ""
        try:
            if '>=' in cmd:
                column_name, value = cmd.split('>=', 1)
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] >= value]
            elif '<=' in cmd:
                column_name, value = cmd.split('<=', 1)
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] <= value]
            elif '>' in cmd:
                column_name, value = cmd.split('>', 1)
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] > value]
            elif '<' in cmd:
                column_name, value = cmd.split('<', 1)
                if value and value[0] in ("'", '"'):
                    value = value[1:-1]
                try:
                    value = type(data[column_name].iloc[0])(value)
                except Exception:
                    pass
                data = data[data[column_name] < value]
            elif '=' in cmd:
                column_name, value = cmd.split('=', 1)
                if value and value[0] in ("'", '"'):
                    value = value[1:-1]
                try:
                    exemplar = backup_data[column_name].tolist()[0]
                    value = type(exemplar)(value)
                except Exception:
                    pass
                data = data[data[column_name] == value]
            elif ' in ' in cmd:
                column_name, value = cmd.split(' in ', 1)
                value_list = [s.strip().strip("'").strip('"') for s in value.strip("[]").split(',')]
                try:
                    value_list = list(map(type(data[column_name].iloc[0]), value_list))
                except Exception:
                    pass
                data = data[data[column_name].isin(value_list)]
            elif 'max' in cmd:
                column_name = cmd.split('max(', 1)[1].split(')')[0]
                data = data[data[column_name] == data[column_name].max()]
            elif 'min' in cmd:
                column_name = cmd.split('min(', 1)[1].split(')')[0]
                data = data[data[column_name] == data[column_name].min()]
        except Exception:
            if column_name and column_name not in data.columns.tolist():
                cols = ', '.join(data.columns.tolist())
                raise Exception(
                    f"The filtering query {cmd} is incorrect. Please modify the column name "
                    f"or use LoadDB to read another table. The column names in the current DB are {cols}."
                )
            if not column_name or not value:
                raise Exception(
                    f"The filtering query {cmd} is incorrect. There is syntax error in the command."
                )
        if len(data) == 0:
            column_values = list(set(backup_data[column_name].tolist())) if column_name in backup_data.columns else []
            if ('=' in cmd) and (str(value) not in [str(cv) for cv in column_values]) and ('>=' not in cmd) and ('<=' not in cmd):
                levenshtein_dist = {cv: Levenshtein.distance(str(cv), str(value)) for cv in column_values}
                sorted_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1])[:5]
                examples = ', '.join(str(i[0]) for i in sorted_dist)
                raise Exception(
                    f"The filtering query {cmd} is incorrect. There is no {value} value in the column. "
                    f"Five example values in the column are {examples}. "
                    f"Please check if you get the correct {column_name} value."
                )
            else:
                return data
    return data


def get_value(data: pd.DataFrame, argument: str):
    try:
        commands = argument.split(', ')
        if len(commands) == 1:
            column = argument
            while column and column[0] in ('[', "'"):
                column = column[1:]
            while column and column[-1] in (']', "'"):
                column = column[:-1]
            if len(data) == 1:
                return str(data.iloc[0][column])
            else:
                answer_list = list(set(data[column].tolist()))
                answer_list = [str(i) for i in answer_list]
                return ', '.join(answer_list)
        else:
            column = commands[0]
            op = commands[-1]
            if 'mean' in op:
                res = [float(x) for x in data[column].tolist()]
                return sum(res) / len(res)
            elif 'max' in op:
                res = data[column].tolist()
                try:
                    res = [float(x) for x in res]
                except Exception:
                    res = [str(x) for x in res]
                return max(res)
            elif 'min' in op:
                res = data[column].tolist()
                try:
                    res = [float(x) for x in res]
                except Exception:
                    res = [str(x) for x in res]
                return min(res)
            elif 'sum' in op:
                res = [float(x) for x in data[column].tolist()]
                return sum(res)
            elif 'list' in op:
                return [str(x) for x in data[column].tolist()]
            else:
                raise Exception(f"The operation {op} contains syntax errors. Please check the arguments.")
    except Exception:
        cols = ', '.join(data.columns.tolist())
        raise Exception(
            f"The column name {argument} is incorrect. Please check the column name. "
            f"The columns in this table include {cols}."
        )


def sql_interpreter(command: str):
    conn = _build_sql_db()
    cur = conn.cursor()
    try:
        results = cur.execute(command).fetchall()
    except Exception as e:
        raise Exception(f"SQL error: {e}")
    return results


_NOW = datetime(2100, 12, 31, 12, 0, 0)


def date_calculator(argument: str) -> str:
    arg = argument.strip()
    m = re.match(r'^(-?\d+)\s*(year|month|day|hour|minute)s?$', arg)
    if not m:
        raise Exception(
            f"The date calculator {argument!r} is incorrect. Use forms like '-1 year' or '0 year'."
        )
    qty = int(m.group(1))
    unit = m.group(2)
    if unit == "year":
        result = _NOW.replace(year=_NOW.year + qty)
    elif unit == "month":
        new_month = _NOW.month + qty
        years_offset, month0 = divmod(new_month - 1, 12)
        result = _NOW.replace(year=_NOW.year + years_offset, month=month0 + 1)
    elif unit == "day":
        result = _NOW + timedelta(days=qty)
    elif unit == "hour":
        result = _NOW + timedelta(hours=qty)
    elif unit == "minute":
        result = _NOW + timedelta(minutes=qty)
    else:
        raise Exception(f"Unsupported unit: {unit}")
    return result.strftime("%Y-%m-%d %H:%M:%S")


def _simple_calculator(query: str):
    from operator import add, sub, mul, truediv, pow
    ops = {'+': add, '-': sub, '*': mul, '/': truediv}
    query = re.sub(r'\s+', '', query)
    try:
        return float(eval(query, {"__builtins__": {}}, {}))
    except Exception:
        raise Exception(
            "Invalid input query for Calculator. Please check the input query or use other functions."
        )


def install_mock_tools():
    """Monkey-patch tools.tabtools and tools.calculator with stub versions
    so toolset_high.run_code picks them up."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from tools import tabtools as _tabtools
    from tools import calculator as _calculator
    _tabtools.db_loader = db_loader
    _tabtools.data_filter = data_filter
    _tabtools.get_value = get_value
    _tabtools.sql_interpreter = sql_interpreter
    _tabtools.date_calculator = date_calculator
    _calculator.WolframAlphaCalculator = _simple_calculator
