from types import SimpleNamespace
import json


def dict_to_namespace(d) -> SimpleNamespace:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def load_config(path) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    config_data = json.loads(text)
    args = dict_to_namespace(config_data)
    return args

rm_prefix = [
    11,
    65520,
]
rm_postfix = [
    11,
    65521,
]