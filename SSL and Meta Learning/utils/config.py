"""YAML config loading with presets, inheritance and `--set a.b=c` overrides.

Configs are plain nested dicts (only python builtins) so they can be stored inside checkpoints
that load with `torch.load(..., weights_only=True)`.
"""
import argparse
import copy
import json
import os

import yaml

from utils.paths import REPO_ROOT

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _resolve(name_or_path):
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    p = os.path.join(CONFIG_DIR, name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml")
    if not os.path.isfile(p):
        avail = sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml"))
        raise FileNotFoundError(f"Config '{name_or_path}' not found. Available presets: {avail}")
    return p


def load_yaml_with_inherit(name_or_path):
    path = _resolve(name_or_path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_yaml_with_inherit(parent)
        cfg = deep_update(base, cfg)
    return cfg


def apply_overrides(cfg, overrides):
    """overrides: list of 'a.b.c=value' strings; value parsed as YAML (so 0.1, true, [1,2] work)."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected key=value")
        key, val = ov.split("=", 1)
        val = yaml.safe_load(val)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def add_common_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("common")
    g.add_argument("--config", default="base", help="preset name in configs/ (small|base|tiny) or a yaml path")
    g.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VAL",
                   help="config overrides, e.g. ssl.lr=0.5 (repeatable)")
    g.add_argument("--run-name", default="main", help="name of the run (outputs in runs/<name>, results/<name>)")
    g.add_argument("--out-root", default=None, help="root for all outputs (default: repo root)")
    g.add_argument("--data-root", default=None, help="dataset root containing labelled/unlabelled folders")
    g.add_argument("--labelled-dir", default=None, help="explicit labelled (ImageFolder) directory")
    g.add_argument("--unlabelled-dir", default=None, help="explicit unlabelled directory")
    g.add_argument("--time-budget-hours", type=float, default=None,
                   help="total GPU budget for the whole pipeline (default from config, 6h)")
    g.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    return parser


def load_config(args):
    cfg = load_yaml_with_inherit(args.config)
    cfg = apply_overrides(cfg, args.set)
    d = cfg.setdefault("data", {})
    for a, k in (("data_root", "root"), ("labelled_dir", "labelled_dir"), ("unlabelled_dir", "unlabelled_dir")):
        if getattr(args, a, None):
            d[k] = os.path.abspath(getattr(args, a))
    if getattr(args, "time_budget_hours", None) is not None:
        cfg["time_budget_hours"] = float(args.time_budget_hours)
    cfg["preset"] = os.path.basename(str(args.config)).replace(".yaml", "")
    return cfg


def cfg_to_json(cfg):
    return json.dumps(cfg, indent=2, sort_keys=True, default=str)
