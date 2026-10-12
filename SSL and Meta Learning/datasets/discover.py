"""Flexible dataset discovery.

The assignment PDF says `unlabelled data` / `labelled data` (with spaces), the real download uses
`unlabelled_data` / `labelled_data`. We match case-insensitively after stripping spaces/_/-:
  name contains 'unlabel'                -> unlabelled set
  name contains 'label' (not 'unlabel')  -> labelled ImageFolder set
Search order: explicit dirs > --data-root > $GNR650_DATA > <repo>/data > <repo>/dataset > <repo>/*.
"""
import os
import re

from datasets.io import IMG_EXTS
from utils.paths import REPO_ROOT


def _norm(name):
    return re.sub(r"[\s_\-]+", "", name.lower())


def _is_img(fn):
    return fn.lower().endswith(IMG_EXTS) and not fn.startswith(".")


def _find_in(root):
    lab = unl = None
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return None, None
    for e in entries:
        p = os.path.join(root, e)
        if not os.path.isdir(p) or e.startswith("."):
            continue
        n = _norm(e)
        if "unlabel" in n and unl is None:
            unl = p
        elif "label" in n and "unlabel" not in n and lab is None:
            lab = p
    return lab, unl


def find_data_dirs(data_root=None, labelled_dir=None, unlabelled_dir=None, verbose=True):
    """Return (labelled_dir, unlabelled_dir); either may be None if not found."""
    if labelled_dir and unlabelled_dir:
        return os.path.abspath(labelled_dir), os.path.abspath(unlabelled_dir)
    candidates = []
    if data_root:
        candidates.append(data_root)
    if os.environ.get("GNR650_DATA"):
        candidates.append(os.environ["GNR650_DATA"])
    candidates += [os.path.join(REPO_ROOT, "data"), os.path.join(REPO_ROOT, "dataset")]
    # also one level deeper (e.g. data/Assignment2/labelled_data) and any top-level repo folder
    extra = []
    for c in list(candidates) + [REPO_ROOT]:
        if os.path.isdir(c):
            for e in sorted(os.listdir(c)):
                p = os.path.join(c, e)
                if os.path.isdir(p) and not e.startswith(".") and e not in ("runs", "cache", ".venv", "venv"):
                    extra.append(p)
    lab, unl = labelled_dir, unlabelled_dir
    found = [_find_in(c) for c in candidates + extra if os.path.isdir(c)]
    both = [(l2, u2) for l2, u2 in found if l2 and u2]
    if both:                                 # prefer the first folder containing both sets
        lab, unl = lab or both[0][0], unl or both[0][1]
    for l2, u2 in found:                     # otherwise accept partial matches
        lab, unl = lab or l2, unl or u2
    lab = os.path.abspath(lab) if lab else None
    unl = os.path.abspath(unl) if unl else None
    if verbose:
        print(f"[data] labelled dir  : {lab}")
        print(f"[data] unlabelled dir: {unl}")
    return lab, unl


def class_sort_key_info(names):
    """Class index order. If every folder name ends in an integer and those integers are unique,
    order numerically (class_2 < class_10); otherwise lexicographic (torchvision ImageFolder rule).
    For zero-padded names like class_00..class_49 both rules agree."""
    nums = [re.search(r"(\d+)$", n) for n in names]
    if all(nums) and len({int(m.group(1)) for m in nums}) == len(names):
        return sorted(names, key=lambda n: int(re.search(r"(\d+)$", n).group(1))), "numeric-suffix"
    return sorted(names), "lexicographic"


def list_labelled(labelled_dir):
    """ImageFolder listing -> (paths, labels, class_names, ordering_rule). Recurses inside class dirs."""
    classes = [d for d in os.listdir(labelled_dir)
               if os.path.isdir(os.path.join(labelled_dir, d)) and not d.startswith(".")]
    classes, rule = class_sort_key_info(classes)
    paths, labels = [], []
    for ci, c in enumerate(classes):
        for dp, dns, fns in os.walk(os.path.join(labelled_dir, c)):
            dns[:] = sorted(d for d in dns if not d.startswith("."))
            for fn in sorted(fns):
                if _is_img(fn):
                    paths.append(os.path.join(dp, fn))
                    labels.append(ci)
    return paths, labels, classes, rule


def list_unlabelled(unlabelled_dir):
    paths = []
    for dp, dns, fns in os.walk(unlabelled_dir):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        paths += [os.path.join(dp, fn) for fn in sorted(fns) if _is_img(fn)]
    return paths
