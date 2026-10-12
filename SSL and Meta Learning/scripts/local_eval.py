#!/usr/bin/env python
"""Local scorer (our own; grade.py is the instructors' and is never created here).

score:     python scripts/local_eval.py score --pred out/predictions.csv --gt gt.csv [--num-classes K]
           gt.csv columns id,file_path,label (label K = unknown). Reports format checks, known accuracy,
           unknown recall, (K+1)-way accuracy, balanced accuracy and a confusion summary.
make-test: python scripts/local_eval.py make-test --out-dir local_test [--holdout-classes class_03 ...]
           [--unknown-dir DIR ...] [--max-per-class 5]
           Builds test.csv + gt.csv from labelled images. Images of --holdout-classes (and of every
           --unknown-dir) are labelled K. IMPORTANT: the final model is trained on ALL labelled images, so
           this local test is optimistic for known classes (training images) and only meaningful for
           unknown detection if the hold-out classes were excluded from training. It is mainly a
           format / end-to-end check; honest numbers come from fit_ood.py's leave-class-out protocol.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    rows = list(csv.reader(text.splitlines()))
    return rows[0], rows[1:], text


def score(args):
    header, pred_rows, text = read_csv(args.pred)
    issues = []
    if header != ["id", "file_path", "label"]:
        issues.append(f"header is {header}, expected ['id', 'file_path', 'label']")
    if text.startswith("﻿"):
        issues.append("file starts with a BOM")
    gh, gt_rows, _ = read_csv(args.gt)
    gi = {h: i for i, h in enumerate(gh)}
    gt = {r[gi["id"]]: int(r[gi["label"]]) for r in gt_rows}
    gt_order = [r[gi["id"]] for r in gt_rows]
    K = args.num_classes
    meta = os.path.join(os.path.dirname(os.path.abspath(args.gt)), "meta.json")
    if K is None and os.path.isfile(meta):
        K = int(json.load(open(meta))["K"])
        print(f"[eval] K={K} from {meta}")
    if K is None and os.path.isfile(args.checkpoint):
        import torch
        K = int(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["num_classes"])
        print(f"[eval] K={K} from {args.checkpoint}")
    if K is None:
        K = max(gt.values())
        print(f"[eval] WARNING: K unknown; assuming K = max gt label = {K} (only right if gt contains unknowns)")
    labels = {}
    for r in pred_rows:
        try:
            labels[r[0]] = int(r[2])
        except (ValueError, IndexError):
            issues.append(f"non-integer label in row {r}")
    if [r[0] for r in pred_rows] != gt_order[:len(pred_rows)] or len(pred_rows) != len(gt_order):
        issues.append("ids/row order differ from the ground truth CSV (or row count differs)")
    bad = [l for l in labels.values() if not (0 <= l <= K)]
    if bad:
        issues.append(f"{len(bad)} labels outside 0..K={K}")
    ids = [i for i in gt_order if i in labels]
    known = [i for i in ids if gt[i] < K]
    unk = [i for i in ids if gt[i] == K]
    kacc = sum(labels[i] == gt[i] for i in known) / max(1, len(known))
    urec = sum(labels[i] == K for i in unk) / max(1, len(unk))
    allacc = sum(labels[i] == gt[i] for i in ids) / max(1, len(ids))
    known_rej = sum(labels[i] == K for i in known) / max(1, len(known))
    conf = Counter((gt[i] if gt[i] < K else "UNK", "UNK" if labels[i] == K else ("ok" if labels[i] == gt[i] else "wrong"))
                   for i in ids)
    res = {"n": len(ids), "n_known": len(known), "n_unknown": len(unk), "K": K,
           "kplus1_accuracy": allacc, "known_accuracy": kacc, "unknown_recall": urec,
           "balanced_accuracy": 0.5 * (kacc + urec) if unk and known else None,
           "known_rejected_as_unknown": known_rej, "format_issues": issues}
    print(json.dumps(res, indent=2))
    print("[eval] confusion summary (gt -> prediction outcome):")
    worst = sorted({k for k, _ in conf}, key=lambda c: -sum(v for (g, o), v in conf.items() if g == c and o != "ok"))
    for c in worst[:15]:
        d = {o: conf[(c, o)] for o in ("ok", "wrong", "UNK") if conf[(c, o)]}
        print(f"   gt={c}: {d}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
    if issues:
        print("[eval] FORMAT ISSUES:\n  - " + "\n  - ".join(issues))
        sys.exit(1)


def make_test(args):
    from datasets.discover import find_data_dirs, list_labelled, list_unlabelled
    lab, _ = find_data_dirs(args.data_root, args.labelled_dir, None)
    paths, labels, classes, _ = list_labelled(lab)
    K = len(classes)
    hold = set(args.holdout_classes or [])
    rows, per = [], Counter()
    for p, l in zip(paths, labels):
        if args.max_per_class and per[l] >= args.max_per_class:
            continue
        per[l] += 1
        rows.append((p, K if classes[l] in hold else l))
    for d in args.unknown_dir or []:
        for p in list_unlabelled(d)[: args.max_unknown]:
            rows.append((p, K))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "file_path"])
        w.writerows([(i, os.path.abspath(p)) for i, (p, _) in enumerate(rows)])
    with open(os.path.join(args.out_dir, "gt.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "file_path", "label"])
        w.writerows([(i, os.path.abspath(p), l) for i, (p, l) in enumerate(rows)])
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"K": K, "class_names": classes, "holdout_classes": sorted(hold)}, f, indent=1)
    print(f"[eval] wrote {len(rows)} rows ({sum(l == K for _, l in rows)} unknown) to {args.out_dir}; K={K}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--num-classes", type=int, default=None,
                   help="K (default: <gt dir>/meta.json, else --checkpoint, else max gt label)")
    s.add_argument("--checkpoint", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "checkpoints", "final.pt"))
    s.add_argument("--json-out", default=None)
    m = sub.add_parser("make-test")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--data-root", default=None)
    m.add_argument("--labelled-dir", default=None)
    m.add_argument("--holdout-classes", nargs="*", default=None)
    m.add_argument("--unknown-dir", nargs="*", default=None)
    m.add_argument("--max-per-class", type=int, default=None)
    m.add_argument("--max-unknown", type=int, default=1000)
    a = ap.parse_args()
    score(a) if a.cmd == "score" else make_test(a)


if __name__ == "__main__":
    main()
