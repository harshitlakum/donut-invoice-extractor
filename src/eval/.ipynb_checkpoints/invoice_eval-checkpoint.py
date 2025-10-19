# src/eval/invoice_eval.py
# Usage:
#   python -m src.eval.invoice_eval --ref_jsonl data/converted/val.jsonl \
#                                   --pred_jsonl out/preds_val.jsonl \
#                                   --out out/metrics_val.json
import argparse, json, re, sys
from collections import Counter
from scipy.optimize import linear_sum_assignment

def _norm_money(x):
    if x in (None, "", "null"): return None
    s = re.sub(r"[^\d.,\-]", "", str(x))
    s = s.replace(",", "")  # assume comma thousands; adapt if you use EU decimals
    try: return f"{float(s):.2f}"
    except: return None

def _norm_text(x):
    return re.sub(r"\s+", " ", str(x or "")).strip().lower()

def _norm_date(x):
    if not x: return None
    try:
        from dateutil import parser
        return parser.parse(str(x), fuzzy=True, dayfirst=False).strftime("%Y-%m-%d")
    except Exception:
        return None

def _field_hits(ref_fields, pred_fields):
    rf, pf = ref_fields or {}, pred_fields or {}
    hits = {}
    hits["vendor"] = (bool(rf.get("vendor")) and bool(pf.get("vendor")) and
                      _norm_text(rf["vendor"]) == _norm_text(pf["vendor"]))
    hits["date"]   = (_norm_date(rf.get("date"))  == _norm_date(pf.get("date")))
    hits["total"]  = (_norm_money(rf.get("total"))== _norm_money(pf.get("total")))
    return hits

def _row_cost(ref_row, pred_row):
    keys = ["description", "quantity", "unit_price", "amount"]
    score, denom = 0, 0
    for k in keys:
        rv = _norm_text(ref_row.get(k))
        pv = _norm_text(pred_row.get(k))
        if rv or pv:
            denom += 1
            if k in ("unit_price", "amount"):
                rv = _norm_money(ref_row.get(k))
                pv = _norm_money(pred_row.get(k))
            score += int(rv == pv)
    return 1.0 - (score / max(1, denom))  # 0 = perfect match

def _row_prf(ref_rows, pred_rows, thr=0.25):
    if not ref_rows and not pred_rows: return (1.0, 1.0, 1.0)
    if not ref_rows or not pred_rows:  return (0.0, 0.0, 0.0)
    C = [[ _row_cost(r, p) for p in pred_rows ] for r in ref_rows]
    r_idx, p_idx = linear_sum_assignment(C)
    matches = sum(1 for i,j in zip(r_idx, p_idx) if C[i][j] <= thr)  # ≥75% col agreement
    P = matches / len(pred_rows)
    R = matches / len(ref_rows)
    F1 = 2*P*R / (P+R+1e-9)
    return (P, R, F1)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", required=True, help="gold jsonl (from converter)")
    ap.add_argument("--pred_jsonl", required=True, help="pred jsonl (decoded Donut)")
    ap.add_argument("--out", default="", help="optional path to write metrics json")
    args = ap.parse_args()

    gold = {x["doc_id"]: x for x in load_jsonl(args.ref_jsonl)}
    preds = list(load_jsonl(args.pred_jsonl))

    f_hits, f_tot = Counter(), Counter()
    row_prec, row_rec, row_f1 = [], [], []

    missed = 0
    for p in preds:
        g = gold.get(p.get("doc_id"))
        if not g:
            missed += 1
            continue
        # fields
        hits = _field_hits(g.get("fields", {}), p.get("fields", {}))
        for k, ok in hits.items():
            f_tot[k] += 1
            f_hits[k] += int(ok)
        # line-items
        P,R,F1 = _row_prf(g.get("line_items", []), p.get("line_items", []))
        row_prec.append(P); row_rec.append(R); row_f1.append(F1)

    report = {
        "counts": {
            "gold": len(gold),
            "preds": len(preds),
            "preds_without_match": missed
        },
        "fields": {
            "vendor": f_hits["vendor"]/max(1, f_tot["vendor"]),
            "date":   f_hits["date"]  /max(1, f_tot["date"]),
            "total":  f_hits["total"] /max(1, f_tot["total"]),
            "micro_F1": None  # not computed here (binary exact-match per field)
        },
        "line_items": {
            "precision": sum(row_prec)/max(1, len(row_prec)),
            "recall":    sum(row_rec) /max(1, len(row_rec)),
            "f1":        sum(row_f1)  /max(1, len(row_f1)),
            "match_threshold": 0.25
        }
    }

    txt = json.dumps(report, indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
