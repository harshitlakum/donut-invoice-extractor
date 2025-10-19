# src/eval/invoice_eval.py
# Fair metrics:
#  - Fields counted ONLY when ref has a value (missing ref => skipped)
#  - Pred missing while ref present => wrong
#  - Line-items PR/F1 computed ONLY for docs with ref line-items > 0
import argparse, json, re, os
from collections import Counter
try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

def _norm_money(x):
    if x in (None, "", "null"): return None
    s = re.sub(r"[^\d.,\-]", "", str(x)).replace(",", "")
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

def _grab(tag, s):
    m = re.search(fr"<{tag}>(.*?)</{tag}>", s or "", flags=re.DOTALL);  return m.group(1).strip() if m else None

def _parse_target(target):
    if not target: return {}, []
    fields = {k: _grab(k, target) for k in ["vendor","date","total"]}
    items = []
    for m in re.finditer(r"<item>(.*?)</item>", target, flags=re.DOTALL):
        blk = m.group(1)
        items.append({
            "description": _grab("description", blk),
            "quantity":    _grab("quantity", blk),
            "unit_price":  _grab("unit_price", blk),
            "amount":      _grab("amount", blk),
        })
    return fields, items

def _cands_from_image(img_path):
    if not img_path: return set()
    base = os.path.basename(img_path)
    stem = os.path.splitext(base)[0]
    return {
        stem,
        stem.split("_p")[0],
        stem.split("-p")[0],
        re.sub(r"[-_]?p\d+$", "", stem),
        base,  # include with page suffix
    }

def _cands_from_id(doc_id):
    if not doc_id: return set()
    stem = os.path.splitext(os.path.basename(doc_id))[0]
    return {
        stem,
        stem.split("_p")[0],
        stem.split("-p")[0],
        re.sub(r"[-_]?p\d+$", "", stem),
    }

# ---------- line-item scoring ----------
_KEYS = ["description","quantity","unit_price","amount"]

def _cell_equal(k, rv, pv):
    if k in ("unit_price","amount"): return _norm_money(rv)==_norm_money(pv)
    return _norm_text(rv)==_norm_text(pv)

def _row_cost(r, p):
    hits=0; denom=0
    for k in _KEYS:
        rv=r.get(k); pv=p.get(k)
        if rv or pv:
            denom+=1
            hits += int(_cell_equal(k, rv, pv))
    return 1.0 - (hits/max(1,denom))  # 0 = perfect

def _row_prf(ref_rows, pred_rows, thr=0.25):
    # Return None if ref has no rows (to signal "skip this doc for LI stats")
    if not ref_rows:
        return None
    if not pred_rows:
        return (0.0, 0.0, 0.0)
    C = [[ _row_cost(r, p) for p in pred_rows ] for r in ref_rows]
    if linear_sum_assignment:
        r_idx, p_idx = linear_sum_assignment(C)
    else:
        used=set(); r_idx=[]; p_idx=[]
        for i,row in enumerate(C):
            best_j,best_c=None,1e9
            for j,c in enumerate(row):
                if j in used: continue
                if c < best_c: best_c,best_j=c,j
            if best_j is not None:
                used.add(best_j); r_idx.append(i); p_idx.append(best_j)
    matches = sum(1 for i,j in zip(r_idx, p_idx) if C[i][j] <= thr)
    P = matches / max(1, len(pred_rows))
    R = matches / max(1, len(ref_rows))
    F1 = 2*P*R / max(1e-9, (P+R))
    return (P, R, F1)

def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Index gold by multiple candidate keys
    gold_index = {}
    gold_docs = 0
    for ex in _load_jsonl(args.ref_jsonl):
        fields = ex.get("fields"); items = ex.get("line_items")
        if (fields is None or items is None) and ex.get("target"):
            fields, items = _parse_target(ex["target"])
        rec = {"fields": fields or {}, "line_items": items or [], "image": ex.get("image")}
        cands = set()
        if ex.get("doc_id"): cands |= _cands_from_id(ex["doc_id"])
        cands |= _cands_from_image(ex.get("image",""))
        if not cands: continue
        gold_docs += 1
        for k in cands: gold_index[k] = rec

    preds = list(_load_jsonl(args.pred_jsonl))

    # Field metrics (count only when REF has a value)
    f_hits, f_tot = Counter(), Counter()

    # Line-item metrics (only docs with ref rows)
    li_prec, li_rec, li_f1 = [], [], []
    unmatched = 0

    for p in preds:
        # find matching ref
        cands = _cands_from_id(p.get("doc_id")) | _cands_from_image(p.get("image",""))
        g = None
        for k in cands:
            if k in gold_index:
                g = gold_index[k]; break
        if g is None:
            unmatched += 1
            continue

        # fields
        for k in ["vendor","date","total"]:
            ref_v = (g.get("fields") or {}).get(k)
            if ref_v is None or ref_v == "":
                continue  # skip if ref missing
            pred_v = (p.get("fields") or {}).get(k)
            ok = False
            if k == "vendor":
                ok = (pred_v is not None and _norm_text(ref_v) == _norm_text(pred_v))
            elif k == "date":
                ok = (_norm_date(ref_v) == _norm_date(pred_v))
            else:  # total
                ok = (_norm_money(ref_v) == _norm_money(pred_v))
            f_tot[k] += 1
            f_hits[k] += int(bool(ok))

        # line-items
        res = _row_prf(g.get("line_items", []), p.get("line_items", []))
        if res is not None:
            P,R,F1 = res
            li_prec.append(P); li_rec.append(R); li_f1.append(F1)

    report = {
        "counts": {
            "gold": gold_docs,
            "preds": len(preds),
            "preds_without_match": unmatched
        },
        "fields": {
            # Note: denominators are only the count of refs where field exists
            "vendor_acc": f_hits["vendor"] / max(1, f_tot["vendor"]),
            "date_acc":   f_hits["date"]   / max(1, f_tot["date"]),
            "total_acc":  f_hits["total"]  / max(1, f_tot["total"]),
            "denominators": dict(f_tot),
        },
        "line_items": {
            "docs_scored": len(li_f1),
            "precision": sum(li_prec)/max(1, len(li_prec)),
            "recall":    sum(li_rec) /max(1, len(li_rec)),
            "f1":        sum(li_f1)  /max(1, len(li_f1)),
            "match_threshold": 0.25,
            "matcher": "hungarian" if linear_sum_assignment else "greedy-fallback"
        }
    }

    txt = json.dumps(report, indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()
