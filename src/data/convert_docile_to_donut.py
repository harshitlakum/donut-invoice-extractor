import argparse, json, os, pathlib
try:
    import pymupdf as pymupdf      # if your wheel exposes this
except ModuleNotFoundError:
    import fitz as pymupdf         # PyMuPDF’s stable import name

def render_pdf(pdf_path: pathlib.Path, out_dir: pathlib.Path, stem: str):
    doc = pymupdf.open(str(pdf_path))
    paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap()
        img_path = out_dir / f"{stem}_{i:03d}.png"
        pix.save(str(img_path))
        paths.append(str(img_path))
    doc.close()
    return paths



def load_label(ann_dir: pathlib.Path, doc_id: str):
    """Load minimal annotation text for given document ID."""
    ann_file = ann_dir / f"{doc_id}.json"
    if ann_file.exists():
        try:
            data = json.loads(ann_file.read_text())
            kv = data.get("kvs") or data.get("fields") or {}
            if isinstance(kv, dict):
                items = [f"{k}: {v}" for k, v in kv.items()]
            elif isinstance(kv, list):
                items = [f"{it.get('key','')}: {it.get('value','')}" for it in kv]
            else:
                items = []
            return " ; ".join(items) or json.dumps(data, ensure_ascii=False)
        except Exception:
            return ""
    return ""


def write_jsonl(lines, path):
    """Write a list of dicts to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to data/docile")
    ap.add_argument("--out", required=True, help="Output directory (e.g., data/converted)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    out = pathlib.Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    pdfs = root / "pdfs"
    anns = root / "annotations"

    for split in args.splits:
        ids_path = root / f"{split}.json"
        if not ids_path.exists():
            print(f"⚠️  Missing {split}.json — skipping.")
            continue

        ids = json.loads(ids_path.read_text())
        out_lines = []
        for doc_id in ids:
            pdf_path = pdfs / f"{doc_id}.pdf"
            if not pdf_path.exists():
                print(f"⚠️  Missing PDF for {doc_id}")
                continue

            rendered = render_pdf(pdf_path, img_dir, doc_id)
            label_text = load_label(anns, doc_id)
            for p in rendered:
                out_lines.append({
                    "image": p,           # image path
                    "gt_parse": label_text  # text / label for Donut
                })

        out_file = out / f"{split}.jsonl"
        write_jsonl(out_lines, out_file)
        print(f"✅ {split}: {len(out_lines)} entries written to {out_file}")


if __name__ == "__main__":
    main()
