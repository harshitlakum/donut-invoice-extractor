# donut-invoice-extractor

OCR-free invoice extraction using **Donut** (Swin encoder + seq2seq decoder).  
This repo ships with a polished **synthetic HTML demo** you can generate in ~60 seconds (no training), plus optional code to **serve an API**, **train**, **evaluate**, and **export to TorchScript**.

---

## ✨ Highlights

- **1-Minute Synthetic Showcase** → single-file HTML gallery with invoices, fields, line-items, and Donut-style linearized targets (no training).
- **Minimal dataset loader** → JSONL with `{image, fields, line_items, target}`.
- **Optional FastAPI** inference server.
- **Lightweight trainer** (Donut finetune).
- **Evaluator** with Hungarian matching for line-items.
- **TorchScript exporter**.
- **CI + tiny unit test** (sanity).

---

## 🧰 Tech Stack

- Python · PyTorch · Hugging Face Transformers  
- FastAPI · Uvicorn  
- Pillow · SciPy · tqdm

---

## 🚀 Quick Start (Demo in ~1 minute)

> Python **3.10+** recommended.

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 2) Generate a synthetic dataset (images + JSONL)

```bash
python -m src.tools.gen_synth_invoices --n 24 --out data/converted_synth
```

### 3) Build the HTML gallery (single file with embedded images)

```bash
python -m src.tools.make_html_demo \
  --pred_jsonl data/converted_synth/val.jsonl \
  --out out/invoice_demo_synth.html \
  --title "Invoice Extraction — Synthetic Showcase"
```

### 4) Open (macOS) or serve locally

```bash
open out/invoice_demo_synth.html
# or
python -m http.server 8000
# then visit: http://localhost:8000/out/invoice_demo_synth.html
```

The HTML shows: **vendor / date / total**, a **line-item** table, and the **Donut-style linearized** text under each card.
Perfect for a **portfolio** or **walkthrough**.

---

## 🗂️ Project Structure

```
src/
  api/main.py               # FastAPI server
  data/ds_jsonl.py          # JSONL dataset (image + target | fields + line_items)
  eval/invoice_eval.py      # metrics & Hungarian matching
  export/torchscript.py     # TorchScript exporter
  train_donut_fast.py       # lightweight trainer
  tools/gen_synth_invoices.py
  tools/make_html_demo.py

configs/
scripts/
  make_synth_showcase.sh    # one-shot demo builder

tests/
  test_linearize.py         # tiny sanity test

data/
out/
```

---

## 📦 Optional: Run the API (with any Donut checkpoint)

**Serve**

```bash
# Example paths; adjust to your checkpoint directories.
export MODEL_PATH="checkpoints/donut-synth/best"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Call**

```bash
curl -F "file=@data/converted_synth/images/synth_00000.png" \
  http://localhost:8000/extract
```

**Endpoints**

* `GET /healthz`
* `POST /extract` → `{ vendor, date, total, line_items, raw }`

---

## 🧪  Train 

> Only if you want **real model outputs**; **not required** for the demo.

```bash
python -m src.train_donut_fast \
  --data_dir data/converted_synth \
  --processor checkpoints/processor \
  --out_dir checkpoints/donut-synth \
  --base_model naver-clova-ix/donut-base \
  --epochs 1 --bsz 1 --grad_accum 8 \
  --max_length 512 --image_size 832 --warmup 50
```

---

## 📊  Evaluate predictions

```bash
python -m src.eval.invoice_eval \
  --ref_jsonl data/converted_synth/val.jsonl \
  --pred_jsonl out/preds.jsonl \
  --out out/metrics.json

cat out/metrics.json
```

* **Fields**: vendor/date/total accuracy (normalized).
* **Line-items**: row-level P/R/F1 via **Hungarian matching**.

---

## 📤 Export to TorchScript

```bash
python -m src.export.torchscript \
  --ckpt checkpoints/donut-synth/best \
  --out exports/ts \
  --img 960

# → exports/ts/donut_ts.pt
```

---

## 📚 Data (how to get it)

You can demo the project with **synthetic data** (no external downloads), or pull public datasets we referenced.
**Respect each dataset’s license/terms.**

### A) Synthetic invoices (used for the showcase) — default path

```bash
python -m src.tools.gen_synth_invoices --n 24 --out data/converted_synth

python -m src.tools.make_html_demo \
  --pred_jsonl data/converted_synth/val.jsonl \
  --out out/invoice_demo_synth.html \
  --title "Invoice Extraction — Synthetic Showcase"
```

### B) DocILE (Invoices/POs; KILE + LIR annotations)

* Site: [https://docile.rossum.ai/](https://docile.rossum.ai/)
* **Access** is token-gated. Request access and receive your **personal token** via email.

**Safe usage pattern (no token leaks):**

1. Create a `.env` file (never commit it):

   ```bash
   echo "DOCILE_TOKEN=your_personal_token_here" > .env
   ```
2. Add `.env` to `.gitignore`:

   ```bash
   echo ".env" >> .gitignore
   ```
3. Use the token from the environment:

   ```bash
   source .env
   ./download_dataset.sh "$DOCILE_TOKEN" annotated-trainval data/docile --unzip
   ./download_dataset.sh "$DOCILE_TOKEN" test             data/docile --unzip
   ```
4. Quick sanity check:

   ```bash
   curl -I "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/${DOCILE_TOKEN}/annotated-trainval.zip"
   ```

**Do not** paste real tokens in code, examples, issues, or commit history.

### C) SROIE (receipts; vendor/date/total)

* Kaggle: [https://www.kaggle.com/datasets/urbikn/sroie-datasetv2](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2)
* Download (requires Kaggle CLI + API token):

  ```bash
  kaggle datasets download -d urbikn/sroie-datasetv2 -p data/sroie
  unzip data/sroie/sroie-datasetv2.zip -d data/sroie
  ```

### D) RVL-CDIP (document classification; sample “invoice” pages)

* Info: [http://www.cs.cmu.edu/~aharley/rvl-cdip/](http://www.cs.cmu.edu/~aharley/rvl-cdip/) (mirror: [https://adamharley.com/rvl-cdip/](https://adamharley.com/rvl-cdip/))
* Access typically requires agreeing to dataset terms (request/approval).
* After download, filter to invoice class and (optionally) annotate fields/line-items for supervision.

**Folder expectations**

```
data/
  converted_synth/            # synthetic (images + train/val/test .jsonl with target)
  docile/                     # if you downloaded DocILE
    annotations/ htmls/ ocr/ pdfs/
    train.json val.json test.json trainval.json
out/
  invoice_demo_synth.html     # generated demo page
```


---

## 🔧 Troubleshooting

* **HTML doesn’t open** → ensure the path is `out/invoice_demo_synth.html`.
* **Images missing in HTML** → `make_html_demo` embeds base64 from paths in the JSONL; verify those image files exist.
* **macOS font warnings** → synthetic generator falls back to PIL’s default font automatically.
* **CUDA not found** → set `CUDA_VISIBLE_DEVICES=''` to force CPU or install proper CUDA drivers.
* **Tokenizer length errors** → try `--max_length 512` and keep image `--image_size` at 832/960.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-improvement`
3. Add tests where sensible (`tests/`)
4. Run lint/test
5. Open a PR with a clear description and before/after results

---

## 🗺️ Roadmap / TODO

* [ ] Add multi-page invoice stitching for HTML demo
* [ ] Plug-and-play Donut checkpoints via config
* [ ] Expand evaluator (edit distance on linearized targets)
* [ ] Dockerfile + Compose for API
* [ ] Demo notebook (Colab)
* [ ] More fonts/templates for synthetic generator

---

## ⚖️ License

**MIT** — see `LICENSE`.

---

## 🔒 Security & Ethical Use Notice

This repository **does not distribute** any private or paid datasets.
Users **must** respect licenses and access terms for third-party datasets (e.g., **DocILE**, **RVL-CDIP**, **SROIE**).

* **Never** commit tokens, API keys, or private dataset links to Git history.
* Use environment variables (e.g., `.env`) and ensure `.env` is listed in `.gitignore`.
* If you discover a potential security or license issue, please open a minimal issue without sharing secrets.

---

## 👤 Maintainer

**Owner:** Harshit Lakum
**Contact:** harshitlakum2012 [at] gmail [dot] com

