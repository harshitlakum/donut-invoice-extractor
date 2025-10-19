Donut Invoice Extractor — Showcase

OCR-free invoice extraction using Donut (Swin encoder + seq2seq decoder).
This repo ships with a polished synthetic HTML demo you can generate in ~60 seconds (no training), plus optional code to train, evaluate, export, and serve an API.

⸻

✨ What’s inside
	•	Synthetic showcase generator → single-file HTML gallery with invoices, fields, line-items, and Donut-style linearized targets.
	•	Minimal dataset loader (JSONL with image, fields, line_items, target).
	•	FastAPI inference API (optional).
	•	Lightweight trainer (optional).
	•	Evaluator with Hungarian matching for line-items (optional).
	•	TorchScript exporter (optional).
	•	Tiny CI and a unit test.

⸻

🚀 1-Minute Demo (no training)

# 1) Setup (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Generate a synthetic dataset (images + JSONL)
python -m src.tools.gen_synth_invoices --n 24 --out data/converted_synth

# 3) Build the HTML gallery (single file with embedded images)
python -m src.tools.make_html_demo \
  --pred_jsonl data/converted_synth/val.jsonl \
  --out out/invoice_demo_synth.html \
  --title "Invoice Extraction — Synthetic Showcase"

# 4) Open (macOS)
open out/invoice_demo_synth.html
# or view via http://localhost:8000/out/invoice_demo_synth.html
python -m http.server 8000

The HTML shows: vendor / date / total, a line-item table, and the Donut-style linearized text under each card. Perfect for a portfolio or walkthrough.

⸻

📦 Optional: Use the API (with any Donut checkpoint)

# Serve
MODEL_PATH=checkpoints/donut-synth/best \
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Call
curl -F "file=@data/converted_synth/images/synth_00000.png" \
     http://localhost:8000/extract

	•	Endpoint: POST /extract → {vendor, date, total, line_items, raw}
	•	Health: GET /healthz

⸻

🧪 Optional: Train on the synthetic set

Only if you want real model outputs; not required for the demo.

python -m src.train_donut_fast \
  --data_dir data/converted_synth \
  --processor checkpoints/processor \
  --out_dir checkpoints/donut-synth \
  --base_model naver-clova-ix/donut-base \
  --epochs 1 --bsz 1 --grad_accum 8 \
  --max_length 512 --image_size 832 --warmup 50


⸻

📊 Optional: Evaluate predictions

python -m src.eval.invoice_eval \
  --ref_jsonl data/converted_synth/val.jsonl \
  --pred_jsonl out/preds.jsonl \
  --out out/metrics.json

cat out/metrics.json

	•	Fields: vendor/date/total accuracy (normalized).
	•	Line-items: row-level P/R/F1 via Hungarian matching.

⸻

📤 Optional: Export to TorchScript

python -m src.export.torchscript \
  --ckpt checkpoints/donut-synth/best \
  --out exports/ts \
  --img 960
# → exports/ts/donut_ts.pt


⸻

🗂️ Project structure

src/
  api/main.py                 # FastAPI server
  data/ds_jsonl.py            # JSONL dataset (image + target | fields + line_items)
  eval/invoice_eval.py        # metrics & Hungarian matching
  export/torchscript.py       # TorchScript exporter
  train_donut_fast.py         # lightweight trainer 

configs/
scripts/
  make_synth_showcase.sh      # one-shot demo builder
tests/
  test_linearize.py           # tiny sanity test

data
 
⸻

📚 Data (how to get it)

You can demo the project with synthetic data (no external downloads), or pull public datasets we referenced.

1) Synthetic invoices (used for the showcase)

No external data. We render invoices + ground-truth JSON ourselves.

python -m src.tools.gen_synth_invoices --n 24 --out data/converted_synth
python -m src.tools.make_html_demo \
  --pred_jsonl data/converted_synth/val.jsonl \
  --out out/invoice_demo_synth.html \
  --title "Invoice Extraction — Synthetic Showcase"

2) DocILE (Invoices/POs; KILE + LIR annotations)
    •    Site: https://docile.rossum.ai/
    •    Access is token-gated. Use the personal token you receive via email.

Download method we used:

# replace with your FULL token (30–40 chars)
TOKEN='cbbd81af...full_token_here...'
./download_dataset.sh "$TOKEN" annotated-trainval data/docile --unzip
./download_dataset.sh "$TOKEN" test               data/docile --unzip

# quick sanity check (should NOT return 403/404)
curl -I "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com/$TOKEN/annotated-trainval.zip"

3) SROIE (receipts; vendor/date/total)
    •    Kaggle: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
Download (requires Kaggle CLI + API token):

kaggle datasets download -d urbikn/sroie-datasetv2 -p data/sroie
unzip data/sroie/sroie-datasetv2.zip -d data/sroie

4) RVL-CDIP (document classification; we sample “invoice” pages)
    •    Info: http://www.cs.cmu.edu/~aharley/rvl-cdip/ (mirror: https://adamharley.com/rvl-cdip/)
Access requires agreeing to the dataset terms (request/approval). After download, filter to invoice class and (optionally) annotate fields/line-items for supervision.

⸻

Folder expectations

data/
  converted_synth/            # synthetic (images + train/val/test .jsonl with target)
  docile/                     # if you downloaded DocILE
    annotations/ htmls/ ocr/ pdfs/
    train.json val.json test.json trainval.json
out/
  invoice_demo_synth.html     # generated demo page

⚠️ Respect each dataset’s license/terms. Synthetic data here is generated locally and free to use for demos.                        



⸻

🔧 Troubleshooting
	•	HTML doesn’t open: ensure the path is out/invoice_demo_synth.html.
	•	Images missing in HTML: make_html_demo embeds base64 from the image path in the JSONL; verify those files exist.
	•	macOS font warnings: the generator falls back to PIL’s default font automatically.

⸻

🧰 Requirements

See requirements.txt. Key libs: torch, transformers, fastapi, Pillow, scipy, tqdm.

⸻

☁️ Push to GitHub

git init
git add -A
git commit -m "feat: initial invoice-extractor showcase"
git branch -M main
# create an empty GitHub repo, then:
git remote add origin git@github.com:YOURUSER/invoice-extractor.git
git push -u origin main


⸻

⚖️ License

MIT — see LICENSE.

⸻

🙋 FAQ

Q: Are we generating images from text?
A: No — the demo renders synthetic invoice images and shows the structured fields + linearized tags. The (optional) model performs image → text/JSON extraction.
