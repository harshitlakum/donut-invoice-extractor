# src/api/main.py
#
# Minimal FastAPI server for Donut invoice extraction (Torch backend).
# Usage:
#   MODEL_PATH=checkpoints/donut-invoice/best uvicorn src.api.main:app --port 8000
# Test:
#   curl -F "file=@data/converted/images/<some_doc>_p1.png" http://localhost:8000/extract
#
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PIL import Image
import io, os, re, torch

from transformers import VisionEncoderDecoderModel, DonutProcessor

# ---------- Config ----------
CKPT = os.getenv("MODEL_PATH", "checkpoints/donut-invoice/best")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "768"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- App ----------
app = FastAPI(title="Invoice Extractor (Donut)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------- Model Load ----------
try:
    processor = DonutProcessor.from_pretrained(CKPT)
    model = VisionEncoderDecoderModel.from_pretrained(CKPT).to(DEVICE).eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model/processor from {CKPT}: {e}")

# ---------- Helpers ----------
TAG_FIELDS = ["vendor", "date", "total"]
ITEM_TAGS = ["description", "quantity", "unit_price", "amount"]

def _open_image_from_upload(file_bytes: bytes, filename: str) -> Image.Image:
    # Accept common image formats. (PDF support can be added later if needed.)
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported image: {filename} ({e})")

def _generate_text(img: Image.Image) -> str:
    enc = processor(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            pixel_values=enc.pixel_values.to(DEVICE),
            max_length=MAX_LENGTH,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
        )[0]
    text = processor.tokenizer.decode(out, skip_special_tokens=True)
    return text.strip()

def _grab_between(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def _parse_items(text: str) -> List[Dict[str, Optional[str]]]:
    items = []
    # Find all <item> ... </item> blocks
    for m in re.finditer(r"<item>(.*?)</item>", text, flags=re.DOTALL):
        blk = m.group(1)
        row = {t: _grab_between(blk, t) for t in ITEM_TAGS}
        items.append(row)
    return items

def parse_donut_output_to_json(text: str) -> Dict[str, Any]:
    data = {k: _grab_between(text, k) for k in TAG_FIELDS}
    data["line_items"] = _parse_items(text)
    data["raw"] = text
    return data

# ---------- Schemas ----------
class ExtractResponse(BaseModel):
    vendor: Optional[str]
    date: Optional[str]
    total: Optional[str]
    line_items: List[Dict[str, Optional[str]]]
    raw: str

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True, "device": DEVICE, "checkpoint": CKPT}

@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    content = await file.read()
    img = _open_image_from_upload(content, file.filename)
    text = _generate_text(img)
    return parse_donut_output_to_json(text)
