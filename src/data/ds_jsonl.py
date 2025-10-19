# src/data/ds_jsonl.py
from torch.utils.data import Dataset
from PIL import Image
import json, os, re

SPECIAL_TAGS = [
    "<vendor>","</vendor>","<date>","</date>","<total>","</total>",
    "<line_items>","</line_items>","<item>","</item>",
    "<description>","</description>","<quantity>","</quantity>",
    "<unit_price>","</unit_price>","<amount>","</amount>"
]

def _norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _linearize_from_fields(fields: dict, line_items: list) -> str:
    """Build Donut target string deterministically from fields + line_items."""
    f = fields or {}
    parts = []
    parts.append(f"<vendor>{_norm_text(f.get('vendor'))}</vendor>")
    parts.append(f"<date>{_norm_text(f.get('date'))}</date>")
    parts.append(f"<total>{_norm_text(f.get('total'))}</total>")
    parts.append("<line_items>")
    for li in (line_items or []):
        parts.append("<item>")
        parts.append(f"<description>{_norm_text(li.get('description'))}</description>")
        parts.append(f"<quantity>{_norm_text(li.get('quantity'))}</quantity>")
        parts.append(f"<unit_price>{_norm_text(li.get('unit_price'))}</unit_price>")
        parts.append(f"<amount>{_norm_text(li.get('amount'))}</amount>")
        parts.append("</item>")
    parts.append("</line_items>")
    return " ".join(parts)

class InvoiceJsonl(Dataset):
    """
    Minimal Donut-style dataset.

    Each JSONL line should contain:
      - "image": path to PNG
      - EITHER "target": linearized tagged string
        OR     {"fields": {...}, "line_items":[...]} (we'll synthesize target)

    Returns:
      {"pixel_values": tensor, "labels": ids, "raw_target": str, "image_path": str}
    """
    def __init__(self, jsonl_path, processor, max_len=768,
                 image_key="image", target_key="target"):
        self.items = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
        self.p = processor
        self.max_len = max_len
        self.image_key = image_key
        self.target_key = target_key

        # ensure BOS/EOS/PAD exist
        changed = False
        if self.p.tokenizer.bos_token is None:
            self.p.tokenizer.add_special_tokens({"bos_token": "<s>"}); changed = True
        if self.p.tokenizer.eos_token is None:
            self.p.tokenizer.add_special_tokens({"eos_token": "</s>"}); changed = True
        if self.p.tokenizer.pad_token is None:
            self.p.tokenizer.add_special_tokens({"pad_token": "<pad>"}); changed = True
        # ensure special tags are in vocab
        self.p.tokenizer.add_tokens([t for t in SPECIAL_TAGS if t not in self.p.tokenizer.get_vocab()], special_tokens=True)

        if changed:
            # caller must have resized decoder embeddings after loading base model
            pass

    def __len__(self):
        return len(self.items)

    def _load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")

    def _get_target_text(self, it):
        if self.target_key in it and it[self.target_key]:
            return it[self.target_key]
        # synthesize from fields + line_items
        return _linearize_from_fields(it.get("fields", {}), it.get("line_items", []))

    def __getitem__(self, idx):
        it = self.items[idx]
        img = self._load_image(it[self.image_key])

        enc = self.p(img, return_tensors="pt")
        tgt = self._get_target_text(it)
        tgt = f"{self.p.tokenizer.bos_token}{tgt}{self.p.tokenizer.eos_token}"

        labels = self.p.tokenizer(
            tgt,
            add_special_tokens=False,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.p.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": enc.pixel_values.squeeze(0),
            "labels": labels,
            "raw_target": tgt,
            "image_path": it[self.image_key],
        }

