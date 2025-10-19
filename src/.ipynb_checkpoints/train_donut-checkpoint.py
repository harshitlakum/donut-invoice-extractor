# src/train_donut.py
# Minimal Donut finetuner (custom loop) + optional val decoding dump.
# Usage:
#   python -m src.train_donut \
#     --data_dir data/converted \
#     --processor checkpoints/processor \
#     --out_dir checkpoints/donut-invoice \
#     --base_model naver-clova-ix/donut-base \
#     --epochs 15 --lr 3e-5 --bsz 4 --grad_accum 4 --warmup 500 --fp16
#
import argparse, os, json, math, itertools, time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (VisionEncoderDecoderModel, DonutProcessor,
                          get_linear_schedule_with_warmup)
from tqdm import tqdm

# local dataset
from src.data.ds_jsonl import InvoiceJsonl

def seed_all(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

def decode_batch(model, processor, pixel_values, max_length=768, device="cpu"):
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values.to(device),
            max_length=max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            decoder_start_token_id=processor.tokenizer.bos_token_id,
        )
    txt = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t.strip() for t in txt]

def save_val_predictions(model, processor, ds_path, out_path, limit=None, bsz=4, device="cpu", max_length=768):
    ds = InvoiceJsonl(ds_path, processor, max_len=max_length)
    if limit: idxs = list(range(min(limit, len(ds))))
    else:     idxs = list(range(len(ds)))
    dl = DataLoader(ds, batch_size=bsz, shuffle=False, num_workers=2, pin_memory=True,
                    collate_fn=lambda b: {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
                                          "raw_target": [x["raw_target"] for x in b],
                                          "image_path": [x["image_path"] for x in b]})
    model.eval()
    preds = []
    for i, batch in enumerate(tqdm(dl, desc="decode(val)")):
        texts = decode_batch(model, processor, batch["pixel_values"], max_length=max_length, device=device)
        # crude parse back to fields/line_items (same logic as API)
        import re
        def grab(tag, s):
            m = re.search(fr"<{tag}>(.*?)</{tag}>", s, flags=re.DOTALL);  return m.group(1).strip() if m else None
        def parse_items(s):
            rows=[]
            for m in re.finditer(r"<item>(.*?)</item>", s, flags=re.DOTALL):
                blk=m.group(1)
                rows.append({
                    "description": grab("description", blk),
                    "quantity":    grab("quantity", blk),
                    "unit_price":  grab("unit_price", blk),
                    "amount":      grab("amount", blk),
                })
            return rows
        for t, imgp in zip(texts, batch["image_path"]):
            doc_id = os.path.basename(imgp).split("_p")[0]
            preds.append({
                "doc_id": doc_id,
                "fields": { "vendor": grab("vendor", t), "date": grab("date", t), "total": grab("total", t) },
                "line_items": parse_items(t),
                "raw": t
            })
        if limit and len(preds) >= limit: break
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in preds[:limit] if limit else preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(preds[:limit] if limit else preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with train.jsonl / val.jsonl")
    ap.add_argument("--processor", required=True, help="path to saved DonutProcessor")
    ap.add_argument("--base_model", default="naver-clova-ix/donut-base")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_decode_limit", type=int, default=200, help="decode at most N val samples each epoch (None=all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- Load processor/model -----
    processor = DonutProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    # resize for new tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.to(device).train()

    # ----- Datasets / loaders -----
    train_ds = InvoiceJsonl(os.path.join(args.data_dir, "train.jsonl"), processor, max_len=args.max_length)
    val_ds   = InvoiceJsonl(os.path.join(args.data_dir, "val.jsonl"),   processor, max_len=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False, num_workers=args.workers, pin_memory=True)

    # ----- Optimizer / scheduler -----
    total_steps = math.ceil(len(train_dl) / max(1,args.grad_accum)) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ----- Training loop -----
    best_vloss = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(enumerate(train_dl, 1), total=len(train_dl), desc=f"train ep{ep}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in pbar:
            with torch.cuda.amp.autocast(enabled=args.fp16):
                out = model(pixel_values=batch["pixel_values"].to(device),
                            labels=batch["labels"].to(device))
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += out.loss.item()
            pbar.set_postfix(loss=running/step)

        # ----- Quick validation loss -----
        model.eval()
        vloss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val"):
                vloss_sum += model(pixel_values=batch["pixel_values"].to(device),
                                   labels=batch["labels"].to(device)).loss.item()
        vloss = vloss_sum / max(1,len(val_dl))
        print(f"[epoch {ep}] val_loss={vloss:.4f}")

        # ----- Save epoch checkpoint -----
        ep_dir = os.path.join(args.out_dir, f"ep{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)

        # ----- Save best -----
        if vloss < best_vloss:
            best_vloss = vloss
            best_dir = os.path.join(args.out_dir, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"✓ new best saved → {best_dir}")

        # ----- Optional: dump a subset of val predictions for F1/Hungarian -----
        preds_path = os.path.join(args.out_dir, f"preds_val_ep{ep}.jsonl")
        n_dec = save_val_predictions(model.eval(), processor,
                                     os.path.join(args.data_dir, "val.jsonl"),
                                     preds_path, limit=args.val_decode_limit,
                                     bsz=max(1, args.bsz//2), device=device,
                                     max_length=args.max_length)
        print(f"→ wrote {n_dec} decoded preds to {preds_path}")

    print(f"done. best_val_loss={best_vloss:.4f}")

if __name__ == "__main__":
    main()
