import os, math, json, argparse, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import VisionEncoderDecoderModel, DonutProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm
from src.data.ds_jsonl import InvoiceJsonl

def seed_all(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--processor", required=True)
    ap.add_argument("--base_model", default="naver-clova-ix/donut-base")
    ap.add_argument("--out_dir", required=True)
    # speed-friendly defaults
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--image_size", type=int, default=832)
    ap.add_argument("--freeze_encoder_epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    use_amp = device in ("cuda", "mps")

    # --- processor & model ---
    processor = DonutProcessor.from_pretrained(args.processor)
    # lock image size smaller for speed
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"height": args.image_size, "width": args.image_size}

    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # ensure tokens exist
    tok = processor.tokenizer
    added = False
    if tok.pad_token is None: tok.add_special_tokens({"pad_token":"<pad>"}); added=True
    if tok.bos_token is None: tok.add_special_tokens({"bos_token":"<s>"});   added=True
    if tok.eos_token is None: tok.add_special_tokens({"eos_token":"</s>"});  added=True
    if added: model.decoder.resize_token_embeddings(len(tok))

    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.eos_token_id           = tok.eos_token_id
    model.config.pad_token_id           = tok.pad_token_id
    model.config.vocab_size             = len(tok)

    # optional: freeze encoder initially (big speed gain)
    for p in model.encoder.parameters(): p.requires_grad = False

    model.to(device).train()

    # --- data ---
    train_ds = InvoiceJsonl(os.path.join(args.data_dir, "train.jsonl"), processor, max_len=args.max_length)
    val_ds   = InvoiceJsonl(os.path.join(args.data_dir, "val.jsonl"),   processor, max_len=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False, num_workers=0, pin_memory=True)

    # --- opt/sched ---
    total_steps = math.ceil(len(train_dl) / max(1,args.grad_accum)) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, total_steps)

    best_vloss = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        running = 0.0
        pbar = tqdm(enumerate(train_dl, 1), total=len(train_dl), desc=f"[{device}] train ep{ep}")
        optimizer.zero_grad(set_to_none=True)
        # unfreeze encoder after warm start, if requested
        if ep > args.freeze_encoder_epochs:
            for p in model.encoder.parameters(): p.requires_grad = True

        for step, batch in pbar:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                out = model(pixel_values=batch["pixel_values"].to(device),
                            labels=batch["labels"].to(device))
                loss = out.loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0:
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
            running += out.loss.item()
            pbar.set_postfix(loss=f"{running/step:.3f}")

        # --- val ---
        model.eval()
        vloss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val"):
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    vloss_sum += model(pixel_values=batch["pixel_values"].to(device),
                                       labels=batch["labels"].to(device)).loss.item()
        vloss = vloss_sum / max(1,len(val_dl))
        print(f"[epoch {ep}] val_loss={vloss:.4f}")

        # save
        ep_dir = os.path.join(args.out_dir, f"ep{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir); processor.save_pretrained(ep_dir)
        if vloss < best_vloss:
            best_vloss = vloss
            best_dir = os.path.join(args.out_dir, "best")
            model.save_pretrained(best_dir); processor.save_pretrained(best_dir)
            print(f"✓ new best → {best_dir}")

    print(f"done. best_val_loss={best_vloss:.4f}")

if __name__ == "__main__":
    main()
