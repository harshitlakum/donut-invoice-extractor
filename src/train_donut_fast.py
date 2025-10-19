import os, math, argparse, json, contextlib
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import VisionEncoderDecoderModel, DonutProcessor, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.data.ds_jsonl import InvoiceJsonl

# ---- speed/compat niceties ---------------------------------------------------
torch.set_float32_matmul_precision("medium")  # helps on Apple Silicon

def seed_all(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with train.jsonl / val.jsonl")
    ap.add_argument("--processor", required=True, help="path to saved DonutProcessor")
    ap.add_argument("--base_model", default="naver-clova-ix/donut-base")
    ap.add_argument("--out_dir", required=True)

    # Fast defaults for M2
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--image_size", type=int, default=832)
    ap.add_argument("--freeze_encoder_epochs", type=int, default=1)
    ap.add_argument("--clip_norm", type=float, default=1.0)

    # Convenience flags
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)  # macOS: avoid forking to keep tokenizers happy
    ap.add_argument("--amp", action="store_true", help="Enable AMP on CUDA only (disabled on MPS to avoid NaNs)")
    ap.add_argument("--max_train_samples", type=int, default=0, help="0=all; else use only first N training samples")
    ap.add_argument("--max_val_samples", type=int, default=0, help="0=all; else use only first N val samples")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    device = pick_device()
    # Disable AMP on MPS to avoid NaNs; allow only on CUDA via --amp
    use_amp = (args.amp and device == "cuda")

    # ---- Processor & model ----------------------------------------------------
    processor = DonutProcessor.from_pretrained(args.processor)
    if hasattr(processor, "image_processor"):
        processor.image_processor.size = {"height": args.image_size, "width": args.image_size}

    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # Ensure PAD/BOS/EOS exist; resize embeddings if we added tokens
    tok = processor.tokenizer
    added = False
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": "<pad>"}); added = True
    if tok.bos_token is None: tok.add_special_tokens({"bos_token": "<s>"});   added = True
    if tok.eos_token is None: tok.add_special_tokens({"eos_token": "</s>"});  added = True
    if added: model.decoder.resize_token_embeddings(len(tok))

    # Tell VED which IDs to use (and turn off cache for training stability/memory)
    model.config.decoder_start_token_id = tok.bos_token_id
    model.config.eos_token_id           = tok.eos_token_id
    model.config.pad_token_id           = tok.pad_token_id
    model.config.vocab_size             = len(tok)
    model.config.use_cache              = False

    # Optional warm start: freeze encoder for speed/stability on CPU/MPS
    for p in model.encoder.parameters(): p.requires_grad = False

    model.to(device).train()

    # ---- Data ----------------------------------------------------------------
    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path   = os.path.join(args.data_dir, "val.jsonl")

    train_ds = InvoiceJsonl(train_path, processor, max_len=args.max_length)
    val_ds   = InvoiceJsonl(val_path,   processor, max_len=args.max_length)

    if args.max_train_samples and args.max_train_samples > 0:
        # Wrap a subset for faster iteration
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples and args.max_val_samples > 0:
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))

    # pin_memory=False on CPU/MPS to avoid warnings; True is okay on CUDA
    pin = (device == "cuda")
    train_dl = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,  num_workers=args.workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.bsz, shuffle=False, num_workers=args.workers, pin_memory=pin)

    # ---- Optimizer / scheduler -----------------------------------------------
    # Separate weight-decay groups (no decay on LayerNorm/bias)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight"]) else no_decay).append(p)
    optimizer = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    total_steps = math.ceil(len(train_dl) / max(1, args.grad_accum)) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup, total_steps)

    # ---- Train loop -----------------------------------------------------------
    best_vloss = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        # Unfreeze encoder after warmup epochs (if requested)
        if ep > args.freeze_encoder_epochs:
            for p in model.encoder.parameters(): p.requires_grad = True

        running = 0.0
        pbar = tqdm(enumerate(train_dl, 1), total=len(train_dl), desc=f"[{device}] train ep{ep}")
        optimizer.zero_grad(set_to_none=True)

        # AMP context: on CUDA only (disabled on MPS/CPU to avoid NaNs)
        if use_amp:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        for step, batch in pbar:
            with autocast_ctx:
                out = model(pixel_values=batch["pixel_values"].to(device),
                            labels=batch["labels"].to(device))
                loss = out.loss

            # guard against NaNs/Infs early
            if not torch.isfinite(loss):
                # skip this batch; log and continue
                pbar.set_postfix_str("loss=NaN → skipped")
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                # Clip before stepping to avoid exploding grads/NaNs
                clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{running/step:.3f}")

        # ---- Validation ----
        model.eval()
        vloss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val"):
                out = model(pixel_values=batch["pixel_values"].to(device),
                            labels=batch["labels"].to(device))
                vloss_sum += float(out.loss)
        vloss = vloss_sum / max(1, len(val_dl))
        print(f"[epoch {ep}] val_loss={vloss:.4f}")

        # ---- Save checkpoints ----
        ep_dir = os.path.join(args.out_dir, f"ep{ep}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)

        if vloss < best_vloss:
            best_vloss = vloss
            best_dir = os.path.join(args.out_dir, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"✓ new best → {best_dir}")

    print(f"done. best_val_loss={best_vloss:.4f}")

if __name__ == "__main__":
    main()
