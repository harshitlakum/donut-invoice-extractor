# src/export/torchscript.py
# Exports TorchScript modules for Donut's encoder and a single decoder step.
# Why this shape? torchscript-ing .generate() is brittle; instead we export:
#   1) encoder_ts.pt:  encoder(pixel_values) -> encoder_hidden_states
#   2) decoder_step_ts.pt: decoder(input_ids, encoder_hidden_states, encoder_attention_mask) -> logits
# You can then run a greedy/beam loop in Python (or ONNX for full E2E).
#
# Usage:
#   python -m src.export.torchscript --ckpt checkpoints/donut-invoice/best --out exports/ts --img 960
#
import argparse, os, torch
from transformers import VisionEncoderDecoderModel, DonutProcessor

class EncoderWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
    def forward(self, pixel_values):
        # pixel_values: (B,3,H,W)
        out = self.enc(pixel_values=pixel_values, return_dict=True)
        # Some encoders also emit last_hidden_state only
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        # Build a full-attention mask (all ones) if encoder doesn't need it.
        attn = torch.ones(hidden.shape[:-1], dtype=torch.long, device=hidden.device)
        return hidden, attn  # (B,Seq,Dim), (B,Seq)

class DecoderStepWrapper(torch.nn.Module):
    def __init__(self, dec, lm_head):
        super().__init__()
        self.dec = dec
        self.lm_head = lm_head
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        # input_ids: (B, T) current decoder tokens
        # returns: logits over vocab for the *last* position (B, V) and full logits (B,T,V)
        out = self.dec(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,  # cache isn't TorchScript-friendly here
            return_dict=True,
        )
        hidden_states = out.last_hidden_state  # (B,T,D)
        logits_full = self.lm_head(hidden_states)  # (B,T,V)
        last_logits = logits_full[:, -1, :]       # (B,V)
        return last_logits, logits_full

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to fine-tuned checkpoint (model+processor)")
    ap.add_argument("--out", required=True, help="Output dir for TorchScript files")
    ap.add_argument("--img", type=int, default=960, help="Square image size used at training")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model/processor
    processor = DonutProcessor.from_pretrained(args.ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt).to(args.device).eval()

    # ---- Export encoder ----
    enc_wrap = EncoderWrapper(model.encoder).to(args.device).eval()
    example_img = torch.randn(1, 3, args.img, args.img, device=args.device)
    with torch.inference_mode():
        traced_enc = torch.jit.trace(enc_wrap, (example_img,), strict=False)
    enc_path = os.path.join(args.out, "encoder_ts.pt")
    traced_enc.save(enc_path)
    print(f"✓ saved encoder: {enc_path}")

    # ---- Export decoder step ----
    # Many decoders live at model.decoder; lm_head projects to vocab.
    dec = model.decoder
    lm_head = model.lm_head
    dec_wrap = DecoderStepWrapper(dec, lm_head).to(args.device).eval()

    # Build example inputs consistent with encoder output
    with torch.inference_mode():
        enc_hidden, enc_mask = enc_wrap(example_img)  # (1, Seq, D), (1, Seq)
    # minimal decoder input_ids (B, T=1) with BOS token
    bos_id = processor.tokenizer.bos_token_id
    if bos_id is None:
        # ensure BOS exists
        processor.tokenizer.add_special_tokens({"bos_token": "<s>"})
        bos_id = processor.tokenizer.bos_token_id
    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=args.device)

    # Trace decoder step
    with torch.inference_mode():
        traced_dec = torch.jit.trace(dec_wrap, (input_ids, enc_hidden, enc_mask), strict=False)
    dec_path = os.path.join(args.out, "decoder_step_ts.pt")
    traced_dec.save(dec_path)
    print(f"✓ saved decoder step: {dec_path}")

    # Optional quick smoke run
    with torch.inference_mode():
        hidden, mask = traced_enc(example_img)
        last_logits, _ = traced_dec(input_ids, hidden, mask)
    print(f"✓ smoke test ok. logits shape: {tuple(last_logits.shape)}")

if __name__ == "__main__":
    main()
