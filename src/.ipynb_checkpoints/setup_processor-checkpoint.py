from transformers import DonutProcessor

tok = ["<vendor>","</vendor>","<date>","</date>","<total>","</total>",
       "<line_items>","</line_items>","<item>","</item>",
       "<description>","</description>","<quantity>","</quantity>",
       "<unit_price>","</unit_price>","<amount>","</amount>"]

p = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
p.tokenizer.add_tokens(tok, special_tokens=True)
p.save_pretrained("checkpoints/processor")
print("✅ Processor saved to checkpoints/processor")
