import torch
from transformers import AutoTokenizer
from models.modeling_llama import LlamaForCausalLM

model_id = "meta-llama/Llama-3.2-1B"

tok = AutoTokenizer.from_pretrained(model_id)
m = LlamaForCausalLM.custom_from_pretrained(
    model_id,
    cache_dir="/home/nnonta/.cache/huggingface",
    torch_dtype=torch.float32,           # keep it simple for a quick test
)
m.config.use_cache = False               # ← important
if tok.pad_token_id is None:             # silence the pad/eos warning
    tok.pad_token = tok.eos_token

m.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
m.to(device)

with torch.no_grad():
    x = tok("Hello, world!", return_tensors="pt").to(device)
    out_ids = m.generate(
        **x,
        max_new_tokens=10,
        do_sample=False,
        use_cache=False,                 # ← belt-and-suspenders (per-call)
    )[0]

print(tok.decode(out_ids, skip_special_tokens=True))

