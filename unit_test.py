from transformers import AutoTokenizer
from models.modeling_llama import LlamaForCausalLM

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
m = LlamaForCausalLM.custom_from_pretrained("meta-llama/Llama-3.2-1B")

x = tok("Hello, world!", return_tensors="pt")
print(tok.decode(m.generate(**x, max_new_tokens=10)[0]))
