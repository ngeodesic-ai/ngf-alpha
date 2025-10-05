from transformers import AutoTokenizer, AutoModelForCausalLM
from wdd_insert import insert_wdd_block

model_name = "gpt2"
tok   = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Insert after tap -9, stronger push for visibility
model, at = insert_wdd_block(
    model,
    after_block_idx=-9,
    gain_scale=6.0,        # turn it up while testing
    rel_clip=0.95,         # allow larger relative step
    last_k=16,             # affect a short span of tokens
    priors_json="stage11_summary_priors.json"
)

# IMPORTANT while bringing up: avoid cache expecting KV from custom block
gen_kwargs = dict(max_new_tokens=64, do_sample=False, use_cache=False)

prompt = "In a 3×3 grid, each row increases by +2..."
ids = tok(prompt, return_tensors="pt")
out = model.generate(**ids, **gen_kwargs)
print(tok.decode(out[0], skip_special_tokens=True))
