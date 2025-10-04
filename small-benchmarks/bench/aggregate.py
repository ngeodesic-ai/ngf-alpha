import json, glob, statistics as st
def load(pattern): return [json.load(open(p)) for p in sorted(glob.glob(pattern))]
def avg_std(xs): return {"mean": sum(xs)/len(xs), "std": (st.pstdev(xs) if len(xs)>1 else 0.0)}

# Load per-track/per-method JSONs you produced
sim_latent = load("bench/runs/sim_latent_arc_seed*.json")
micro_latent = load("bench/runs/micro_latent_arc_seed*.json")  # if you emit these
sim_report = load("bench/runs/sim_report_seed*.json")

# Pick the fields you care about
fields = ["accuracy_exact","precision","recall","f1","hallucination_rate","omission_rate"]

def summarize(blobs):
    out = {}
    for f in fields:
        vals = [b.get(f) for b in blobs if f in b]
        if vals:
            out[f] = avg_std(vals)
    return out

consolidated = {
  "latent_arc": {
    "sim_denoiser": summarize(sim_latent),
    "micro_llm":    summarize(micro_latent),
  },
  "report_path": {
    # If your sim JSON splits by method internally, you can keep all:
    # Otherwise store whatever the file gives you (stock/geodesic/denoiser).
    "sim": summarize(sim_report)
  },
  "seeds": len(sim_latent)
}
json.dump(consolidated, open("bench/consolidated_benchmark.json","w"), indent=2)
print("Wrote bench/consolidated_benchmark.json")
