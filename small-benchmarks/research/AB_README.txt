A/B scoring helper

1) Score a single run:
   python3 /mnt/data/arc_scorer.py      --generations /mnt/data/generations_stock.v4b.tap9.jsonl      --truths /mnt/data/<truths>.csv      --out /mnt/data/metrics_stock.v4b.tap9.json      --matches_csv /mnt/data/matches_stock.csv

2) Build the A/B scoreboard:
   python3 /mnt/data/ab_scoreboard.py      --stock /mnt/data/generations_stock.v4b.tap9.jsonl      --geo   /mnt/data/generations_geo_steps.v4b.tap9.jsonl      --truths /mnt/data/<truths>.csv      --out   /mnt/data/ab_scoreboard.v4b.tap9.json
