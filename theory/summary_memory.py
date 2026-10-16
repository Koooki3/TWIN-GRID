"""Analytical summary memory per scorer (floats per KV head), per block and per token, and relative to the KV cache
(2d floats per token per KV head). Two columns (reviewer batch 7): the MINIMAL sufficient statistics each scorer's
formula needs, and what the CURRENT implementation (src/sparse_attn.py) actually caches.
Minimal statistics:
  quest   : per-channel min and max of the keys                             2d
  em      : mean + min + max                                                3d
  mean    : mean key (MoBA-style mean pooling)                              d
  cobs    : mean (d) + covariance of the r-dim projected keys r(r+1)/2      d + r(r+1)/2
  nf_pt1  : 1 exact outlier key (d) + bulk mean (d) + bulk box min/max (2d) + bulk projected covariance r(r+1)/2
The shared projection basis W (d x r per KV head per layer) is amortised over all blocks and listed apart.
Current implementation caches (per block): cobs keeps the centred projection of every token (B*r) instead of the
r x r covariance; nf_pt1 likewise keeps B*r bulk projections plus the outlier index (m); quest / em / mean as minimal.
TWIN-GRID (+g2f) doubles the number of blocks per token.
Usage: summary_memory.py [d] [r]"""
import sys

d = int(sys.argv[1]) if len(sys.argv) > 1 else 128          # Llama-3.1-8B and Qwen2.5 head dim
r = int(sys.argv[2]) if len(sys.argv) > 2 else 8            # cobs / nf rank (plan protocol --rank 8)
tri = r * (r + 1) // 2
minimal = {"quest": (2 * d, "min+max 2d"), "em": (3 * d, "mean+min+max 3d"), "mean": (d, "mean d"),
           "cobs": (d + tri, f"mean d + proj. cov r(r+1)/2 = {d}+{tri}"),
           "nf_pt1": (4 * d + tri, f"outlier d + bulk mean d + bulk box 2d + bulk proj. cov = {d}+{d}+{2 * d}+{tri}")}


def impl(s, B):
    return {"quest": 2 * d, "em": 3 * d, "mean": d, "cobs": d + B * r, "nf_pt1": 4 * d + B * r + 1}[s]


arms = [("quest16", "quest", 16, 1), ("quest32", "quest", 32, 1), ("quest32+g2f", "quest", 32, 2), ("em", "em", 32, 1),
        ("mean", "mean", 32, 1), ("mean+g2f", "mean", 32, 2), ("cobs", "cobs", 32, 1), ("cobs+g2f", "cobs", 32, 2),
        ("cobs16", "cobs", 16, 1), ("nf_pt1", "nf_pt1", 32, 1), ("nf_pt1+g2f", "nf_pt1", 32, 2)]
kv = 2 * d
print(f"d={d}, r={r}; shared projection basis W: {d * r} floats per KV head per layer (not per block)")
for s, (v, how) in minimal.items():
    print(f"  minimal per block {s:7s} {v:5d} = {how}")
print(f"{'arm':12s} {'block':>5s} {'grids':>5s} {'min/token':>10s} {'vs q32':>7s} {'vs q16':>7s} {'%KV':>6s} "
      f"{'impl/token':>10s}")
q32 = minimal["quest"][0] / 32
for name, s, B, g in arms:
    ft = minimal[s][0] * g / B
    it = impl(s, B) * g / B
    print(f"{name:12s} {B:5d} {g:5d} {ft:10.2f} {ft / q32:7.2f} {ft / (2 * q32):7.2f} {100 * ft / kv:5.1f}% {it:10.2f}")
