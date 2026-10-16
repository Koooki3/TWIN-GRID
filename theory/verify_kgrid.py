"""k-grid covering (theory v8 §6): grids offset by s = B/k (k | B). Every span of length l <= (k-1)B/k + 1 lies inside one
block of some grid; some span of length (k-1)B/k + 2 lies in none (sharp). Also: a single grid of block B/2 splits some
span of every length >= 2, and a span of length <= B/2 + 1 meets at most 2 of its blocks. Brute force; exit 1 on failure."""
import sys

bad = []


def inside(x, l, B, off):
    return (x - off) // B == (x + l - 1 - off) // B


for B in (4, 8, 16, 32, 64):
    for k in [d for d in range(1, B + 1) if B % d == 0 and d <= 8]:
        s = B // k
        L = (k - 1) * B // k + 1
        offs = [j * s for j in range(k)]
        xs = range(2 * B, 6 * B)
        for l in range(1, L + 1):
            if not all(any(inside(x, l, B, o) for o in offs) for x in xs):
                bad.append(f"B={B} k={k} l={l} not covered")
                break
        if all(any(inside(x, L + 1, B, o) for o in offs) for x in xs):
            bad.append(f"B={B} k={k}: bound not sharp")
    h = B // 2
    for l in range(2, h + 2):
        if all(inside(x, l, h, 0) for x in range(2 * B, 6 * B)):
            bad.append(f"single grid B/2 never splits length {l}")
        if max(((x + l - 1) // h) - (x // h) + 1 for x in range(2 * B, 6 * B)) > 2:
            bad.append(f"span {l} meets >2 blocks of size B/2")
print("PROBLEMS", bad[:5] if bad else "none")
sys.exit(1 if bad else 0)
