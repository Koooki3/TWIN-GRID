"""Brute-force check of the two-grid covering lemma (theory v7 §2): for even B, every span of length l <= B/2 + 1 lies in
one block of grid 0 (boundaries kB) or grid 1 (kB + B/2); some span of length B/2 + 2 lies in neither. Exit 1 on failure."""
import sys

bad = 0
for B in (4, 8, 16, 32, 64):
    def inside(x, l, off):
        return (x - off) // B == (x + l - 1 - off) // B
    for l in range(1, B // 2 + 2):
        for x in range(B, 5 * B):
            if not (inside(x, l, 0) or inside(x, l, B // 2)):
                bad += 1
    if all(inside(x, B // 2 + 2, 0) or inside(x, B // 2 + 2, B // 2) for x in range(B, 5 * B)):
        bad += 1                                  # bound not tight
print("violations", bad)
sys.exit(1 if bad else 0)
