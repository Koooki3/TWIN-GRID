## Accuracy (mean, 95% bootstrap CI over samples)

### ctx=32768 budget=1024  (n: multikey=20, multivalue=20, single_essay=20)

| method | multikey | multivalue | single_essay | all |
|---|---|---|---|---|
| dense | 0.95 [0.85,1.00] | 0.90 [0.84,0.96] | 1.00 [1.00,1.00] | 0.95 [0.90,0.98] |
| quest | 0.75 [0.55,0.95] | 0.44 [0.30,0.60] | 0.70 [0.50,0.90] | 0.63 [0.52,0.74] |
| cobs | 0.70 [0.50,0.90] | 0.31 [0.23,0.41] | 0.90 [0.75,1.00] | 0.64 [0.53,0.74] |
| sum_dual | 0.85 [0.70,1.00] | 0.74 [0.59,0.89] | 1.00 [1.00,1.00] | 0.86 [0.78,0.94] |

Gate (paired over all samples, ctx=32768, budget=1024): dual − best single channel
- summary: dual=0.863 mass=0.637 peak=0.629; dual−cobs=+0.225 [+0.125,+0.333]

## Offline stats at the first decode step (layers >= 2)

### ctx=32768 budget=1024

| method | needle-mass retained | mass captured | rel. output error |
|---|---|---|---|
| quest | 0.953 | 0.9041 | 0.1451 |
| cobs | 0.968 | 0.9194 | 0.1197 |
| sum_dual | 0.978 | 0.9177 | 0.1235 |

### budget=1024 (n=60: multikey=20, multivalue=20, single_essay=20)

| method | multikey | multivalue | single_essay | all | − sum_dual [95% CI] |
|---|---|---|---|---|---|
| quest | 0.75 | 0.44 | 0.70 | 0.629 | -0.233 [-0.342,-0.129] |
| cobs | 0.70 | 0.31 | 0.90 | 0.637 | -0.225 [-0.333,-0.125] |
| sum_dual | 0.85 | 0.74 | 1.00 | 0.863 | +0.000 [+0.000,+0.000] |

