# Sweep, 2026-07-18 16:57

Corpus: 21 files · golden set: 20 questions · relevance rule: `all` · overlap: 50

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm` | 1000 | 5 | 734 | **0.627** | +0.000 | 86.7% | 4/20 |
| `minilm` | 500 | 10 | 1539 | **0.529** | -0.097 | 92.5% | 4/20 |
| `bge-small` | 1000 | 5 | 734 | **0.754** | +0.128 | 95.0% | 2/20 |
| `bge-small` | 500 | 10 | 1539 | **0.642** | +0.015 | 93.3% | 4/20 |

Baseline is row 1 (minilm · 1000 · top-5).

## What I conclude

**1. Which single change bought the most MRR?**
Switching the encoder, `minilm` → `bge-small`. It gained **+0.128** at chunk 1000 and
**+0.113** at chunk 500, so it helped in *both* chunk settings. It's the only change that
improved things at all. Chunk size never bought me anything: dropping from 1000 to 500
actually *hurt* both encoders (minilm −0.098, bge-small −0.112), so if anything the
bigger 1000-char chunk is the better setting on this corpus.

**2. Is the gap bigger than noise at n=20?**
With only 20 questions, one question flipping between a miss and a rank-1 hit moves MRR by
1/20 = 0.05, so I treat ~0.05 as "one question's worth" of noise.
- The encoder gain (~+0.11 to +0.13) is more than two questions' worth **and** it shows up
  in the same direction under both chunk sizes. That consistency makes me believe it's a
  real effect, not luck. So yes, the encoder gap clears the noise floor.
- But I would NOT read into the small gaps. `bge-small·500` (0.642) vs `minilm·1000`
  (0.627) differ by only 0.015, which is inside the noise and I can't tell those two apart.
- Caveat I'll state plainly: n=20 is small and relevance here is a keyword proxy, not real
  relevance, so even the encoder gap is "confident enough to act on," not "proven."

**3. What would I actually ship?**
`bge-small` at **chunk 1000 / top-5**, the top row of the table: best MRR (0.754) and
fewest misses (2/20). It costs the same as minilm (local, CPU, no API key), so the
encoder upgrade is essentially free. I'd keep chunks at 1000 rather than 500 because
shrinking them only hurt. Bottom line: **the lever that mattered was the encoder, not the
chunk size**, so that's the one thing I'd change.
