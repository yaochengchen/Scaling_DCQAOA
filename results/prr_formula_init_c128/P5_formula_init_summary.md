# Formula-initialized refinement, 12 held-out graphs of Fig. 3 (n = 12, p = 2)

## c128: lightning.gpu, complex128, maxiter 200, ftol 1e-10, gtol 1e-05

- Formula angles: mean ratio 0.8932. Refined by L-BFGS-B from the formula: 0.9010 (gain mean 0.0078, median 0.0059, range 0.0023–0.0297); mean 20 value-and-gradient calls (99 % of the gain within 8–15 calls); all 12 chains stop with a regular convergence message.
- Mean best-so-far along the formula chain: 0.8932 (call 1), 0.8954 (5), 0.9001 (10), 0.9010 (20).
- Refined formula equals the best of the 20 random-start chains within 1e-4 on 10/12 graphs; a random chain beats it by more than 1e-4 on 2 graphs (largest excess 0.0010).
- Random starts (240 chains, campaign instance-scaled box): mean start ratio 0.5935, mean final 0.8019 (median 0.8618); 68/240 (28%) reach the fixed formula value within 0.001, 57/240 (24%) reach the refined value; mean 34 calls per chain; 0 ABNORMAL line-search stops.
- Mean best-so-far along random chains: 0.5935 (call 1), 0.7512 (10), 0.7838 (20), 0.7986 (40), 0.8014 (60).

## c64: lightning.gpu, complex64, maxiter 200, ftol 1e-06, gtol 0.001

- Formula angles: mean ratio 0.8932. Refined by L-BFGS-B from the formula: 0.9010 (gain mean 0.0078, median 0.0059, range 0.0023–0.0297); mean 30 value-and-gradient calls (99 % of the gain within 8–15 calls); all 12 chains stop with a regular convergence message.
- Mean best-so-far along the formula chain: 0.8932 (call 1), 0.8954 (5), 0.9001 (10), 0.9010 (20).
- Refined formula equals the best of the 20 random-start chains within 1e-4 on 10/12 graphs; a random chain beats it by more than 1e-4 on 2 graphs (largest excess 0.0010).
- Random starts (240 chains, campaign instance-scaled box): mean start ratio 0.5935, mean final 0.7997 (median 0.8594); 67/240 (28%) reach the fixed formula value within 0.001, 56/240 (23%) reach the refined value; mean 39 calls per chain; 6 ABNORMAL line-search stops.
- Mean best-so-far along random chains: 0.5935 (call 1), 0.7505 (10), 0.7838 (20), 0.7958 (40), 0.7993 (60).
