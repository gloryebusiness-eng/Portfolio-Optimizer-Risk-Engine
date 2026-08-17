\## D-004: Ledoit-Wolf shrinkage toward constant correlation



\*\*Decision.\*\* Implement LW shrinkage as a swappable covariance estimator

alongside the sample estimator, with a constant-correlation target.



\*\*Why.\*\* Sample covariance is unbiased but high-variance; eigenvalue bias

is worst exactly where the optimizer concentrates weight (the inverse).



\*\*Target choice.\*\* Constant-correlation rather than the identity or a

single-factor model. Identity discards the fact that equities co-move at

all. A single-factor target is defensible but imports a model assumption

I would then have to defend separately.



\*\*Measured result.\*\* On a 5-asset / 10-year sample, delta ≈ \[FILL IN] and

the condition number moved from \[FILL] to \[FILL] — a small effect. This is

expected: n >> p here. The technique earns its keep in the high-dimensional

regime, which experiment 02 tests directly.



\*\*Verification.\*\* Property tests assert symmetry, positive semi-definiteness,

delta in \[0,1], and that delta → 0 as n → ∞ with p fixed.

