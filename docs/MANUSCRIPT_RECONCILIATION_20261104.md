# Reconcile the local manuscript and executable evidence

The five-page September local TempoRAC PDF is not numerically empty. Its SHA-256 is `c250d9aa7090ff83ef815b9fd4411af64f5c1c781df48ea0ae722159eca7feb3`. All five pages were extracted and visually inspected during this intake. The original PDF was not modified.

Page 4 reports **0.168 AvgMAE / 0.725 AvgOBO** for TempoRAC, four baseline rows and four ablation rows. Page 3 states that the experiments use **110 training / 51 validation videos**, with **268 / 134 tracks**, and that the reported evaluation is on validation. The baseline paragraph says prior-method numbers come from their original papers. The manuscript states three-seed means but does not provide per-seed predictions, standard deviations, run identities, checkpoint bindings or an exact evaluation formula.

These statements must be distinguished from the current executable repository: the audited formal campaign remains unrun and its accepted confirmatory result cells remain empty. Neither state proves the historical PDF values were fabricated or valid. They are unverified manuscript claims until the original measurements and evaluation population can be recovered. Do not delete them from the internal record or promote them into matched CVPR main-table evidence.

The local template ZIP was inspected without extraction or modification. It contains template resources, not the missing raw results. This bounded search does not establish that no evidence exists elsewhere.

There is also a specification mismatch: the PDF's displayed period equations describe masked ACF followed by FFT, while its implementation paragraph and the inspected current `cue.py` use a detached irregular-clock NUDFT. The final Method must be reconciled with the actual frozen executable method, not copied wholesale from either old draft.

The next provenance task is to locate the original per-seed outputs and metric implementation, bind their exact input population and source version, and distinguish them from unrelated supervised/PAMS history. Until that succeeds, retain the values as historical claims only. Baseline rows copied from papers must not be treated as same-input, same-split reproductions.
