# Assumption Ledger
<!-- ASK mode: never. Scientific ambiguities are escalated as one list to root/Astra. -->
| ID | Underdetermined interface or meaning | Choice | Class | Source |
|---|---|---|---|---|
| A001 | Author MMAction version has no verifiable tag | v0.24.1 compatibility candidate, not an author-environment claim | semantic | user / route decision |
| A002 | Runtime isolation without replacing live Torch | New venv reads existing Python3.10/Torch2.5.1+cu124 packages; added legacy packages only in new project path | interface | default |
| A003 | Compiler mismatch (system CUDA13 vs Torch CUDA12.4) | Stage a project-private CUDA12.4 compiler; do not bypass mismatch detection | interface | default |
| A004 | Learned default position test is invalid | Strict zero-table assertion and correct B,T,D tensor access; train remains None and author transformer permute retained | semantic | user / route decision |
| A005 | CSV and np.bool compatibility | Positional iloc, bool spelling, empty arrays reshape(0,3); no target corrections | interface | user / route decision |
| A006 | Resume plain Python warmup state absent | Persist cls_warmup_step/coef_ce_now/coef_count_now + RNG, optimizer, scheduler and zero invariant | interface | user / route decision |
| A007 | GPU checks forbidden while jobs healthy | Compilation/import CPU only; no environment-ready or performance claim | interface | user |
| A008 | Additional deprecated NumPy int aliases occur in the official frame sampler | Replace only np.int with builtin int in private vendor loading.py; preserve index arithmetic | interface | default |
| A009 | Author ROIAlign autograd backward returns five gradients for four forward inputs | Remove the extra trailing None; numerical kernels and returned input gradient remain unchanged | interface | default |
| A010 | tools/__init__.py eagerly imports analysis/deployment folders absent from released tree | Remove those unrelated eager imports so existing tools.misc.evaluation can import | interface | default |
| A011 | MMCV best checkpoint deletes old best before new checkpoint is durable | Candidate process overrides only EvalHook._save_ckpt: identical strict comparison, atomic local save, old best retained | interface | default |
