# Assumptions
ASK=never. Frozen science is inherited from racnet-contract-astra/ASSUMPTIONS.md, unchanged.
| ID | Interface gap | Chosen | Class | Source |
|---|---|---|---|---|
| I01 | full native source deployment | copy only original scientific modules into isolated vendor; exact original SHA verified before use | interface | frozen contract |
| I02 | label separation | input-only inference manifest physically excludes scalar, bounds, TSM; separate scorer records refer by qualified key after prediction persistence | interface | frozen contract |
| I03 | acquisition incomplete during preparation | immutable qualification snapshot from complete member indices; no new CRC/finite reads; unresolved rows PENDING; refresh snapshot only after new members arrive | interface | frozen contract |
| I04 | shared dependencies | independent venv .pth reads existing Py310 base and TransRAC overlay; no dependency installs or writes to source envs | interface | default |
| I05 | checkpoint transaction | epoch payloads saved under generation filename, commit JSON pointer atomic; conveniences latest/best point to committed generation; recovery uses commit pointer only | interface | default |
| I06 | replay accounting | durable per-step start/end journal; restart counts all starts since committed epoch as replayed attempts and separately records known successful updates / uncertain interrupted step | interface | default |
| I07 | final resume | reuse complete per-video probability+count only when binding and SHA match; fail on mismatch rather than recompute silently | interface | frozen contract |
| I08 | prepare may run between partial/completed QUVA production and continuation | extraction manifest contains only fixed raw identity, length and destination fields; generated feature SHA stays exclusively in prediction qualification | interface | sweep (R1) |
| I09 | total cost evidence across resumed invocations | completion records distinguish sums of retained original per-output durations, reused/new subsets and observed current-invocation elapsed; retained sums exclude abandoned partial work and are not wall-clock sums | interface | sweep (R2) |
