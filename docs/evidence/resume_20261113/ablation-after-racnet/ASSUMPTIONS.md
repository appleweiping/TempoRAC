# Assumptions (ASK=never)
|ID|Interface decision|Choice|Source|
|--|--|--|--|
|A1|Canonical RAC trajectory|Match canonical launch release SHA to exactly one finite-pipeline receipt; do not choose latest by timestamp.|user|
|A2|Root execution authorization|--execute additionally requires root release with entry SHA, exact queue, GPU UUID, pipeline receipt SHA and startup expiry. --check is read-only.|default|
|A3|Completed work reuse|No rehash of old ablation payloads; verify old prefix receipt provenance, then unchanged worker --resume retains its own authoritative completion validation. RAC metadata identity uses existing committed checkpoint SHA and final evaluation binding; checkpoint files checked for existence/bytes, no torch loading.|user/default|
|A4|Storage|Reuse existing shared128GiB reserve; no additional scientific budget or matrix.|user|
No semantic choices: scientific contract dc60, source489, worker d23 unchanged. Any failed stage/partial residual blocks; no restart.
