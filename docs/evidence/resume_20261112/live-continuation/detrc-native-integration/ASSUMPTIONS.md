# Assumptions and interfaces

| ID | Choice before implementation | Class | Source |
|---|---|---|---|
| I1 | Default JPEG/extraction plan conservatively1244, required1243 excludes only missing-count stu6_44 from optimization; retain all357 eval rows | interface | frozen route |
| I2 | ROI uses inherited0-based inclusive frame_range, renumber JPEG1..T; no spatial crop beyond author short256 resize | semantic | frozen route/manifest |
| I3 | Official OpenCV JPEG encoding and exact resize integer arithmetic; producer can run on CPU before GPU is free | interface | parent |
| I4 | Per-video durable completion; incomplete video replays, completed SHA-bound artifacts reused | interface | default |
| I5 | Complete raw JPEG retention under shared64GiB derivative cap; if cap/reserve fails stop without sampling/deletion | interface | frozen budget / default retention |
| I6 | RC131 common diagnostic evaluated once on native130-val-OBO-selected weight; does not influence selection | semantic | frozen route |
| I7 | Empty(0,3) structural GT placeholders only; labels reside solely in separate scorer file/process | interface | frozen route |
| I8 | One24GiB-class GPU >=24000MiB visible, free<500MiB/no compute process; no sharing or microbatch/AMP fallback | interface | frozen resource boundary |
| I9 | GPU occupancy budgets16h extraction+30h train/val+2h final48h, persisted across resumes; CPU JPEG time reported separately | interface | frozen budget |
| I10 | Root-owned release artifact binds contract/code hash and stage; no production starts during preparation | interface | parent |
| I11 | Space gate reserves128GiB plus RACnet author128629792634 bytes minus current STATE durable coverage; refresh at least each60seconds, never sparse logical sizes or network totals | interface | latest parent instruction |
