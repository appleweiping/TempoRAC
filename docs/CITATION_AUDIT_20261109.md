# Citation audit — `paper/cvpr27/main.bib` (26 entries)

Checked 2026-11-08 against CVF Open Access pages (BibTeX blocks), Crossref DOI records, publisher pages
(Springer, IOS Press), arXiv API/HTML and authors' pages. Read-only on the repo apart from this file;
`main.bib` and the `.tex` files were not changed. "OK" means title, authors and their order, venue, year
and any pages given all match the source. 9 entries are not cited in any `.tex` file and do not appear in
the References: tang2024multicounter, tang2026multicounterplus, luiten2021hota, ristani2016idmetrics,
welch1967fft, lomb1976unequal, griffin1984stft, shazeer2017moe and jacobs1991adaptive.

## 1. Entry table

| key | status | what is wrong | corrected / suggested BibTeX fields | source URL |
|---|---|---|---|---|
| levy2015live | OK | — | — | https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html ; https://doi.org/10.1109/ICCV.2015.346 |
| runia2018realworld | OK | — | — | https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html ; https://doi.org/10.1109/CVPR.2018.00939 |
| dwibedi2020repnet | OK (note) | Pages 10384–10393 are the IEEE Xplore/DOI pagination (Crossref); CVF Open Access shows pp. 10387–10396. The `url` points to CVF but the pages follow Xplore. That is consistent with the DOI, not an error. | optional: keep as is (DOI-consistent) | https://openaccess.thecvf.com/content_CVPR_2020/html/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.html ; https://api.crossref.org/works/10.1109/CVPR42600.2020.01040 |
| hu2022transrac | OK (note) | Same situation: pages 18991–19000 = Xplore/DOI; CVF shows pp. 19013–19022. The CVF title capitalises "Multi-Scale … With"; the bib follows Xplore ("Multi-scale … with"). | optional: keep as is (DOI-consistent) | https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html ; https://api.crossref.org/works/10.1109/CVPR52688.2022.01843 |
| tang2024multicounter | OK | — (FAIA vol. 392, pp. 242–249, 6 authors in this order) | — | https://ebooks.iospress.nl/doi/10.3233/FAIA240494 |
| yao2025poserac | OK (note) | Matches the ICONIP 2024 version of record (LNCS 15290, Part V, pp. 255–270, Yao & Zou). The arXiv preprint 2303.08450 has a different title ("PoseRAC: Pose Saliency Transformer for Repetitive Action Counting") and 3 authors (Yao, Cheng, Zou). The 0.236/0.560 in `tab/main.tex` was not re-checked against the ICONIP chapter text (paywalled, not accessed). | — | https://link.springer.com/chapter/10.1007/978-981-96-6588-4_18 ; https://link.springer.com/book/10.1007/978-981-96-6588-4 |
| ma2026detrc | OK | The bib is already correct: it is **IJCV 2026**, 134(8), article 364 (online 2026-07-25). Author order Ma, Li, Shang, Zhu, Ci, Qiao, Wang matches Springer and arXiv. The "arXiv 2025" label in `docs/LITERATURE_NUMBERS_20261108.md` is outdated: arXiv 2403.01543 v1 is from 2024-03, v4 from 2025-12. 0.2622/0.5430 (ViT) appears in arXiv v4 Table 1; the IJCV full text was not checked. | optional: `pages = {364}` instead of `note` | https://link.springer.com/article/10.1007/s11263-026-02939-4 ; https://arxiv.org/abs/2403.01543 |
| tang2026multicounterplus | OK | — (36(7):9462–9476, 7 authors in this order) | — | https://doi.org/10.1109/TCSVT.2026.3670243 (Crossref record) |
| gao2026pams | OK | Confirmed as **CVPR 2026 Findings** (CVF path `CVPR2026F`, booktitle "… (CVPR) Findings"), pp. 8143–8152; authors Gao, Li, Li. | — | https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html |
| wang2026d2stx | OK | Confirmed as **CVPR 2026 Findings**, pp. 8205–8214; 5 authors in this order. | — | https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html |
| luiten2021hota | OK | — (129(2):548–578; published online 2020-10, issue 2021-02) | — | https://doi.org/10.1007/s11263-020-01375-2 |
| ristani2016idmetrics | OK | — (LNCS 9914, pp. 17–35, eds. Hua & Jégou) | — | https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2 |
| welch1967fft | OK | — | — | https://doi.org/10.1109/TAU.1967.1161901 |
| lomb1976unequal | OK | — | — | https://doi.org/10.1007/BF00648343 |
| griffin1984stft | OK | — | — | https://doi.org/10.1109/TASSP.1984.1164317 |
| shazeer2017moe | OK (note) | Title, authors and ICLR 2017 confirmed via the Google Research publication page, which links `openreview.net/pdf?id=B1ckMDqlg`, and via arXiv 1701.06538. The OpenReview page itself sits behind a browser/bot challenge; it was not bypassed and so not read directly. | — | https://research.google/pubs/outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer/ ; https://arxiv.org/abs/1701.06538 |
| zhang2020context | OK | — (CVF pp. 670–678) | optional: `month = {June}` | https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Context-Aware_and_Scale-Insensitive_Temporal_Repetition_Counting_CVPR_2020_paper.html |
| zhang2021sight | OK | — (CVF pp. 14070–14079) | — | https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Repetitive_Activity_Counting_by_Sight_and_Sound_CVPR_2021_paper.html |
| li2024mfl | OK | — (WACV 2024, pp. 6499–6508, Li & Xu) | optional: `month = {January}` | https://openaccess.thecvf.com/content/WACV2024/html/Li_Repetitive_Action_Counting_With_Motion_Feature_Learning_WACV_2024_paper.html |
| li2023fullres | OK | — (arXiv 2305.13778, v1 2023-05-23; no journal_ref in arXiv metadata) | — | https://arxiv.org/abs/2305.13778 |
| sinha2024escounts | OK | Nothing wrong; pages and month are missing. | add `pages = {3056--3073}`, `month = {December}` | https://openaccess.thecvf.com/content/ACCV2024/html/Sinha_Every_Shot_Counts_Using_Exemplars_for_Repetition_Counting_in_Videos_ACCV_2024_paper.html |
| yao2025countllm | **FIX** | **Wrong authors 3–4.** The bib has "Zhu, Zhiqi and Zou, Yuexian"; CVF and arXiv list Zhiqi **Huang** and Lei **Li**. Pages are also missing. CVPR 2025 main conference. | `author = {Yao, Ziyu and Cheng, Xuxin and Huang, Zhiqi and Li, Lei}`, `pages = {19143--19153}`, `month = {June}` | https://openaccess.thecvf.com/content/CVPR2025/html/Yao_CountLLM_Towards_Generalizable_Repetitive_Action_Counting_via_Large_Language_Model_CVPR_2025_paper.html ; https://arxiv.org/abs/2503.17690 |
| feichtenhofer2019slowfast | OK | — | optional: `month = {October}` | https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html |
| jacobs1991adaptive | OK | — | — | https://doi.org/10.1162/neco.1991.3.1.79 |
| feichtenhofer2022masked | OK (note) | Title, 4 authors and NeurIPS 2022 confirmed (NeurIPS virtual poster page and arXiv 2205.09113). proceedings.neurips.cc refused the connection from this machine, so volume/pages were **not** verified; do not add pages from memory. | — | https://neurips.cc/virtual/2022/poster/55378 ; https://arxiv.org/abs/2205.09113 |
| gan2023multistride | **FIX** | **Wrong first author and incomplete list.** The bib has "Gan, Zhiyuan and others"; the chapter is by Guoxi Gan, Jia Su, Zonghui Wen and Shenmeng Zhang (Capital Normal Univ.). Pages, volume and part are also missing. | `author = {Gan, Guoxi and Su, Jia and Wen, Zonghui and Zhang, Shenmeng}`, `booktitle = {Knowledge Science, Engineering and Management: 16th International Conference, KSEM 2023, Proceedings, Part III}`, `series = {Lecture Notes in Computer Science}`, `volume = {14119}`, `pages = {295--306}`, `publisher = {Springer Nature Switzerland}`, `address = {Cham}`, `year = {2023}`, `doi = {10.1007/978-3-031-40289-0_24}` | https://link.springer.com/chapter/10.1007/978-3-031-40289-0_24 ; https://link.springer.com/book/10.1007/978-3-031-40289-0 |

UNVERIFIED: none. Two entries were confirmed only through secondary primary pages (the OpenReview page
for shazeer2017moe and the NeurIPS proceedings page for feichtenhofer2022masked were unreachable), as
noted in the table.

## 2. Task 2 — Where PAMS Countix 0.098 MAE / 0.879 OBO comes from

Sources: the CVF paper PDF and supplement were re-downloaded today. Their SHA-256 hashes (`f8085d91…`
and `84b10b8f…`) are identical to the copies archived on 2026-11-04 in `E:/ICASSP_TempoRAC/aris-work/literature/`.
- Paper: https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf
- Supplement: https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Gao_Count_What_Repeats_CVPRF_2026_supplemental.pdf

Findings:
1. **Videos/split: not specified.**
   - Sec. 4.1 (p. 8148) describes Countix only by its full published size, 8,757 videos, which is train+val+test combined.
   - It gives no test-split size and says nothing about download attrition.
   - It mentions no human-only or pose-detectable filtering.
   - It names no pose estimator. Sec. 3.3 and Supp. 7.11 mention only a pose-extraction step costing 25.81 ms/frame.
   - The Table 1 caption (p. 8148) defines "In-Domain" as trained and tested on the same dataset. For PAMS this training is label-free, on skeletons.
2. **The supplement says nothing about Countix.** The 9-page supplement never mentions Countix or UCFRep in its text. All of its analyses (OBN, tempo perturbation, sensitivity, failure cases) are on RepCount; the speed tests use the 152-video test set.
3. **The metric definition does not match the numbers.**
   - Sec. 4.1 defines MAE as the "average absolute difference between the predicted count" and the ground truth, with no normalisation by GT. OBO is defined as the fraction of videos with |C_pred − C_gt| ≤ 1.
   - Arithmetic check (my inference): with OBO = 0.879, at least 12.1% of videos have an absolute error ≥ 2. The un-normalised MAE would then be ≥ 0.242, which is above the reported 0.098.
   - The same holds for RepCount: OBO 0.842 implies MAE ≥ 0.316, versus the reported 0.129. Supp. Table 3 also shows 8.6% of RepCount videos with an error > 5.
   - So the reported MAE cannot follow the prose definition. It is presumably GT-normalised (the TransRAC/CountLLM convention), but the paper never says so.
4. **Baseline Countix numbers have no stated source.**
   - RepNet is listed at 0.432/0.393. RepNet's own paper reports 0.364/0.697.
   - Skeleton baselines (PoseRAC, GMFL, SPKDB-Net) also have Countix entries, with no stated origin.
   - Only CountLLM's 0.215/0.801 matches its source paper.
5. **Transfer rows.**
   - Countix→RepCount is reported as 0.129/0.842, identical to PAMS in-domain RepCount; the paper itself points this out.
   - RepCount→Countix is 0.099/0.861.
6. **No code, split list or checkpoint was found.** There is no link in the paper, the supplement or the CVF page, and a web search found none.

Conclusion: the paper leaves the Countix subset and the MAE normalisation unspecified, so 0.098/0.879 cannot
be placed under the same protocol as the RGB rows. Suggestion: the ¶ footnote in `tab/main.tex` should say
that the authors do not specify the Countix subset or the MAE normalisation, and the number should not be
treated as a like-for-like best.

## 3. Task 3 — CountLLM numbers and UCFRep protocol

Sources: the CVF PDF (pp. 19147–19148) and arXiv v2 HTML (https://arxiv.org/html/2503.17690v2) carry identical numbers.
- Table 1, VideoMAE encoder: RepCount **0.162/0.639**, UCFRep **0.116/0.839**, both confirmed. The VSwinT rows are 0.209/0.552 and 0.153/0.802.
- Table 2, Countix, VideoMAE: **0.215/0.801**, confirmed. VSwinT is 0.246/0.706.
- Table 3, cross-dataset: RepCount→UCFRep 0.139/0.816; RepCount→Countix 0.281/0.730.
- **Protocol (Sec. 4.1, paraphrased):**
  - RepCount and Countix: hyperparameters are tuned on val and results reported on test.
  - UCFRep: results are reported directly on the validation split.
  - Whether UCFRep is scored on whole videos or on annotated segments is **not stated**. CVF has no supplement and arXiv v2 has no appendix. Neither split's video count is stated.
  - MAE is GT-normalised (Eq. 13) and OBO follows Eq. 12. At test time, frames are sampled so that each clip has 32 frames.
  - Training uses WebVid-10M in stage 1, with Vicuna-7B as the LLM.
- Side note: CountLLM's Table 2 prints RepNet's Countix OBO as 0.303, which is RepNet's OBO *error*. `tab/main.tex` correctly uses 0.697.

## 2026-11-11 additions (to verify against the publisher pages before submission)
- `xiao2023cpack`: Xiao, Liu, Zhang, Muennighoff, Lian, Nie, "C-Pack: Packed Resources For General Chinese Embeddings", arXiv:2309.07597 (BGE embedding models; we use bge-base-en-v1.5). Title as on the latest arXiv version; a SIGIR 2024 version exists -- confirm venue/pages if citing the proceedings.
- `wolpert1992stacked`: D. H. Wolpert, "Stacked generalization", Neural Networks 5(2):241--259, 1992.
