# Historical paper build recovery — 2026-11-12

Outcome: the existing truthful historical original-split draft now compiles and renders on the server. This is infrastructure completion, not a final scientific manuscript or submission-readiness decision. No new experimental results were inserted.

## Scope and provenance

- Server: SSH alias temporac-seeta, verified project root `/root/autodl-tmp/TempoRAC`.
- Inherited source/build evidence: `paperbuild_fidelity_20261111_v1`; preserved unchanged.
- New isolated build: `/root/autodl-tmp/TempoRAC/paperbuild_recovery_20261112_v1`.
- Existing T1 font-encoding preamble repair was inherited. No TeX, BibTeX, style, scientific result, table row, experiment source or shared Git state was changed in this recovery.
- All 22 isolated source files have matching before/after SHA256. All 22 also match the current local `paper/cvpr27` source files (`shared_source_match.json`). Existing shared PDFs, logs and auxiliary files were not overwritten. No Git operation was run.
- All compile/render computation ran on the server. E: received scripts, evidence, PDFs and PNGs only. No local experiment, dependency install, GPU operation, schedule or external submission was performed.

## Focused repair and build

The inherited failure occurred after TeX and BibTeX: xdvipdfmx could not resolve `phvr8t` to a virtual/physical font. The matching `.vf`, map-selected font files, encodings and glyph-list resources were absent from the isolated local bundle. Retrieved exact byte ranges from the existing official `tlextras-2022.0r0.tar` bundle using the inherited index. `resource_recovery.json` records 340 newly fetched members and their SHA256; resources already recovered before the grouped fetch remain cached.

Reused the existing Tectonic 0.17.0 executable and existing project-only TeX/Poppler/PyMuPDF tools under `tools/paper`; no experiment environment changed. One focused resource repair succeeded. The normal cross-document settling sequence supplement → main → supplement → main completed with four exit codes 0. It is not four failed repair attempts.

Final command for each entry point, in the isolated source directory:

```
/root/autodl-tmp/TempoRAC/tools/paper/bin/tectonic \
  -b /root/autodl-tmp/TempoRAC/tools/paper/localbundle \
  --keep-logs --keep-intermediates main.tex
```

Use `supplement.tex` for the supplementary entry point. Preserve TMPDIR and XDG_CACHE_HOME under `tools/paper/tmp` and `tools/paper/cache`. Future changed manuscripts must be staged into a new isolated build directory; the recovery script intentionally uses the historical source and must not be mistaken for a final-results refresh.

## Acceptance

| Requirement | Status | Evidence |
|---|---|---|
| Preserve source/science | Pass | 22/22 source hashes unchanged and current-local match; no shared source/Git writes |
| Recover concrete build failure | Pass | Both PDFs exist; all four settling builds exit0; resource receipt and final compile logs retained |
| Reference/citation resolution | Pass for resolution; warning hygiene partial | Zero undefined references/citations, no `??`, `[?]`, `[TODO`, `[VERIFY]` or replacement glyph in extracted final text; duplicate-label warnings remain below |
| Font embedding | Pass | Poppler `pdffonts`: all 26 main and18 supplement fonts embedded, subsetted and Unicode mapped |
| Page organization | Pass for this historical draft | Main7 pages; Conclusion, LLM Usage Statement and References all start on page6; separate supplement4 pages, references start page3 |
| Render every page | Pass | Server PyMuPDF and Poppler renders:7+4 pages each; final Poppler set1275×1650 at150dpi, hashes saved |
| Visual layout | Pass for bounded render QA | All11 pages inspected; no clipped body text, broken glyphs, overlapping tables or figure placeholders; page-specific notes below |
| Final scientific manuscript | Not reached | Corrected full3/new official comparison and final claim/citation assurance remain owned by root; current PDF explicitly retains historical original-split findings |

The main PDF is122555 bytes; supplement67979 bytes. The supplement being below100KiB is not evidence of corruption: it is a valid4-page text/table PDF with embedded fonts.

## Page-specific visual inspection

| Page | Findings |
|---|---|
| Main1 | Anonymous title/abstract and two-column introduction fit; normal first-page title space; no clipped text |
| Main2 | Framework diagram and caption render; compact diagram labels remain readable, but enlargement can be considered at final editorial layout; columns continue normally below |
| Main3 | Method, formulas and objective fit; equations legible; no overlap or extra blank column space |
| Main4 | Method continues into setup across columns/pages; metrics and source-overlap disclosure render without clipping |
| Main5 | Main comparison and ablation captions/headers correspond; white/black horizontal-rule tables fit; numbers and bold cells render; compact ablation type remains readable |
| Main6 | Conclusion continues naturally across columns; limitations, named-tool LLM statement and references follow; no artificial page break |
| Main7 | References continue, with natural trailing blank space after final entry rather than a broken section/float |
| Supplement1 | Anonymous supplement title, provenance table and overlap disclosure fit; table is white/black with horizontal rules; cross-column continuation normal |
| Supplement2 | Multi-benchmark table fits; compact typography but readable; no split header or detached caption; prose continues around table |
| Supplement3 | Full-width ablation table and single-column variant table align with captions; original negative rows and mixed winners remain visible; references begin below |
| Supplement4 | References finish in left column; remaining blank area is natural end-of-document space |

An apparent missing review header in batch image presentation was checked individually. The original PNGs contain the headers and full line numbers. Single-page and all-page Poppler renders of main5 are byte-identical (SHA2564878fb6dbc9a49e6b59e0ae0141378174f97897f4507247b42d994856085cf17). No renderer/PDF defect was established; do not propagate the preliminary suspicion as a finding.

## Warnings and limits retained

- Main and supplement each retain11 natbib multiply-defined citation keys because the inherited `xr` package imports bibliography labels from the other document, whose overlapping keys are also defined locally. Displayed citation numbers resolve; this recovery did not change the source/reference strategy or independently certify citation correctness.
- Cached `lineno.sty` has legacy Latin-1 bytes in an author-name comment at line296. Tectonic warns that invalid UTF-8 is replaced; the comment is not typeset and no replacement glyph appears in either final extracted PDF. Bundle file preserved.
- Both logs retain unused subcaption-setup warning; supplement retains three `h`→`ht` float-placement notices. Neither log contains overfull/underfull box warnings.
- No final claim audit, literature re-review, corrected-result insertion, official-rule re-freeze, author approval or submission occurred. Final work must use the new completed experiment evidence and rerender all pages after any manuscript edit.

## Artifacts

- E: `main.pdf`, `supplement.pdf`, `poppler/`11 final page PNGs and SHA manifest, `rendered/` text/font/metadata reports and initial PyMuPDF PNGs.
- `build-evidence/`: final `.log`, `.aux`, `.bbl`, `.blg`, source manifests, resource receipt and PDF copies.
- `build_evidence.tar`, `poppler_pages.tar`: downloaded server evidence packages; retained.
- Server helper scripts: `tools/paper/recover_build_20261112.py`, `render_pages_20261112.py`, `render_poppler_20261112.py`.

Skills: full pinned paper-compile (`f76ec8d6...`,9381 bytes; no mandatory support-file links) and full PDF skill read. PDF artifact-operation marker ran successfully once before build creation. No broad rewriting or scientific review was attempted.
