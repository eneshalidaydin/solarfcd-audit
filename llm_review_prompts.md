# Language-model visual review prompts

This document reproduces the prompts given to the language-model reviewers whose visual decisions on candidate image pairs are recorded in `visual_pair_reviews.csv` (16,303 pairs, column `review_round`). `08a_gorsel_etiket_derle.py` compiled these decisions, and `08_asama2_manifest.py` used them as manual edges, as cannot-link constraints and for calibrating the matcher when it built the photo and scene groups of `solarfcd_manifest.csv`. The audit rounds also estimated the residual cross-fold leakage of each version of the grouping.

## Reviewers

The reviewers were Claude Opus 5.5 (Anthropic) agents run through Claude Code, Anthropic's agentic coding tool, in November 2026 (16–18 November). Each agent worked in its own context with one of the prompts below. It built montages and full-resolution crops of the image pairs with its own short Python scripts, viewed them with Claude Code's file-reading tool, which displays images, and wrote one decision per pair with a confidence level and a one-line reason. A Claude Code workflow script (JavaScript) orchestrated the agents of each round: it passed the prompts, collected a structured summary (JSON) from every agent and started the blind second opinions, the adjudicator and the critic where the design required them. A further 21 pair decisions (tags `r3_benim_*`) were made from montages by the main Claude Code session, also running Claude Opus 5.5, while the r3 grouping was prepared, so no separate prompt exists for them.

## Human check of 115 decisions

The author checked 115 of these decisions by hand, blind to the reviewers' verdicts, in a browser form that showed the two images of each pair side by side as 520-pixel thumbnails (`12_insan_kontrol_seti.py`, `12b_secimli_form.py`).

- Part A (15 pairs): the 14 pairs that the r4 audit adjudicated as leak or unsure, plus one pair found by its critic.
- Part B (100 pairs): a random sample (seed 20261119) of the compiled decisions, 50 of them same photo, same scene or uncertain and 50 of them definitely unrelated.

The checked pairs are flagged in `visual_pair_reviews.csv` (`human_checked`, `human_check_id`), the author's decisions are in `insan_kararlari.csv`, and `13_insan_uyum.py` computes the agreement. None of the 50 definitely unrelated decisions of Part B was judged related by the author (0/50, Wilson 95% CI 0–7.1%). Of the 42 same-photo or same-scene decisions of Part B, the author confirmed 17 (40.5%, 95% CI 27.0–55.5%), so the reviewers linked too many pairs, which errs on the leak-safe side. The author judged all 15 uncertain pairs of Parts A and B unrelated, and 3 of the 15 pairs of Part A related. The binary Cohen's κ in Part B was 0.425 (n = 92).

## How the prompts are reproduced

The prompts are copied from the workflow scripts with three kinds of editing.

- Local paths are replaced by placeholders: `<WORK_DIR>` is the working folder of the analysis (the pilot folder, `C:/solarfcd/pilot` in the original runs), `<SOLARFCD_DIR>` is the SolarFCD-Final archive, `<DATA_DIR>` is the data folder that contains it, `<DOWNLOAD_DIR>` is the folder with the originally downloaded archives and `<PYTHON>` is the interpreter of the conda environment in `environment.yml`. Backslashes in paths are written as forward slashes.
- Template variables are shown in braces, for example `{p}` for the chunk number or `{k}` for the reviewer number, and JSON results passed from earlier agents are shown as a description in braces.
- `[CTX]` or `[PRE]` at the start of a prompt stands for the shared context block given first in the same section.

Folder, file and column names are kept as in the original runs, so many of them are Turkish. The most frequent ones are explained below, and the internal manifest columns are mapped to the deposit columns in `README.md`.

| Name in the prompts | Meaning |
|---|---|
| `kararlar.csv` with `kod`, `karar`, `guven`, `gerekce` | decision file with pair code, decision, confidence and reason |
| `girdi`, `parca_{p}.csv`, `havuz_tam.csv` | input folder, chunk {p}, full candidate pool |
| `tarama`, `dogrulama` | screening, verification (also the folder of the audits) |
| `orneklem`, `gozden1`, `gozden2`, `birlestir`, `kod`, `azinlik`, `celiski`, `yakin`, `elestirmen` | agent labels: sample, reviewer 1, reviewer 2, merge, code, minority, conflict, near miss, critic |
| `asama2`, `manifest_asama2.csv` | stage-2 output folder and its internal manifest |
| `kati_grup`, `foto_grup`, `sahne_grup`, `genis_grup` | strict, photo, scene and broad (r1 only) group |
| `celiskili`, `uygun`, `temsilci`, `kat_v1`, `kat_v2`, `kat_v1_t1`…`t4` | label conflict, eligible, representative, primary fold, fold of all eligible files, folds with other seeds |
| `aile`, `kaynak`, `termal`, `dosya` | file-name family, source, thermal image, file name |
| `kenarlar.csv`, `eslesmeler.csv`, `aday_eslesmeleri.csv`, `gorsel_etiketler.csv` | grouping edges, match cache, verified candidate pairs, compiled visual decisions |
| `inl`, `umin`, `gecerli`, `alt_pay`, `kimlik`, `filigran`, `kap_kucuk`, `ncc_hp` | inliers, distinct inlier positions, valid homography, footer share of inliers, identity-frame deviation, watermark flag, inlier hull coverage of the smaller image, high-pass normalised cross-correlation |
| `celiskili_gruplar.csv`, `celiski_kenarlari.csv`, `kat_arasi_sizinti.csv`, `kat_arasi_yakin_adaylar.csv`, `buyuk_gruplar.csv`, `kalibrasyon.csv`, `ozet` | conflicting groups, conflict edges, cross-fold leakage, cross-fold near candidates, largest groups, calibration, summary |
| `notlar` | notes |

## Overview of the rounds

| Round | Workflow script | Date | `review_round` tags | Pairs |
|---|---|---|---|---|
| Audit of the first grouped manifest (r1) | `solarfcd-stage2-manifest-verify` | 16 November 2026 | `r1_leak, r1_sift, r1_conflict_tespit, r1_critic` | 304 |
| Audit of manifest r2 | `solarfcd-stage2-r2-verify` | 17 November 2026 | `r2_*` | 3,970 |
| Audit of manifest r3 | `solarfcd-stage2-r3-verify` | 17 November 2026 | `r3_denetim_*` | 6,198 |
| r3 preparation (main session) | – | 17 November 2026 | `r3_benim_*` | 21 |
| Visual screening for manifest r4, round 1 | `solarfcd-r4-gorsel-tarama` | 17 November 2026 | `r4_tarama, r4_tarama_rastlanti` | 1,457 |
| Visual screening for manifest r4, round 2 | `solarfcd-r4-gorsel-tarama-tur2` | 17 November 2026 | `r4_tarama2` | 514 |
| Audit of manifest r4 | `solarfcd-stage2-r4-verify` | 18 November 2026 | `r4_denetim_*` | 3,089 |
| Targeted visual screening for manifest r4.1 | `solarfcd-r41-kaggle-tarama` | 18 November 2026 | `r4_tarama41, r41_tarama_rastlanti` | 2,137 |

Pairs is the number of pairs in `visual_pair_reviews.csv` with at least one decision from the round; a pair can carry decisions from several rounds.

## Audit of the first grouped manifest (r1)

Workflow script `solarfcd-stage2-manifest-verify`, run on 16 November 2026. Four auditors each tried to break one property of the first grouped manifest (cross-fold leakage, precision of the SIFT edges, label-conflict exclusions, usability of the folds), and a completeness critic weighed their findings. The leak, sift and conflict auditors and the critic judged image pairs by viewing them; the folds auditor did not.

Each auditor returned the property checked, a verdict (holds, holds with caveats, broken), key numbers and findings (severity, problem, evidence, fix); the critic returned code defects, protocol gaps, decisions for the author and a verdict.

### Shared context PRE (prepended to every prompt of this audit)

```text
CONTEXT: SolarFCD photovoltaic fault dataset; a Q1 paper needs a LEAK-FREE grouped 5-fold protocol. The analyst built a stage-2 manifest with <WORK_DIR>/08_asama2_manifest.py (read it first). Outputs in <WORK_DIR>/asama2/:
- manifest_asama2.csv: one row per image file (3,742). Columns: path, rel_path, dosya, split (original official split), label (Healthy / Surface_Obstruction / Structural_Fault / Electrical_Fault), ok, aile (file-name family), kaynak (source dataset), termal (thermal C#/S# files), md5, width, height, kati_grup (strict group = connected components of MD5-equality ∪ dihedral pHash Hamming ≤ 4), kati_grup_boyutu, kati_grup_etiketleri, kati_grup_kaynaklari, celiskili_grup (strict group has >1 label), temsilci (one representative per strict group: largest pixel area, tie → shortest rel_path), genis_grup (broad group = strict ∪ SIFT+RANSAC-verified DINOv2 candidate edges: inlier ≥ 30 and inlier/good-matches ≥ 0.6), genis_grup_boyutu, genis_karisik, kat_v1, kat_v2.
- kat_v1 (recommended): RGB only, non-conflicted strict groups, one representative per strict group; StratifiedGroupKFold(5, seed 42) over genis_grup. kat_v2: RGB, non-conflicted, ALL files, same genis_grup→fold mapping. Excluded rows have fold = -1.
- karmalar.csv (md5 + 8 dihedral pHashes as 16-hex strings ph0..ph7 per path), dinov2_d4_gommeler.npy (8×N×768 float16, DINOv2 ViT-B/14 embeddings for the 8 dihedral transforms, row order = manifest order of ok rows), sift_aday_ciftler.csv (19,126 DINOv2 candidate pairs i,j with inlier, iyi_eslesme, aynali, S, phash_d, rel_i, rel_j, label_i, label_j; i/j index manifest rows), sift_dogrulanan_kenarlar.csv (1,182 accepted edges), sift_kontrol.csv (calibration: 300 positive / 300 random negative pairs), celiskili_gruplar.csv, kat_arasi_sizinti.csv, kat_sinif_v1.csv, kat_sinif_v2.csv, ozet.json, ozet.md, kalibrasyon_*.png.
- The training script <WORK_DIR>/02_egit.py uses: --protocol grouped --fold-col kat_v1 --group-col genis_grup; test = rows with fold == k; remaining rows with fold ≥ 0 are split into train/inner-validation by GroupShuffleSplit(test_size=0.125, seed 42) on the group column; early stopping on inner-validation macro-F1.
- Images: <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/ (3,742 files). Python with numpy/pandas/scipy/sklearn/PIL/imagehash/opencv(cv2)/torch/timm: <PYTHON> (set PYTHONUTF8=1). The Read tool displays images (use it to LOOK at pairs). The GPU is free (inference only if needed).
HARD RULES: read-only. Never modify/move/delete anything under <DATA_DIR>, <DOWNLOAD_DIR>, or existing files in <WORK_DIR>. Write your scripts/outputs ONLY in <WORK_DIR>/dogrulama/asama2_<YOUR_LABEL>/. Do not train models.
YOUR JOB: try hard to BREAK one property of this manifest (below). Recompute independently where possible (your own code, not the analyst's derived numbers). LOOK at images for every judgement about duplicates/same-scene. Report concrete, file-level evidence and exact numbers, and propose specific fixes (with the parameter or code change).
```

### Auditor 'leak' (prompt = [PRE], then 'YOUR_LABEL: leak', then this task)

```text
PROPERTY: kat_v1 and kat_v2 are leak-free across folds. (1) Independently recompute MD5 and your own rotation/mirror-invariant perceptual similarity from the image files and list every cross-fold pair (different fold, both ≥ 0) that is byte-identical or at pHash Hamming ≤ 4, ≤ 6, ≤ 8, ≤ 10. (2) LOOK at all cross-fold pairs at ≤ 8 and at least 15 of those at 9-10, plus the 17 kat_v1 cross-fold pairs with DINOv2 dihedral cosine ≥ 0.97 (from the embeddings), plus the 20 highest-inlier SIFT candidate pairs that were REJECTED and cross folds. Classify each as same photo (incl. crop/rescale/re-encode/mirror), same scene/installation (different shot), or unrelated. (3) Estimate the residual cross-fold leakage of kat_v1 as a count of test-fold images with a same-photo or same-scene partner in another fold, and say whether it is acceptable for a Q1 leak-free claim.
```

### Auditor 'sift' (prompt = [PRE], then 'YOUR_LABEL: sift', then this task)

```text
PROPERTY: the SIFT-verified edges (inlier ≥ 30 and ratio ≥ 0.6) are mostly true same-photo/same-scene links, and the rejected candidates do not hide many true duplicates. (1) Draw a seeded stratified random sample of 40 ACCEPTED edges across inlier bands (30-50, 50-100, 100-300, ≥300) and LOOK at every pair; classify same photo / same scene / unrelated (e.g. watermark or repetitive-grid false match). (2) Draw 30 REJECTED candidates with inlier 15-29 or ratio < 0.6 but inlier ≥ 30, LOOK, classify. (3) Estimate precision and the miss rate; check specifically whether watermark (e.g. shutterstock) matches link unrelated images and how large such watermark groups get; check whether any accepted edge links images of different labels that are clearly the same photo (an undetected label conflict that should be excluded). Recommend whether to change thresholds.
```

### Auditor 'conflict' (prompt = [PRE], then 'YOUR_LABEL: conflict', then this task)

```text
PROPERTY: exclusions are correct and complete. (1) All thermal files (C#.jpeg / S#.jpeg, 407) and all conflicted strict groups are excluded from kat_v1 and kat_v2; no remaining strict group has >1 label. (2) Conflicts are real: for strict groups whose conflict arises only via pHash edges (not identical MD5), LOOK at the images and check they are truly the same photo (a false pHash link would wrongly delete data). List every non-thermal conflicted file (63) with its group and say whether each conflict is genuine. (3) Undetected conflicts: find same-photo pairs with different labels that are NOT in the same strict group (e.g. via the SIFT-accepted edges, genis_karisik groups, or your own crop-robust check) and LOOK at them; these would remain in the data as label noise. (4) Representatives: verify exactly one temsilci per strict group and that the selection rule was applied; verify kat_v1 == representatives of eligible groups.
```

### Auditor 'folds' (prompt = [PRE], then 'YOUR_LABEL: folds', then this task)

```text
PROPERTY: the folds are usable for a Q1 benchmark. (1) Verify every genis_grup lies in exactly one fold in kat_v1 and kat_v2, the fold class counts in kat_sinif_v1.csv / kat_sinif_v2.csv, and that no eligible row was left unassigned. (2) Reproduce 02_egit.py's inner split (GroupShuffleSplit 0.125, seed 42, group col genis_grup) for each outer fold of kat_v1 and kat_v2 and report class counts in train / inner-validation / test — the analyst found only 2-11 Electrical_Fault and 2-6 Structural_Fault images in kat_v1 inner validation. Quantify how noisy early stopping on inner-validation macro-F1 would be (e.g. the macro-F1 change caused by flipping one Structural_Fault prediction). (3) Compare alternatives and recommend one, with the exact code change in 02_egit.py it would need: (a) use another outer fold as validation (3 train folds / 1 val / 1 test), (b) StratifiedGroupKFold-based inner split, (c) fixed epoch budget without early stopping, (d) early stopping on inner-validation loss. (4) Given only 44 Structural_Fault and 79 Electrical_Fault distinct RGB images in kat_v1, state what per-class metrics can be reported credibly (e.g. pooled out-of-fold predictions over all 5 folds, repeated grouped CV with different fold seeds) and estimate the 95% CI width of per-class recall for SF with n = 44.
```

### Completeness critic

```text
[PRE]

YOUR_LABEL: critic
You are the COMPLETENESS CRITIC. Read 08_asama2_manifest.py and 02_egit.py carefully (look for bugs: index alignment between manifest rows, karmalar.csv and embeddings; symmetric/diagonal handling; representative selection; StratifiedGroupKFold use; the -1 exclusion convention vs 02_egit.py's filters), then weigh the four auditors' findings below. Identify code defects, protocol gaps a Q1 reviewer would raise (scene-level overlap, watermark/source shortcuts, source-stratified or leave-one-source-out evaluation, class imbalance, statistics), and the decisions only the user can make (each with a recommended default). Keep it concrete.

AUDITOR FINDINGS:
{JSON findings of the four auditors}
```

## Audit of manifest r2

Workflow script `solarfcd-stage2-r2-verify`, run on 17 November 2026. A sampling agent drew 120 evaluation images (seed 20261117) and built candidate sheets with retrieval signals that were independent of the grouping pipeline, without judging them. Two blind reviewers labelled every candidate pair, and a third agent merged the two reviews and adjudicated every disagreement, every unsure pair and every pair called a leak. In parallel, a code auditor, a census of all Structural_Fault and Electrical_Fault images and a label-conflict auditor judged further pairs, and a completeness critic re-checked the most consequential claims by viewing the images.

The sampler returned the folder, counts and signals used; the reviewers returned their counts, the path of labels.csv and every same_photo, same_scene or unsure pair with judgement, confidence and reason; the merger and the auditors returned the property checked, a verdict, key numbers and findings; the critic returned a verdict, a residual-leak statement, spot checks, consistency issues, unchecked items, code defects and decisions for the author.

### Shared context CTX (prepended to every prompt of this audit)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset with classes Electrical_Fault (EF), Healthy (H), Structural_Fault (SF), Surface_Obstruction (SO). Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. Pilot folder: <WORK_DIR>.

TOOLS. Python: <PYTHON> (torch with CUDA on an RTX 4080 Laptop 12 GB, timm, opencv-python-headless 5 with SIFT/AKAZE/ORB/USAC_MAGSAC, imagehash, PIL, numpy, pandas, scipy, scikit-learn). Locally cached timm weights (pretrained=True works offline): convnext_tiny.fb_in22k_ft_in1k, swin_tiny_patch4_window7_224.ms_in22k_ft_in1k, tf_efficientnetv2_s.in21k_ft_in1k, resnetv2_101.a1h_in1k, vit_base_patch14_dinov2.lvd142m. Always run python with -B. Put your scripts and ALL outputs only in your own folder <WORK_DIR>/dogrulama/asama2_r2/<LABEL>/ (create it). Do not modify or create any other file. Other agents share the GPU: keep your GPU memory under ~3 GB.

OBJECT UNDER TEST. The stage-2 grouped 5-fold manifest "r2": <WORK_DIR>/asama2/manifest_asama2.csv (MD5 514def109da27e1c18ee61a48edce95d), built by <WORK_DIR>/08_asama2_manifest.py — read its docstring first. Key columns: rel_path, path, dosya, split, label, aile, kaynak, termal, md5, width, height, kati_grup ⊆ foto_grup ⊆ sahne_grup, celiskili (foto_grup carries more than one label → excluded), uygun (= RGB and not celiskili), temsilci (one representative per foto_grup among uygun rows), kat_v1 (fold 0-4 for temsilci rows, -1 = not in v1; the PRIMARY benchmark, 2,106 images, 421-422 per fold), kat_v2 (fold for all uygun rows; sensitivity only), kat_v1_t1..t4 (same grouping, other fold seeds). Folds come from StratifiedGroupKFold over sahne_grup. Other files in <WORK_DIR>/asama2/: kenarlar.csv (every grouping edge with its type: kati, foto_geometrik, foto_gorsel, sahne_geometrik, dino_mknn, exif, dosyano, sahne_gorsel), eslesmeler.csv (82,005 verified candidate pairs, ONE file pair per pair of kati groups, with RootSIFT/MAGSAC metrics: inl, iyi, umin = distinct inlier positions, gecerli = valid homography, alt_pay = share of inliers in the bottom 12% strip, kap_a/kap_b = hull coverage, ncc / ncc_hp = warped (high-pass) NCC, ayna = mirrored, kaynaklar = candidate sources, r70/r80 = exhaustive-screen counts), celiskili_gruplar.csv, celiski_kenarlari.csv, kat_arasi_sizinti.csv, kat_arasi_yakin_adaylar.csv, kalibrasyon.csv, gorsel_etiketler.csv (visual decisions from the previous audit, used as manual edges), ozet.md, ozet.json. The previous version r1 is archived in <WORK_DIR>/asama2/r1/.

HISTORY. r1 was audited (<WORK_DIR>/dogrulama/asama2_{leak,sift,conflict,folds,critic}/) and found to leak at scene level: about 18% of test images had a same-photo or same-scene partner in another fold; it also kept hidden same-photo label conflicts. r2 rebuilds the grouping to fix this. Your job is to try to BREAK r2. Be adversarial and quantitative; do not take the pipeline's own diagnostics on trust.

DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, sky-replaced, watermarked, re-compressed or embedded in a collage/advert. same_scene: a different photograph of the same physical installation, roof, array, panel or event (consecutive shots, before/after cleaning, another angle of the same roof, the same damaged panel). unrelated: different installations even if similar (same panel model, generic rooftops, same stock watermark, same advert template). A LEAK = two kat_v1 images (kat_v1 >= 0) in DIFFERENT folds that are same_photo or same_scene. Every visual judgement must come from actually viewing the images (Read tool on JPG/PNG files or montages you create; open/crop the originals when unsure) and must be recorded with a one-line reason. Keep montages legible: at most ~2400 px wide, thumbnails >= 220 px.
```

### Sampling agent 'orneklem' (builds the sample and the candidate sheets; does not judge)

```text
[CTX]
LABEL = orneklem. ROLE: build a seeded audit sample and candidate sheets for two blind reviewers. You do NOT judge leaks yourself.
1. Sampling, fixed before any retrieval: read the manifest, keep rows with kat_v1 >= 0, sort by rel_path, rng = numpy.random.default_rng(20261117), choose 120 rows without replacement. Save sample.csv (order, query_id Q000..Q119, rel_path, label, kat_v1) before computing anything else.
2. Retrieval must be INDEPENDENT of the r2 pipeline: do not use DINOv2 and do not read any file under <WORK_DIR>/asama2/ except manifest_asama2.csv (only rel_path, path, dosya, aile, label, kat_v1). Candidate pool for a query = kat_v1 >= 0 rows whose fold differs from the query's fold. Signals:
 (a) ConvNeXt-T (timm convnext_tiny.fb_in22k_ft_in1k, num_classes=0, 224 px, ImageNet normalisation) embeddings, L2-normalised; similarity = max cosine over the 8 dihedral transforms (4 rotations x mirror) of the candidate; top 6.
 (b) Swin-T (swin_tiny_patch4_window7_224.ms_in22k_ft_in1k) the same way; top 4.
 (c) 32x32 grey thumbnail NCC, max over the 8 dihedral transforms; top 3.
 (d) Local features: AKAZE (cv2.AKAZE_create) on the image resized to 1024 px long side; BFMatcher(NORM_HAMMING) kNN k=2 with Lowe ratio 0.8 plus a mutual (cross) check; cv2.findHomography(..., cv2.USAC_MAGSAC, 5.0); also try the horizontally mirrored candidate. Run it against the union of the query's top 80 ConvNeXt and top 40 Swin cross-fold neighbours; keep the top 3 by inliers if inliers >= 12.
 (e) Metadata: same Kaggle family (dosya like Imgclean_N_0 or Imgdirty_N_1) with |N difference| <= 5, and same EXIF camera model (tag 272) with DateTimeOriginal (36867, else 306) on the same day; nearest first, cap 6 per query.
3. Sheets: one block per query = the query (thick red border) followed by its de-duplicated candidates (up to 14), thumbnails about 230 px, each with a caption: short file name, label, fold, signals and scores, and its id Qqqq-Ccc. Wrap a query's candidates onto a second line if needed; put 2 queries per sheet; keep sheets <= 2400 px wide (about 60 sheets: sheet_00.jpg ...). Save candidates.csv (query_id, cand_id, query_rel, cand_rel, label_q, label_c, fold_q, fold_c, signals, scores, sheet). View at least three sheets yourself to make sure they are legible, and fix them if not.
Return the folder path, counts, the signals actually used, and any problem. Do NOT label pairs.
```

### Blind reviewers 'gozden1' and 'gozden2' ({k} = 1 or 2, {other} = the other reviewer)

```text
[CTX]
LABEL = gozden{k}. ROLE: blind reviewer {k} of 2. Another reviewer labels the same sheets independently: do NOT open <WORK_DIR>/dogrulama/asama2_r2/gozden{other}/, and do NOT look at the manifest's group columns (kati_grup, foto_grup, sahne_grup) or at asama2/eslesmeler.csv / kenarlar.csv before you have finished judging.
INPUT: <WORK_DIR>/dogrulama/asama2_r2/orneklem/ (sheet_*.jpg, candidates.csv, sample.csv).
TASK: view every sheet. For every candidate decide same_photo / same_scene / unrelated / unsure relative to its query. For anything that could be same_photo or same_scene, open both originals at full resolution (crop and zoom regions with your own small script) before deciding. Evidence that counts: identical roof, tiles, chimney, vents, trees, skyline or fence geometry; identical dirt, droppings, cracks, burn marks or snail-trail patterns; the same impact point on a broken panel; the same person or vehicle; matching or adjacent stock-photo IDs; matching EXIF session. Generic similarity (same panel type, similar sky, same watermark) is NOT enough.
Write labels.csv in your folder (query_id, cand_id, query_rel, cand_rel, judgement, confidence high/medium/low, reason). A query is a leak if any of its candidates is same_photo or same_scene (folds always differ by construction).
Return the counts, the labels.csv path and every same_photo / same_scene / unsure pair with your reason.
```

### Merger and third reviewer 'birlestir'

```text
[CTX]
LABEL = birlestir. ROLE: merge the two blind reviews, adjudicate as third reviewer, and explain every confirmed leak.
INPUTS: <WORK_DIR>/dogrulama/asama2_r2/orneklem/ (sample.csv, candidates.csv, sheets), .../gozden1/labels.csv, .../gozden2/labels.csv. Their returned summaries: PREP={JSON summary returned by the sampling agent} REVIEWER1={JSON summary returned by reviewer 1} REVIEWER2={JSON summary returned by reviewer 2}
1. Pair level: cross-table of the two reviewers' judgements; Cohen's kappa for leak-type (same_photo or same_scene) vs unrelated, unsure reported separately.
2. Query level: leak rate per reviewer, union (either reviewer) and consensus (both), each with a Wilson 95% CI, and extrapolated to the 2,106 kat_v1 images.
3. Adjudication: view every pair where the reviewers disagree or either says unsure, and every pair either reviewer called a leak; decide yourself with a reason (open the originals). Report the adjudicated per-query leak rate with Wilson CI, per fold, and per class of the query.
4. For every adjudicated leak pair, explain why r2 missed it: look up both files in asama2/manifest_asama2.csv (kati/foto/sahne groups, folds); find the pair of their kati groups in asama2/eslesmeler.csv (one file pair per kati-group pair; match on the kati groups of rel_i/rel_j, not on the exact files) and report inl, umin, gecerli, alt_pay, ncc_hp, kaynaklar if present, or that the pair was never a candidate; say which rule or threshold would have caught it and what that rule would cost (for example how many extra edges or how much larger the largest group would get — estimate from eslesmeler.csv).
Return the numbers, the adjudicated leak list and the per-leak diagnosis.
```

### Code and construction auditor 'kod'

```text
[CTX]
LABEL = kod. ROLE: recompute and attack the r2 construction.
1. Rebuild the partitions from asama2/kenarlar.csv as connected components (kati_grup from type kati; foto_grup from kati + foto_geometrik + foto_gorsel; sahne_grup from all types) and compare them with the manifest columns (identical partitions? report any difference).
2. Recompute kati groups independently from the images: MD5 of the bytes plus 8 dihedral pHashes (imagehash.phash on PIL 'L' after im.draft('L',(512,512)) for JPEG/MPO, as in hash_isci), distance = minimum over all 64 orientation pairs, threshold <= 4, union with MD5 equality. It must equal kati_grup.
3. Verify the bookkeeping: celiskili = foto_grup with more than one label; uygun = not termal and not celiskili; exactly one temsilci per foto_grup that has uygun rows, chosen by largest width*height, then shortest rel_path, then lexicographic rel_path; kat_v1 >= 0 exactly on temsilci rows and kat_v2 >= 0 exactly on uygun rows; every sahne_grup is in one fold for kat_v1, kat_v2 and every kat_v1_t*; reproduce the folds with sklearn StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) on the temsilci rows in manifest order with y = label + '|' + kaynak and groups = sahne_grup (seed 42 for kat_v1, 1..4 for kat_v1_t1..t4).
4. Check every number in asama2/ozet.md and ozet.json against your own recomputation.
5. Reproducibility: run "python -B <WORK_DIR>/08_asama2_manifest.py --cikti <your folder>/yeniden" from <WORK_DIR> as working directory (it recomputes hashes, DINOv2 embeddings, SIFT features, the GPU screen and all matches into your folder only, ~15 min). Compare the resulting manifest with the official one (MD5 and, if different, which rows/columns differ and why — e.g. GPU float16 non-determinism at thresholds).
6. Read 08_asama2_manifest.py critically: candidate generation (any holes besides the intended skip of pairs inside one kati group?), the matcher (mutual NN + ratio, MAGSAC, homography validity, distinct positions, footer rule), NCC, EXIF and file-number rules, group-level mutual kNN, calibration merge, leak table, cache validation, and the -1 conventions. Report concrete bugs with a failing input where possible.
7. Check compatibility with <WORK_DIR>/02_egit.py --protocol grouped --fold-col kat_v1 --group-col sahne_grup by reading bolmeleri_hazirla and grup_ayir and replaying their split logic in your own script (do NOT train and do NOT write into <WORK_DIR>/runs). Report inner-validation class counts per fold.
Return defects with severity and evidence.
```

### Census of the minority classes 'azinlik'

```text
[CTX]
LABEL = azinlik. ROLE: exhaustive census of the minority classes, which drive macro-F1 (kat_v1 has only 34 SF and 73 EF images).
For every kat_v1 image of class Structural_Fault or Electrical_Fault, compare it with EVERY kat_v1 image in a different fold (all classes) using a local-feature matcher that differs from r2's (r2 uses RootSIFT at 800 px with 2000 keypoints, mutual NN + Lowe 0.8, MAGSAC 4 px). Suggested: SIFT on the image capped at 1400 px long side with up to 4000 keypoints, Lowe 0.75 plus cross-check, cv2.USAC_MAGSAC 5 px, also against the mirrored image; or AKAZE. GPU descriptor matching is fine. Add global-embedding neighbours from ConvNeXt-T and Swin-T (timm, dihedral max cosine), top 10 cross-fold each.
View, for every SF/EF image, its strongest cross-fold candidates (at least the top 5 by inliers and the top 5 by each embedding) and judge same_photo / same_scene / unrelated. Pay special attention to shattered or cracked panels photographed from other angles, collages that contain other images, the same burnt cell or snail trail, and the same site at different crops.
Report: every cross-fold same_photo / same_scene pair involving SF or EF (files, folds, reason); the number of distinct SF and EF scenes per fold under r2; whether one scene dominates a fold's SF or EF test set; and whether any SF/EF v1 image looks mislabelled (as an aside, with reason).
```

### Label-conflict auditor 'celiski'

```text
[CTX]
LABEL = celiski. ROLE: check the label-conflict exclusions and the same-photo deduplication, and look for remaining hidden conflicts.
1. For every RGB foto_grup with celiskili = True (termal = False): view its files and the cross-label edges listed in asama2/celiski_kenarlari.csv. For each cross-label edge decide: same_photo (true conflict) or not (before/after of the same roof = same_scene, or unrelated). Report false merges, the files they wrongly excluded, and whether the group would split into label-pure parts without them.
2. Same-label deduplication: from asama2/kenarlar.csv take the foto_geometrik edges whose two files have the SAME label and lie in different kati groups; view a seeded random sample of 40 (numpy default_rng(7)) and judge same_photo vs not. Estimate the precision of the same-photo rule (these merges reduce kat_v1 to one representative per photo).
3. Hidden conflicts: among uygun rows (kat_v2 >= 0) search for same_photo pairs that carry DIFFERENT labels but lie in different foto_grups. Use an independent screen (e.g. ConvNeXt/Swin top-20 cross-label neighbours plus local-feature verification and warped high-pass NCC), then view every candidate the screen flags. Report confirmed hidden conflicts with files, labels and folds.
4. Report same-label same_photo pairs that lie in different foto_grups AND different kat_v1 folds (these are leaks).
```

### Completeness critic 'elestirmen'

```text
[CTX]
LABEL = elestirmen. ROLE: completeness critic. The other agents' results follow as JSON: {JSON results of the other agents of this audit}
1. Check the claims for consistency with each other and with the files.
2. What was not checked (classes, folds, signals, columns, variants such as kat_v2 or kat_v1_t*)?
3. Re-run spot checks of the most consequential claims yourself: view the images of every adjudicated leak pair, every claimed false conflict merge and every claimed hidden conflict (or a seeded sample of at least 15 if there are more).
4. Decide whether r2 is fit as the stage-2 benchmark manifest; state the residual leak rate (with CI) that should be reported, and how the protocol should be described in a paper. List code defects to fix and the decisions that must go to the dataset owner, each with your recommendation.
```

## Audit of manifest r3

Workflow script `solarfcd-stage2-r3-verify`, run on 17 November 2026. Same design as the r2 audit on a fresh sample of 160 images that excluded the photo groups of the r2 queries (seed 20261118), without a cap on the number of candidates, plus an agent that sampled the near-miss bands the manifest had not linked and inspected its largest scene groups ('yakin'). Reviewer 1 was interrupted twice (a usage limit and a network error). The archived records hold the last interrupted run and its continuation, whose prompt templates are identical apart from the wording of the resume note; the continuation finished reviewer 1 from its own saved notes and then ran the merger and the critic, which read the saved results of the other agents.

Returned summaries as in the r2 audit.

### Shared context CTX (prepended to every prompt of this audit)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset with classes Electrical_Fault (EF), Healthy (H), Structural_Fault (SF), Surface_Obstruction (SO). Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. Pilot folder: <WORK_DIR>.

TOOLS. Python: <PYTHON> (torch with CUDA on an RTX 4080 Laptop 12 GB, timm, opencv-python-headless 5 with SIFT/ORB/USAC_MAGSAC — AKAZE/KAZE/BRISK are NOT available —, imagehash, PIL, numpy, pandas, scipy, scikit-learn). Locally cached timm weights (pretrained=True works offline): convnext_tiny.fb_in22k_ft_in1k, swin_tiny_patch4_window7_224.ms_in22k_ft_in1k, tf_efficientnetv2_s.in21k_ft_in1k, resnetv2_101.a1h_in1k, vit_base_patch14_dinov2.lvd142m. Always run python with -B. Put your scripts and ALL outputs only in your own folder <WORK_DIR>/dogrulama/asama2_r3/<LABEL>/ (create it; if it already exists from an interrupted run, you may reuse files only after re-verifying them). Do not modify or create any other file. Other agents share the GPU: keep your GPU memory under ~3 GB.

OBJECT UNDER TEST. The stage-2 grouped 5-fold manifest "r3": <WORK_DIR>/asama2/manifest_asama2.csv (MD5 {MD5 of the manifest under test}), built by <WORK_DIR>/08_asama2_manifest.py from visual labels compiled by <WORK_DIR>/08a_gorsel_etiket_derle.py (read both docstrings first). Key columns: rel_path, path, dosya, split, label, aile, kaynak, termal, md5, width, height, kati_grup ⊆ foto_grup ⊆ sahne_grup, celiskili (foto_grup carries more than one label → excluded), uygun (= RGB and not celiskili), temsilci (one representative per foto_grup among uygun rows), kat_v1 (fold 0-4 for temsilci rows, -1 = not in v1; the PRIMARY benchmark: 2,105 images, 420-422 per fold), kat_v2 (fold for all uygun rows; sensitivity only), kat_v1_t1..t4 (same grouping, other fold seeds). Folds = StratifiedGroupKFold over sahne_grup, stratified by label x source. Other files in <WORK_DIR>/asama2/: kenarlar.csv (every grouping edge with its type), aday_eslesmeleri.csv (all 159,038 verified candidate pairs of this build, ONE file pair per pair of kati groups — the largest-pixel member of each kati group —, with deep RootSIFT/MAGSAC metrics: inl, umin = distinct inlier positions, gecerli = valid homography, alt_pay = footer share, kimlik = identity-frame deviation, filigran = watermark/template flag, kap_kucuk = inlier hull coverage of the smaller image, ncc / ncc_hp, kaynaklar = candidate sources, r70/r80 = exhaustive-screen counts, foto_kenari / sahne_kenari = whether the pair became an edge), eslesmeler.csv (match cache), celiskili_gruplar.csv, celiski_kenarlari.csv, celiski_supheleri.csv, zorunlu_ayrik_ihlalleri.csv, kat_arasi_sizinti.csv, kat_arasi_yakin_adaylar.csv (representative-level cross-fold near misses), kalibrasyon.csv, buyuk_gruplar.csv, gorsel_etiketler.csv (+ _ham: the visual decisions of the two previous audits, used as manual edges), ozet.md, ozet.json. Earlier versions: asama2/r1/, asama2/r2/.

HISTORY. r1 leaked at scene level (~18% of test images had a same-photo/same-scene partner in another fold). r2 (audited in <WORK_DIR>/dogrulama/asama2_r2/) still leaked: 14/120 = 11.7% (Wilson 95% CI 7.1-18.6%) in a blind dual-review audit; residual leaks were mostly cleaning-company job series (consecutive shots, before/after cleaning of the same roof with labels Healthy vs Surface_Obstruction), stock shoots and a product-listing series. r3 adds a high-resolution symmetric matcher (1400 px, 5000 RootSIFT keypoints), wider candidate generation (file-number windows for all families, Healthy<->SO cross-label neighbours, Kaggle clean-minus-dirty number offsets 240-310 / 350-410), every previously confirmed pair as a manual edge, and drops blind file-number edges. Your job is to try to BREAK r3. Be adversarial and quantitative; do not take the pipeline's own diagnostics on trust.

DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, sky-replaced, watermarked, re-compressed or embedded in a collage/advert. same_scene: a different photograph of the same physical installation, roof, array, panel or event (consecutive shots, before/after cleaning, another angle of the same roof, the same damaged panel, the same product unit). unrelated: different installations even if similar (same panel model, generic rooftops, same stock watermark, same advert template, same crew equipment only). A LEAK = two kat_v1 images (kat_v1 >= 0) in DIFFERENT folds that are same_photo or same_scene. Every visual judgement must come from actually viewing the images (Read tool on JPG/PNG files or montages you create; open/crop the originals when unsure; note that some files carry a wrong EXIF orientation flag, so compare raw pixel orientation too) and must be recorded with a one-line reason. Keep montages legible: at most ~2400 px wide, thumbnails >= 220 px.
```

### Sampling agent 'orneklem' (builds the sample and the candidate sheets; does not judge)

```text
[CTX]
LABEL = orneklem. ROLE: build a seeded FRESH audit sample and candidate sheets for two blind reviewers. You do NOT judge leaks yourself.
1. Sampling, fixed before any retrieval: read the manifest (only rel_path, path, dosya, aile, label, kat_v1, foto_grup), keep rows with kat_v1 >= 0. Exclude every row whose foto_grup contains a file listed in <WORK_DIR>/dogrulama/asama2_r2/orneklem/sample.csv (column rel_path; those were the previous audit's queries). Sort the remaining rows by rel_path, rng = numpy.random.default_rng(20261118), choose 160 rows without replacement. Save sample.csv (order, query_id Q000..Q159, rel_path, label, kat_v1) before computing anything else.
2. Retrieval must be INDEPENDENT of the r3 pipeline: do not use DINOv2 and do not read any file under <WORK_DIR>/asama2/ other than manifest_asama2.csv (with the columns above). Candidate pool for a query = kat_v1 >= 0 rows whose fold differs from the query's fold. Signals:
 (a) ConvNeXt-T (convnext_tiny.fb_in22k_ft_in1k, num_classes=0, 224 px, ImageNet normalisation) L2-normalised embeddings; similarity = max cosine over the 8 dihedral transforms of the candidate; top 6.
 (b) Swin-T (swin_tiny_patch4_window7_224.ms_in22k_ft_in1k) the same way; top 4.
 (c) 32x32 grey thumbnail NCC, max over the 8 dihedral transforms; top 3.
 (d) Local features: ORB (5000 kp, Hamming) and SIFT (5000 kp) on the image resized to 1024 px long side; kNN k=2 with Lowe ratio 0.8 plus a mutual check; cv2.findHomography(..., cv2.USAC_MAGSAC, 5.0); also try the horizontally mirrored candidate. Before matching, if BOTH images look like stock previews (dark/branded footer strip or tiled watermark), drop keypoints in the bottom 12% strip. Run it against the union of the query's top 80 ConvNeXt and top 40 Swin cross-fold neighbours; keep the top 3 by inliers if inliers >= 12.
 (e) Metadata: same file-name family with |N difference| <= 10 for all families (Imgclean_N_0, Imgdirty_N_1, Clean (N), Bird (N), Dust (N), Snow (N), Electrical (N), Physical (N)), nearest 4; Kaggle before/after offsets: for an Imgclean_N query, Imgdirty_M with N-M in [240,310] or [350,410] (and the reverse for Imgdirty queries), top 3 by ConvNeXt similarity; same EXIF camera model (tag 272) with DateTimeOriginal (36867, else 306) on the same day, nearest 3.
3. NO CAP: include every de-duplicated candidate from the signals above (expect about 15-25 per query). Save candidates.csv (query_id, cand_id, query_rel, cand_rel, label_q, label_c, fold_q, fold_c, signals, scores, sheet).
4. Sheets: one block per query = the query (thick red border) followed by its candidates, thumbnails about 230 px in RAW pixel orientation (as PIL opens them without exif_transpose), each with a caption: short file name, label, fold, signals and scores, and its id Qqqq-Ccc; if an image's EXIF orientation tag is not 1, add a small second thumbnail with the EXIF orientation applied and mark it. Wrap onto further lines as needed; 2 queries per sheet; sheets <= 2400 px wide (sheet_00.jpg ...). View at least four sheets yourself to make sure they are legible, and fix them if not.
Return the folder path, counts, the signals actually used, and any problem. Do NOT label pairs.
```

### Blind reviewers 'gozden1' and 'gozden2' ({k} = 1 or 2, {other} = the other reviewer)

```text
[CTX]
LABEL = gozden{k}. ROLE: blind reviewer {k} of 2. Another reviewer labels the same sheets independently: do NOT open <WORK_DIR>/dogrulama/asama2_r3/gozden{other}/, and do NOT look at the manifest's group columns (kati_grup, foto_grup, sahne_grup), at asama2/kenarlar.csv / aday_eslesmeleri.csv / gorsel_etiketler.csv, or at the previous audits' folders before you have finished judging and frozen your labels (record the MD5 of labels.csv when frozen).
INPUT: <WORK_DIR>/dogrulama/asama2_r3/orneklem/ (sheet_*.jpg, candidates.csv, sample.csv).
TASK: view every sheet. For every candidate decide same_photo / same_scene / unrelated / unsure relative to its query. For anything that could be same_photo or same_scene, open both originals at full resolution (crop and zoom regions with your own small script; check raw and EXIF-applied orientation) before deciding. Evidence that counts: identical roof, tiles, chimney, vents, trees, skyline, fence, pool, wall ornaments or street geometry; identical dirt, droppings, cracks, burn marks or snail-trail patterns; the same impact point on a broken panel; the same person or vehicle together with the same roof; matching or adjacent stock-photo IDs from the same contributor; the same product unit on the same stand. Generic similarity (same panel type, similar sky, same watermark, crew equipment alone) is NOT enough.
Write labels.csv in your folder (query_id, cand_id, query_rel, cand_rel, judgement, confidence high/medium/low, reason). A query is a leak if any of its candidates is same_photo or same_scene (folds always differ by construction).
Return the counts, the labels.csv path and every same_photo / same_scene / unsure pair with your reason. After freezing, you may look at asama2/aday_eslesmeleri.csv and kenarlar.csv to note (without changing labels) whether each positive pair was a candidate/edge.{for reviewer 1 only: the RESUME note below}
```

### RESUME note appended to the prompt of reviewer 1 after the two interruptions

```text
RESUME (reviewer 1 only). Your earlier run was cut off twice (account usage limit at 12:24, then a network error on retry) before it wrote labels.csv. Your own per-sheet notes are in <WORK_DIR>/dogrulama/asama2_r3/gozden1/notlar/sheet_00.txt ... sheet_75.txt (one line per candidate: cand_id|judgement|confidence|reason; '#' lines are query notes; 2,572 of the 2,703 candidates), with your helper scripts gor.py / tab.py and saved views in gozden1/gorunum/. Do NOT redo those sheets: keep their judgements as they are, except where a line is missing, malformed or ambiguous, which you must re-judge (log every such case in gozden1/devam_notu.txt). Then judge sheets 76-79 (the remaining 131 candidates) exactly as specified above, write labels.csv for ALL 2,703 candidates (check that the count matches candidates.csv and that every cand_id occurs once), freeze it, record its MD5, and return as specified. The blindness rules above still apply in full: do not open gozden2/, any birlestir folder (including _kesinti_birlestir_tek_gozden_20261117/), the folder _tamamlanan/ (other agents' results), the group columns or the pipeline files before freezing.
```

### Merger and third reviewer 'birlestir'

```text
[CTX]
LABEL = birlestir. ROLE: merge the two blind reviews, adjudicate as third reviewer, and explain every confirmed leak.
INPUTS: <WORK_DIR>/dogrulama/asama2_r3/orneklem/ (sample.csv, candidates.csv, sheets), .../gozden1/labels.csv, .../gozden2/labels.csv. Their returned summaries: PREP={JSON summary returned by the sampling agent} REVIEWER1={JSON summary returned by reviewer 1} REVIEWER2={JSON summary returned by reviewer 2}
1. Pair level: cross-table of the two reviewers' judgements; Cohen's kappa for leak-type (same_photo or same_scene) vs unrelated, unsure reported separately.
2. Query level: leak rate per reviewer, union and consensus, each with a Wilson 95% CI, extrapolated to the 2,105 kat_v1 images. Remember the sample excluded the previous audit's 120 queries (state the sampling frame size).
3. Adjudication: view every pair where the reviewers disagree or either says unsure, and every pair either reviewer called a leak; decide yourself with a reason (open the originals). Report the adjudicated per-query leak rate with Wilson CI, per fold, and per class of the query. Write pairs_final.csv (all pairs with j1, j2, your j3 and final_leak).
4. For every adjudicated leak pair, explain why r3 missed it: look up both files in asama2/manifest_asama2.csv (kati/foto/sahne groups, folds); find the pair of their kati groups in asama2/aday_eslesmeleri.csv (one file pair per kati-group pair; match on the kati groups of rel_i/rel_j) and report inl, umin, gecerli, alt_pay, kimlik, filigran, ncc_hp, kaynaklar, or that the pair was never a candidate; say which rule or threshold would have caught it and estimate its cost from aday_eslesmeleri.csv (extra edges, size of the largest kat_v1 group, labelled negatives from gorsel_etiketler.csv that it would link).
Return the numbers, the adjudicated leak list and the per-leak diagnosis.
NOTE: write into a NEW folder <WORK_DIR>/dogrulama/asama2_r3/birlestir/. Ignore <WORK_DIR>/dogrulama/asama2_r3/_kesinti_birlestir_tek_gozden_20261117/: an earlier merge attempt started there with reviewer 1's incomplete notes after reviewer 1 had failed; do not reuse anything from it.
```

### Census of the minority classes 'azinlik'

```text
[CTX]
LABEL = azinlik. ROLE: exhaustive census of the minority classes under r3 (kat_v1 has 34 SF and 72 EF images).
For every kat_v1 image of class Structural_Fault or Electrical_Fault, compare it with EVERY kat_v1 image in a different fold (all classes) using a local-feature matcher that differs from r3's (r3 uses RootSIFT at 1400 px with 5000 keypoints, symmetric mutual NN + Lowe 0.8, MAGSAC 5 px). Suggested: plain SIFT on the image capped at 1600 px long side with up to 6000 keypoints, Lowe 0.75 plus cross-check, cv2.USAC_MAGSAC 5 px, also mirrored, plus ORB; add global-embedding neighbours from ConvNeXt-T and Swin-T (dihedral max cosine), top 10 cross-fold each. View, for every SF/EF image, its strongest cross-fold candidates (at least the top 5 by inliers and the top 5 by each embedding) and judge same_photo / same_scene / unrelated. Pay special attention to the cases the previous audit flagged: Electrical (90)/(91)/(115) vs (96); the Electrical (151)/(152)/(153) series and Electrical (4); Electrical (140) vs (6); Physical (12)-P vs Physical (68)/(75); collages containing other images.
Report: every cross-fold same_photo / same_scene pair involving SF or EF (files, folds, reason); the number of distinct SF and EF scenes per fold under r3; whether one scene dominates a fold's SF or EF test set; and, as an aside, SF/EF images that look mislabelled (with reason).
```

### Near-miss bands and largest scene groups 'yakin'

```text
[CTX]
LABEL = yakin. ROLE: (1) estimate the leak yield of the near-miss bands r3 did NOT link, and (2) check whether the largest r3 scene groups are genuine series or chains of unrelated installations.
1. From asama2/kat_arasi_yakin_adaylar.csv (representative-level cross-fold pairs; columns temsilci_i/temsilci_j are the kat_v1 representatives) and asama2/aday_eslesmeleri.csv, define bands among cross-fold, not-linked representative pairs: (A) gecerli and inl 6-9 with umin >= 6; (B) gecerli and inl >= 10 but altbilgi or filigran flagged; (C) DINOv2 S >= 0.95 with no geometry (inl < 6); (D) candidate source contains 'ofset' or 'once_sonra' (Healthy vs SO) with inl >= 5; (E) same file-name family with |N difference| <= 2 (any family) and S >= 0.90. Draw a seeded random sample (numpy default_rng(7)) of up to 25 pairs per band, view each pair (open originals when needed) and judge same_photo / same_scene / unrelated / unsure. Report the yield per band with Wilson CIs and the extrapolated number of leaking pairs per band.
2. For each of the 3 largest sahne_grup by kat_v1 count (see asama2/buyuk_gruplar.csv), view a seeded sample of up to 25 member images (montage) and the edges holding the group together (asama2/kenarlar.csv): is it one installation / one company's job series, or are unrelated installations chained by false edges? List false edges you find (pair + edge type + reason). Over-grouping is leak-safe; report it as a fold-balance / semantics issue.
```

### Code and construction auditor 'kod'

```text
[CTX]
LABEL = kod. ROLE: recompute and attack the r3 construction.
1. Rebuild the partitions from asama2/kenarlar.csv as connected components (kati from type kati; foto from kati + foto_geometrik + foto_gorsel; sahne from all types) and compare them with the manifest columns.
2. Verify the bookkeeping: celiskili = foto_grup with more than one label; uygun = not termal and not celiskili; exactly one temsilci per foto_grup with uygun rows (largest width*height, then shortest rel_path, then lexicographic); kat_v1 >= 0 exactly on temsilci rows and kat_v2 >= 0 exactly on uygun rows; every sahne_grup is in one fold for kat_v1, kat_v2 and every kat_v1_t*; reproduce the folds with sklearn StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) on the temsilci rows in manifest order with y = label + '|' + kaynak and groups = sahne_grup (seed 42 for kat_v1, 1..4 for t1..t4).
3. Check every number in asama2/ozet.md / ozet.json against your own recomputation, and check that the edge rules in kenarlar.csv follow from aday_eslesmeleri.csv and gorsel_etiketler.csv as the docstring says (including the cross-label same-photo threshold 0.95 and the cannot-link rule for same-photo edges).
4. Reproducibility: from <WORK_DIR> as working directory run "python -B 08_asama2_manifest.py --gorsel asama2/gorsel_etiketler.csv --cikti <your folder>/yeniden --onbellek <your folder>/yeniden_onbellek" (do NOT run 08a_gorsel_etiket_derle.py: it writes into asama2). This recomputes hashes, DINOv2 embeddings, deep SIFT features, the GPU screen and all ~159k matches into your folder only (about 50 minutes; it needs about 3-4 GB of GPU memory). Compare the resulting manifest with the official one (MD5; if different, which rows/columns differ and why).
5. Read 08_asama2_manifest.py and 08a_gorsel_etiket_derle.py critically: the symmetric matcher (both-way ratio test, MAGSAC, symmetric validity), NCC (normalised convolution, frame choice), candidate generation, edge rules, the cannot-link logic, label compilation and its override rule, representative-level diagnostics, caches and their validation, and the -1 conventions. Report concrete bugs with a failing input where possible.
6. Check compatibility with <WORK_DIR>/02_egit.py --protocol grouped --fold-col kat_v1 --group-col sahne_grup by replaying bolmeleri_hazirla / grup_ayir in your own script (do NOT train; do NOT write into <WORK_DIR>/runs). Report inner-validation class counts per fold.
Return defects with severity and evidence.
```

### Completeness critic 'elestirmen'

```text
[CTX]
LABEL = elestirmen. ROLE: completeness critic. The other agents' results follow as JSON: {JSON results of the other agents of this audit}
1. Check the claims for consistency with each other and with the files.
2. What was not checked (classes, folds, signals, columns, variants such as kat_v2 or kat_v1_t*)?
3. Re-run spot checks of the most consequential claims yourself: view the images of every adjudicated leak pair and every claimed false chain edge (or a seeded sample of at least 15 if there are more).
4. Decide whether r3 is fit as the stage-2 benchmark manifest; state the residual leak rate (with CI) that should be reported and how the protocol should be described in a paper; compare with r2 (11.7%, CI 7.1-18.6%). List code defects to fix and the decisions that must go to the dataset owner, each with your recommendation.
```

## Visual screening for manifest r4, round 1

Workflow script `solarfcd-r4-gorsel-tarama`, run on 17 November 2026. 1,442 unlinked pairs of representative images that lay in different folds of manifest r3 (file-number series, weak geometric matches, strong mutual matches) were split into five chunks of about 288 pairs. One agent screened each chunk, and every pair that it did not judge unrelated was judged again by a blind verifier with the same context. 08a_gorsel_etiket_derle.py records a pair as same photo or same scene when both agents chose one of these classes, as uncertain when only one did or either answered unsure, and otherwise as unrelated with the certainty 'probable', so that the unrelated decision of a single screener cannot veto a geometric scene edge. Pairs outside the chunk lists that the screeners of rounds 1 and 2 noted in passing were added as same-scene decisions (tag r4_tarama_rastlanti).

Each agent returned the number of pairs, the counts of same_photo, same_scene and unsure decisions, the path of its decision file (kararlar.csv) and every pair it did not judge unrelated (kod, karar, guven, gerekce = pair code, decision, confidence, reason).

### Shared context CTX (prepended to every prompt of this round)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset (classes Electrical_Fault, Healthy, Structural_Fault, Surface_Obstruction). Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. A pair's rel_i / rel_j are paths relative to <SOLARFCD_DIR>/ (forward slashes).
TOOLS. Python: <PYTHON> (opencv-python-headless, PIL, numpy, pandas). Always run python with -B. Put your scripts and ALL outputs only in your own folder (given below); create it. Do not modify or create any other file. Do not use the GPU.
TASK CONTEXT. These pairs come from a grouped cross-validation manifest; both images of every pair are in DIFFERENT folds. If they show the same photograph or the same physical scene, that is a leak that must be fixed by linking them. Your judgements will be used as manual edges, so both misses and false alarms matter; be careful and specific.
DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, sky-replaced, watermarked, re-compressed or embedded in a collage/advert. same_scene: a different photograph of the same physical installation, roof, array, panel or event (consecutive shots of one job, before/after cleaning of the same roof, another angle of the same roof, the same damaged panel, the same product unit, consecutive stock-photo IDs of the same shoot). unrelated: different installations even if similar (same panel model, generic rooftops, same stock watermark or footer, same advert template, same crew equipment only). unsure: you cannot decide even after viewing the originals at full resolution.
EVIDENCE THAT COUNTS: identical roof shape, tiles, chimney, vents, skylights, trees, skyline, fence, pool, walls, street geometry, vehicles parked in the same place; identical dirt, droppings, cracks, burn marks; the same impact point on a broken panel; the same person together with the same roof. Generic similarity is NOT enough. Panels themselves are repetitive: look at the BACKGROUND and the non-panel context.
METHOD. Build your own montages (each pair side by side, both images scaled to the same height >= 300 px, the pair code written on it; at most ~2400 px wide per montage) and view them with the Read tool. For every pair that is not clearly unrelated at montage size, open both originals at full resolution (crop and zoom regions with your own small script; check raw and EXIF-applied orientation, and also try a mirrored version). Record EVERY pair with a one-line reason.
```

### Screener ({p} = chunk number)

```text
[CTX]
LABEL = tarama_{p}. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama_{p}/.
INPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi/parca_{p}.csv (columns kod, rel_i, rel_j; about 288 pairs). Do NOT open any other file under <WORK_DIR>/dogrulama/asama2_r4_hazirlik (other reviewers' folders or girdi/havuz_tam.csv) and do not look at any grouping/manifest files: judge from the images only.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama_{p}/kararlar.csv with columns kod, rel_i, rel_j, karar (same_photo / same_scene / unrelated / unsure), guven (high / medium / low), gerekce. It must contain every kod of your input exactly once (check this with a script before you finish).
Return the counts, the csv path and every pair you did not judge unrelated (kod, karar, guven, gerekce).
```

### Blind verifier (pairs that the screener did not judge unrelated)

```text
[CTX]
LABEL = dogrulama_{p}. ROLE: blind second reviewer. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama_{p}/.
INPUT: the pairs with these codes from <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi/parca_{p}.csv: {codes of the pairs that the screener did not judge unrelated}. Judge each pair yourself from the images (montage, then full-resolution originals for every pair). Do NOT open <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama_{p}/ or any other reviewer folder; you do not know what the first reviewer decided.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama_{p}/kararlar.csv (kod, rel_i, rel_j, karar, guven, gerekce) covering exactly these codes.
Return the counts, the csv path and every pair you did not judge unrelated.
```

## Visual screening for manifest r4, round 2

Workflow script `solarfcd-r4-gorsel-tarama-tur2`, run on 17 November 2026. 514 unlinked representative pairs regardless of fold (mutual best matches among consecutive shots, mutual best clean/dirty offset partners, near misses within one fold) in two chunks of about 257 pairs, with the same design as round 1.

As in round 1.

### Shared context CTX (prepended to every prompt of this round)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset (classes Electrical_Fault, Healthy, Structural_Fault, Surface_Obstruction). Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. A pair's rel_i / rel_j are paths relative to <SOLARFCD_DIR>/ (forward slashes).
TOOLS. Python: <PYTHON> (opencv-python-headless, PIL, numpy, pandas). Always run python with -B. Put your scripts and ALL outputs only in your own folder (given below); create it. Do not modify or create any other file. Do not use the GPU.
TASK CONTEXT. These pairs come from a grouped cross-validation manifest and are currently NOT linked into the same group. If they show the same photograph or the same physical scene, they must be linked, otherwise they can end up in different folds (a leak). Your judgements will be used as manual edges, so both misses and false alarms matter; be careful and specific. Many pairs are consecutive shots of one cleaning job, or a dirty (before) and a clean (after) shot of the same roof: look hard for that.
DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, sky-replaced, watermarked, re-compressed or embedded in a collage/advert. same_scene: a different photograph of the same physical installation, roof, array, panel or event (consecutive shots of one job, before/after cleaning of the same roof, another angle of the same roof, the same damaged panel, the same product unit, consecutive stock-photo IDs of the same shoot). unrelated: different installations even if similar (same panel model, generic rooftops, same stock watermark or footer, same advert template, same crew equipment only). unsure: you cannot decide even after viewing the originals at full resolution.
EVIDENCE THAT COUNTS: identical roof shape, tiles, chimney, vents, skylights, trees, skyline, fence, pool, walls, street geometry, vehicles parked in the same place; identical dirt, droppings, cracks, burn marks; the same impact point on a broken panel; the same person together with the same roof. Generic similarity is NOT enough. Panels themselves are repetitive: look at the BACKGROUND and the non-panel context.
METHOD. Build your own montages (each pair side by side, both images scaled to the same height >= 300 px, the pair code written on it; at most ~2400 px wide per montage) and view them with the Read tool. For every pair that is not clearly unrelated at montage size, open both originals at full resolution (crop and zoom regions with your own small script; check raw and EXIF-applied orientation, and also try a mirrored version). Record EVERY pair with a one-line reason.
```

### Screener ({p} = chunk number)

```text
[CTX]
LABEL = tarama2_{p}. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama2_{p}/.
INPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi_tur2/parca_{p}.csv (columns kod, rel_i, rel_j; about 257 pairs). Do NOT open any other file under <WORK_DIR>/dogrulama/asama2_r4_hazirlik (other reviewers' folders, girdi/, girdi2/ or girdi_tur2/havuz_tam.csv) and do not look at any grouping/manifest files: judge from the images only.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama2_{p}/kararlar.csv with columns kod, rel_i, rel_j, karar (same_photo / same_scene / unrelated / unsure), guven (high / medium / low), gerekce. It must contain every kod of your input exactly once (check this with a script before you finish).
Return the counts, the csv path and every pair you did not judge unrelated (kod, karar, guven, gerekce).
```

### Blind verifier (pairs that the screener did not judge unrelated)

```text
[CTX]
LABEL = dogrulama2_{p}. ROLE: blind second reviewer. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama2_{p}/.
INPUT: the pairs with these codes from <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi_tur2/parca_{p}.csv: {codes of the pairs that the screener did not judge unrelated}. Judge each pair yourself from the images (montage, then full-resolution originals for every pair). Do NOT open <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama2_{p}/ or any other reviewer folder; you do not know what the first reviewer decided.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama2_{p}/kararlar.csv (kod, rel_i, rel_j, karar, guven, gerekce) covering exactly these codes.
Return the counts, the csv path and every pair you did not judge unrelated.
```

## Audit of manifest r4

Workflow script `solarfcd-stage2-r4-verify`, run on 18 November 2026. A fresh sample of 160 images that excluded the photo groups of the r2 and r3 queries (seed 20261119), two blind reviewers, a merger and third reviewer, a code and reproducibility auditor and a completeness critic. The 14 pairs that this audit adjudicated as leak or unsure, together with one pair found by its critic, form Part A of the author's hand check.

Returned summaries as in the r2 audit.

### Shared context CTX (prepended to every prompt of this audit)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset with classes Electrical_Fault (EF), Healthy (H), Structural_Fault (SF), Surface_Obstruction (SO). Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. Pilot folder: <WORK_DIR>.

TOOLS. Python: <PYTHON> (torch with CUDA on an RTX 4080 Laptop 12 GB, timm, opencv-python-headless 5 with SIFT/ORB/USAC_MAGSAC — AKAZE/KAZE/BRISK are NOT available —, imagehash, PIL, numpy, pandas, scipy, scikit-learn, networkx). Locally cached timm weights (pretrained=True works offline): convnext_tiny.fb_in22k_ft_in1k, swin_tiny_patch4_window7_224.ms_in22k_ft_in1k, tf_efficientnetv2_s.in21k_ft_in1k, resnetv2_101.a1h_in1k, vit_base_patch14_dinov2.lvd142m. Always run python with -B. Put your scripts and ALL outputs only in your own folder <WORK_DIR>/dogrulama/asama2_r4/<LABEL>/ (create it; if it already exists from an interrupted run, you may reuse files only after re-verifying them). Do not modify or create any other file. Other agents share the GPU: keep your GPU memory under ~3 GB.

OBJECT UNDER TEST. The stage-2 grouped 5-fold manifest "r4": <WORK_DIR>/asama2/manifest_asama2.csv (MD5 {MD5 of the manifest under test}), built by <WORK_DIR>/08_asama2_manifest.py from visual labels compiled by <WORK_DIR>/08a_gorsel_etiket_derle.py (read both docstrings first). Key columns: rel_path, path, dosya, split, label, aile, kaynak, termal, md5, width, height, kati_grup ⊆ foto_grup ⊆ sahne_grup, celiskili (foto_grup carries more than one label → excluded), uygun (= RGB and not celiskili), temsilci (one representative per foto_grup among uygun rows), kat_v1 (fold 0-4 for temsilci rows, -1 = not in v1; the PRIMARY benchmark: 2,095 images, 418-420 per fold), kat_v2 (fold for all uygun rows; sensitivity only), kat_v1_t1..t4 (same grouping, other fold seeds). Folds = StratifiedGroupKFold over sahne_grup, stratified by label x source. Other files in <WORK_DIR>/asama2/: kenarlar.csv (every grouping edge with its type), aday_eslesmeleri.csv (all verified candidate pairs, one file pair per pair of kati groups, with deep RootSIFT/MAGSAC metrics in BOTH directions: inl/umin/gecerli/alt_pay/kimlik/kap_kucuk/ncc_hp and t_* for the reverse direction; sahne_kenari = OR of both directions after vetoes; sahne_yon; ilgisiz_vetosu; foto_filigranli), gorsel_etiketler.csv (visual decisions of all previous audits and of two visual screening rounds, used as manual edges), ozet.md, ozet.json, kalibrasyon.csv, buyuk_gruplar.csv. Earlier versions and their files: asama2/r1/, r2/, r3/.

HISTORY. r2 audit: 14/120 = 11.7% of test images had a same-scene partner in another fold. r3 audit (dogrulama/asama2_r3/): 15/160 = 9.4% (Wilson 5.8-14.9%), mostly Surface_Obstruction before/after-cleaning pairs and consecutive shots of one cleaning job with weak geometry (inl 4-7), plus stock-photo shoots and a watermark veto that also killed same-viewpoint before/after pairs. r4 changes: two-direction matching (OR), narrowed watermark veto, NCC always computed on coincident frames, true-pixel-area coverage, all 'belirsiz' visual pairs become scene edges, certain 'ilgisiz' labels veto geometric scene edges, widened clean/dirty offset candidates (200-420), complete cache keys, and about 2,000 additional visually screened pairs added as manual edges (dogrulama/asama2_r4_hazirlik/). Your job is to try to BREAK r4. Be adversarial and quantitative; do not take the pipeline's own diagnostics on trust.

DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, sky-replaced, watermarked, re-compressed or embedded in a collage/advert. same_scene: a different photograph of the same physical installation, roof, array, panel or event (consecutive shots, before/after cleaning, another angle of the same roof, the same damaged panel, the same product unit, consecutive stock-photo IDs of one shoot). unrelated: different installations even if similar (same panel model, generic rooftops, same stock watermark, same advert template, same crew equipment only). A LEAK = two kat_v1 images (kat_v1 >= 0) in DIFFERENT folds that are same_photo or same_scene. Every visual judgement must come from actually viewing the images (Read tool on JPG/PNG files or montages you create; open/crop the originals when unsure; some files carry a wrong EXIF orientation flag, so compare raw pixel orientation too) and must be recorded with a one-line reason. Look hard at BACKGROUND context (neighbouring houses, trees, walls, pools, vehicles, street), because the panels themselves are repetitive. Keep montages legible: at most ~2400 px wide, thumbnails >= 220 px.
```

### Sampling agent 'orneklem' (builds the sample and the candidate sheets; does not judge)

```text
[CTX]
LABEL = orneklem. ROLE: build a seeded FRESH audit sample and candidate sheets for two blind reviewers. You do NOT judge leaks yourself.
1. Sampling, fixed before any retrieval: read the manifest (only rel_path, path, dosya, aile, label, kat_v1, foto_grup), keep rows with kat_v1 >= 0. Exclude every row whose foto_grup contains a file listed in <WORK_DIR>/dogrulama/asama2_r2/orneklem/sample.csv or <WORK_DIR>/dogrulama/asama2_r3/orneklem/sample.csv (column rel_path; the previous audits' queries). Sort the remaining rows by rel_path, rng = numpy.random.default_rng(20261119), choose 160 rows without replacement. Save sample.csv (order, query_id Q000..Q159, rel_path, label, kat_v1) before computing anything else.
2. Retrieval must be INDEPENDENT of the r4 pipeline: do not use DINOv2 and do not read any file under <WORK_DIR>/asama2/ other than manifest_asama2.csv (with the columns above), and do not read dogrulama/asama2_r4_hazirlik/. Candidate pool for a query = kat_v1 >= 0 rows whose fold differs from the query's fold. Signals:
 (a) ConvNeXt-T (convnext_tiny.fb_in22k_ft_in1k, num_classes=0, 224 px, ImageNet normalisation) L2-normalised embeddings; similarity = max cosine over the 8 dihedral transforms of the candidate; top 6.
 (b) Swin-T (swin_tiny_patch4_window7_224.ms_in22k_ft_in1k) the same way; top 4.
 (c) 32x32 grey thumbnail NCC, max over the 8 dihedral transforms; top 3.
 (d) Local features: ORB (5000 kp) and SIFT (5000 kp) on the image resized to 1024 px long side; kNN k=2 with Lowe ratio 0.8 plus a mutual check; cv2.findHomography(..., cv2.USAC_MAGSAC, 5.0); also the mirrored candidate. If BOTH images look like stock previews (dark/branded footer strip or tiled watermark), drop keypoints in the bottom 12% strip. Run against the union of the query's top 80 ConvNeXt and top 40 Swin cross-fold neighbours; keep the top 3 by inliers if inliers >= 12.
 (e) Metadata: same file-name family with |N difference| <= 10 for all families (Imgclean_N_0, Imgdirty_N_1, Clean (N), Bird (N), Dust (N), Snow (N), Electrical (N), Physical (N)), nearest 4; Kaggle before/after offsets: for an Imgclean_N query, Imgdirty_M with N-M in [180,440] (and the reverse for Imgdirty queries), top 4 by ConvNeXt similarity; same EXIF camera model (tag 272) with DateTimeOriginal (36867, else 306) on the same day, nearest 3. Also apply (e) to every file in the query's foto_grup (some photos exist under two family names), mapping hits to their kat_v1 representative (same foto_grup).
3. NO CAP: include every de-duplicated candidate (expect about 15-25 per query). Save candidates.csv (query_id, cand_id, query_rel, cand_rel, label_q, label_c, fold_q, fold_c, signals, scores, sheet).
4. Sheets: one block per query = the query (thick red border) followed by its candidates, thumbnails about 230 px in RAW pixel orientation, each captioned with short file name, label, fold, signals, scores and id Qqqq-Ccc; if an image's EXIF orientation tag is not 1, add a small second thumbnail with the orientation applied. 2 queries per sheet; sheets <= 2400 px wide (sheet_00.jpg ...). View at least four sheets yourself to make sure they are legible.
Return the folder path, counts, the signals actually used, and any problem. Do NOT label pairs.
```

### Blind reviewers 'gozden1' and 'gozden2' ({k} = 1 or 2, {other} = the other reviewer)

```text
[CTX]
LABEL = gozden{k}. ROLE: blind reviewer {k} of 2. Another reviewer labels the same sheets independently: do NOT open <WORK_DIR>/dogrulama/asama2_r4/gozden{other}/, and do NOT look at the manifest's group columns (kati_grup, foto_grup, sahne_grup), at asama2/kenarlar.csv / aday_eslesmeleri.csv / gorsel_etiketler.csv, at dogrulama/asama2_r4_hazirlik/ or at the previous audits' folders before you have finished judging and frozen your labels (record the MD5 of labels.csv when frozen).
INPUT: <WORK_DIR>/dogrulama/asama2_r4/orneklem/ (sheet_*.jpg, candidates.csv, sample.csv).
TASK: view every sheet. For every candidate decide same_photo / same_scene / unrelated / unsure relative to its query. For anything that could be same_photo or same_scene, open both originals at full resolution (crop and zoom with your own small script; check raw and EXIF-applied orientation) before deciding. Evidence that counts: identical roof, tiles, chimney, vents, trees, skyline, fence, pool, wall ornaments or street geometry; identical dirt, droppings, cracks, burn marks; the same impact point on a broken panel; the same person or vehicle together with the same roof; matching or adjacent stock-photo IDs from the same contributor; the same product unit on the same stand. Generic similarity (same panel type, similar sky, same watermark, crew equipment alone) is NOT enough.
Work sheet by sheet and write your per-sheet decisions to <WORK_DIR>/dogrulama/asama2_r4/gozden{k}/notlar/sheet_NN.txt as you go (so an interrupted run can resume from them; if such notes already exist from an interrupted run of yours, keep them and continue).
Write labels.csv in your folder (query_id, cand_id, query_rel, cand_rel, judgement, confidence high/medium/low, reason), covering every candidate exactly once. A query is a leak if any of its candidates is same_photo or same_scene.
Return the counts, the labels.csv path and every same_photo / same_scene / unsure pair with your reason.
```

### Merger and third reviewer 'birlestir'

```text
[CTX]
LABEL = birlestir. ROLE: merge the two blind reviews, adjudicate as third reviewer, and explain every confirmed leak.
INPUTS: <WORK_DIR>/dogrulama/asama2_r4/orneklem/ (sample.csv, candidates.csv, sheets), <WORK_DIR>/dogrulama/asama2_r4/gozden1/labels.csv, <WORK_DIR>/dogrulama/asama2_r4/gozden2/labels.csv. Their returned summaries: PREP={JSON summary returned by the sampling agent} REVIEWER1={JSON summary returned by reviewer 1} REVIEWER2={JSON summary returned by reviewer 2}
1. Pair level: cross-table of the two reviewers' judgements; Cohen's kappa for leak-type (same_photo or same_scene) vs unrelated, unsure reported separately.
2. Query level: leak rate per reviewer, union and consensus, each with a Wilson 95% CI, extrapolated to the 2,095 kat_v1 images (state the sampling frame size after excluding the r2 and r3 queries).
3. Adjudication: view every pair where the reviewers disagree or either says unsure, and every pair either reviewer called a leak; decide yourself with a reason (open the originals). Report the adjudicated per-query leak rate with Wilson CI (main figure), an upper figure counting unsure-only queries, per fold, and per class of the query. Write pairs_final.csv (all pairs with j1, j2, your j3 and final_leak). Compare with r3 (15/160 = 9.4%, 5.8-14.9%) and r2 (14/120) with Fisher's exact test.
4. For every adjudicated leak pair, explain why r4 missed it (groups, folds, the kati-group pair in asama2/aday_eslesmeleri.csv with both-direction metrics, or never a candidate; any visual label) and whether it is split also in kat_v1_t1..t4.
Return the numbers, the adjudicated leak list and the per-leak diagnosis.
```

### Code and reproducibility auditor 'kod'

```text
[CTX]
LABEL = kod. ROLE: recompute and attack the r4 construction.
1. Rebuild the partitions from asama2/kenarlar.csv as connected components (kati from type kati; foto from kati + foto_geometrik + foto_gorsel; sahne from all types) and compare them with the manifest columns.
2. Verify the bookkeeping: celiskili, uygun, temsilci (largest width*height, then shortest rel_path, then lexicographic), kat_v1 >= 0 exactly on temsilci rows, kat_v2 >= 0 exactly on uygun rows; every sahne_grup in one fold for kat_v1, kat_v2 and every kat_v1_t*; reproduce the folds with StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed) on the temsilci rows in manifest order with y = label + '|' + kaynak and groups = sahne_grup (seed 42 for kat_v1, 1..4 for t1..t4).
3. Check every number in asama2/ozet.md / ozet.json and that the edge rules in kenarlar.csv follow from aday_eslesmeleri.csv and gorsel_etiketler.csv as the docstring says: two-direction OR, the narrowed watermark veto (kimlik < 0.10 and not ncc_hp >= 0.6 and (alt_pay >= 0.5 or kap_kucuk < 0.05 or S < 0.85)), the certain-'ilgisiz' veto of geometric scene edges, 'belirsiz' visual edges in all classes, the watermarked same-photo rule, the cross-label same-photo threshold 0.95 and the cannot-link rule. Check that 08a_gorsel_etiket_derle.py compiles the visual decisions faithfully (spot-check 30 random rows of gorsel_etiketler_ham.csv against their source files, including the r4 screening folders).
4. Reproducibility: from <WORK_DIR> as working directory run "python -B 08_asama2_manifest.py --gorsel asama2/gorsel_etiketler.csv --cikti <your folder>/yeniden --onbellek <your folder>/yeniden_onbellek" (do NOT run 08a: it writes into asama2). This recomputes hashes, DINOv2 embeddings, features and all matches in both directions into your folder only (about 60-70 minutes, 3-4 GB GPU). Compare the resulting manifest with the official one (MD5; if different, which rows/columns differ and why).
5. Read 08_asama2_manifest.py and 08a_gorsel_etiket_derle.py critically and report concrete bugs with failing inputs where possible.
6. Fold composition: the largest sahne_grup now holds 226 kat_v1 images. Look at which edges hold it together (networkx bridges/articulation points at foto_grup level; view a seeded sample of 20 of its edges) and say whether it is one company's job archive or a chain of unrelated installations; report class/source balance per fold for kat_v1 and kat_v1_t*, and the inner-validation SF/EF counts that <WORK_DIR>/02_egit.py (--protocol grouped --fold-col kat_v1 --group-col sahne_grup) would produce (replay bolmeleri_hazirla/grup_ayir; do NOT train; do NOT write into runs).
Return defects with severity and evidence.
```

### Completeness critic 'elestirmen'

```text
[CTX]
LABEL = elestirmen. ROLE: completeness critic. The other agents' results follow as JSON: {JSON results of the other agents of this audit}
1. Check the claims for consistency with each other and with the files.
2. What was not checked (classes, folds, signals, columns, variants such as kat_v2 or kat_v1_t*)?
3. Re-run spot checks of the most consequential claims yourself: view the images of every adjudicated leak pair (or a seeded sample of at least 15 if there are more), and look for leaks the sample could not see (e.g. partners never generated as candidates) by viewing the file-number neighbours (|dN| <= 3) and clean/dirty offset neighbours of every leak and of 10 random sample queries.
4. Decide whether r4 is fit as the stage-2 benchmark manifest; state the residual leak rate (with CI) that should be reported and how the protocol should be described in a paper; compare with r3 (15/160 = 9.4%) and r2 (14/120 = 11.7%). List code defects to fix and the decisions that must go to the dataset owner, each with your recommendation.
```

## Targeted visual screening for manifest r4.1

Workflow script `solarfcd-r41-kaggle-tarama`, run on 18 November 2026. 2,121 unlinked pairs of Kaggle rooftop-cleaning photographs (Imgclean = after cleaning, Imgdirty = before) without an earlier visual decision (same image size with file numbers at most five apart, and best clean/dirty offset partners), regardless of fold, in nine chunks of about 236 pairs, with the same design as the r4 screening rounds. The resulting edges were added to r4, which gave the final manifest r4.1 (solarfcd_manifest.csv). Pairs noted in passing carry the tag r41_tarama_rastlanti.

As in the r4 screening rounds.

### Shared context CTX (prepended to every prompt of this round)

```text
PROJECT. SolarFCD is a public PV-panel image classification dataset. Images are READ-ONLY (never modify, move or delete): <SOLARFCD_DIR>/{Train,Valid,Test}/<class>/. A pair's rel_i / rel_j are paths relative to <SOLARFCD_DIR>/ (forward slashes).
TOOLS. Python: <PYTHON> (opencv-python-headless, PIL, numpy, pandas). Always run python with -B. Put your scripts and ALL outputs only in your own folder (given below); create it. Do not modify or create any other file. Do not use the GPU.
TASK CONTEXT. All pairs are Kaggle rooftop-cleaning photos (Imgclean = after cleaning, Imgdirty = before) that are currently NOT linked into the same group of a grouped cross-validation manifest. If they show the same photograph or the same physical scene, they must be linked, otherwise they can end up in different folds (a leak). Earlier audits showed that the remaining leaks are exactly this kind: consecutive shots of one cleaning job and dirty/clean shots of the same roof, often with only subtle shared background cues. Your judgements become manual edges, so both misses and false alarms matter.
DEFINITIONS. same_photo: the same photograph, possibly cropped, rescaled, rotated, mirrored, recoloured, watermarked or re-compressed. same_scene: a different photograph of the same physical roof/array/house (consecutive shots of one job, before/after cleaning, another angle of the same roof). unrelated: different houses even if similar (same panel model, same tile type, same crew equipment only). unsure: you cannot decide even after viewing the originals at full resolution.
EVIDENCE THAT COUNTS: identical roof shape, tiles, chimney, vents, skylights, conduit runs, trees, skyline, fence, pool, walls, garden furniture, street geometry, neighbouring houses, vehicles parked in the same place; identical array layout (same number of rows/modules in the same arrangement, same tilt legs); identical dirt, droppings or damage. Generic similarity is NOT enough. Panels are repetitive: look at the BACKGROUND and the array layout.
METHOD. Build your own montages (each pair side by side, both images scaled to the same height >= 300 px, the pair code written on it; at most ~2400 px wide per montage) and view them with the Read tool. For every pair that is not clearly unrelated at montage size, open both originals at full resolution (crop and zoom regions with your own small script; also try a mirrored and a 180-degree rotated version). Record EVERY pair with a one-line reason. Work in batches and save your decisions to a notes file as you go, so an interrupted run can resume from them (if such notes already exist in your folder from an interrupted run of yours, keep them and continue).
```

### Screener ({p} = chunk number)

```text
[CTX]
LABEL = tarama41_{p}. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama41_{p}/.
INPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi_r41b/parca_{p}.csv (columns kod, rel_i, rel_j; about 236 pairs). Do NOT open any other file under <WORK_DIR>/dogrulama/asama2_r4_hazirlik (other reviewers' folders, other girdi folders, girdi_r41b/havuz_tam.csv) and do not look at any grouping/manifest files: judge from the images only.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama41_{p}/kararlar.csv with columns kod, rel_i, rel_j, karar (same_photo / same_scene / unrelated / unsure), guven (high / medium / low), gerekce. It must contain every kod of your input exactly once (check this with a script before you finish).
Return the counts, the csv path and every pair you did not judge unrelated (kod, karar, guven, gerekce).
```

### Blind verifier (pairs that the screener did not judge unrelated)

```text
[CTX]
LABEL = dogrulama41_{p}. ROLE: blind second reviewer. Your folder: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama41_{p}/.
INPUT: the pairs with these codes from <WORK_DIR>/dogrulama/asama2_r4_hazirlik/girdi_r41b/parca_{p}.csv: {codes of the pairs that the screener did not judge unrelated}. Judge each pair yourself from the images (montage, then full-resolution originals for every pair). Do NOT open <WORK_DIR>/dogrulama/asama2_r4_hazirlik/tarama41_{p}/ or any other reviewer folder; you do not know what the first reviewer decided.
OUTPUT: <WORK_DIR>/dogrulama/asama2_r4_hazirlik/dogrulama41_{p}/kararlar.csv (kod, rel_i, rel_j, karar, guven, gerekce) covering exactly these codes.
Return the counts, the csv path and every pair you did not judge unrelated.
```
