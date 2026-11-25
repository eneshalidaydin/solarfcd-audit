# SolarFCD audit: leakage, label ambiguity and domain shift in image-based PV fault detection

This repository accompanies the article:

> E.H. Aydın, *Leakage, label ambiguity and domain shift in image-based photovoltaic fault classification and implications for operation and maintenance*, submitted to *Renewable Energy* (2026).

Archived on Zenodo: [https://doi.org/10.5281/zenodo.24500535](https://doi.org/10.5281/zenodo.24500535) (concept DOI, which always resolves to the latest version). This is version 1.1.0 of 25 November 2026, prepared for the manuscript submitted to *Renewable Energy* (see [Version history](#version-history)).

It provides everything needed to reproduce the analyses **except the images themselves and the trained model weights**:

- the duplicate and scene groups for the SolarFCD benchmark;
- the leak-controlled folds;
- the expert labels and the author's hand check of the language-model pair decisions;
- the overlap screening of the external datasets and the mapping of their labels;
- the model predictions for all protocols (P1 official split, P2 random folds, P3 scene-grouped folds and external data);
- the analysis scripts, the prompts of the language-model reviewers and an environment file.

The images must be obtained from their original sources (see [Data sources and licences](#data-sources-and-licences)). Trained model weights are not included, so the external predictions and the Grad-CAM maps require retraining with the scripts. All files are stored in the repository root. The files of the additional analyses share the prefix `revision_analyses_`.

## Data files

| File | Content |
|---|---|
| `solarfcd_manifest.csv` | All 3,742 files of the SolarFCD release with MD5 hash, source family, photo group, scene group, label conflicts, the 2,095-image evaluation set, folds and expert-audited labels |
| `link_edges.csv` | Edges used to build photo and scene groups (MD5/pHash, geometric matching, DINOv2, EXIF, visual review) |
| `visual_pair_reviews.csv` | Visual decisions on 16,303 candidate image pairs (same photo, same scene, uncertain or unrelated) used for grouping and for the residual-leakage audit. `review_round` names the review rounds (see `llm_review_prompts.md`), and `human_checked` and `human_check_id` mark the 115 decisions that the author checked by hand |
| `insan_kararlari.csv` | The author's blinded hand decisions on these 115 pairs and on 8 Electrical_Fault and Structural_Fault images whose labels a language-model reviewer had queried, kept as recorded (see [Key columns](#key-columns)) |
| `expert_audit.csv` | Blinded labels of three PV experts for 150 images (112 model-flagged candidates and 38 random controls) and the majority consensus |
| `external_overlap_kaggle_pv_panel_defect.csv`, `external_overlap_rf_fault.csv`, `external_overlap_rf_soiling.csv` | Overlap screening of the external datasets against SolarFCD (MD5, pHash, DINOv2 cosine) |
| `external_label_mapping.csv` | Mapping of the external folder and box labels to the four SolarFCD classes: the RF-Fault priority rule, the RF-Soiling rule and the Kaggle folder mapping, with the number of images per box-class combination before and after overlap screening |

## Result files

| File | Content |
|---|---|
| `oof_predictions.csv` | Out-of-fold class probabilities (single view) for every model, seed and fold under protocols P2, P3 and P3 with audited labels |
| `p1_predictions.csv` | Class probabilities (single view) of the five P1 models (official split, seed 42) on the 375 official test images. Table 3 reports the accuracy with test-time augmentation (TTA) that the training script stored in `results.json`; the per-image TTA probabilities of P1 were not saved |
| `external_predictions_rf_fault.csv`, `external_predictions_rf_soiling.csv` | Class probabilities of the SolarFCD-trained models on the screened external images, with TTA. For each backbone (`ConvNeXt`, `Swin`, `DINOv2-224`, `DINOv2-392`) they are the mean of the five fold models of the three seeds, and the columns `*_P2_seed42` and `*_P3_seed42` average the seed-42 fold models of P2 and P3. E3 is the mean of the `ConvNeXt`, `Swin` and `DINOv2-392` columns |
| `few_shot_adaptation.csv` | Few-shot adaptation results on RF-Fault (`28_az_ornek_uyarlama.py`) |
| `az_ornek_gruplu.csv`, `az_ornek_gruplu.txt` | Grouped few-shot adaptation on RF-Fault, as written by `28b_az_ornek_gruplu.py`. Columns of the CSV: `tekrar` (repetition), `n_test`, `test_cogunluk` (majority-class share of the test half), `butce` (target budget), `n_egitim` (adaptation images), `yontem` (method: `ince_ayar_solarfcd` and `ince_ayar_imagenet` = fine-tuning from SolarFCD or ImageNet weights, `1nn_*` and `dogrusal_*` = 1-NN and linear probe on frozen features), `acc`, `rec_H`, `rec_SF`, `rec_SO`, `f1`. The TXT file holds the means per budget and method |
| `original_alpha*.csv`, `original_ambiguous_removed_alpha*.csv`, `audited_alpha*.csv` | Per-split results of the Learn-then-Test selective classifier and LAC conformal sets (1,000 splits each, α = 2, 5 and 10%) under the three label conditions |
| `05b_label_copy_baseline_out.txt` | Output of `05b_label_copy_baseline.py`: the training-free label-copy baseline on the official split (168/375 = 44.8%) |

## Documentation and environment

| File | Content |
|---|---|
| `README.md` | This file |
| `llm_review_prompts.md` | Prompts of the language-model reviewers, overview of the review rounds and design of the author's hand check |
| `environment.yml` | Exact conda environment (`conda env export -n solarfcd` on Windows 11, x64, with the local `prefix` line removed) |
| `CITATION.cff`, `LICENSE`, `LICENSE-DATA.md` | Citation metadata and licences |

## Code files

The numbered scripts are the analysis scripts in execution order. Their comments and docstrings are in Turkish, except in `05b_label_copy_baseline.py`, so the table gives a one-line English description of each. Scripts marked *New* were added in version 1.1.0.

| Script | Description |
|---|---|
| `ortak.py` | Shared helpers: image loading, split and class from the folder names, UTF-8 console |
| `01_veri_denetimi.py` | Inventory of the released archive (split and class from folder names), perceptual hashes, optional deep near-duplicate search (`--derin`) and report of near-duplicates across splits |
| `02_egit.py` | Training and evaluation with timm and PyTorch; used for P1 (official split, early stopping on validation macro-F1, TTA metrics in `results.json`) |
| `02b_egit_asama2.py` | `02_egit.py` with options for inner validation and checkpoint selection (`--val-mode`, `--select`); used for all P2 and P3 runs without inner validation, evaluating the last epoch |
| `03_pilot.py` | Runs P1: smoke test, five backbones and a DINOv2 linear probe on the official split, then `04` |
| `04_sonuclari_topla.py` | *New.* Collects `runs/*/results.json` into one summary table together with the reference result of the SolarFCD article |
| `05_ek_analiz.py` | *New.* Source family × split × class counts, near-duplicates across splits, and P1 accuracy by family and for test files with and without a training copy |
| `05b_label_copy_baseline.py` | *New.* Training-free label-copy baseline on the official split (44.8%), re-implemented as a stand-alone script from the audit recorded during the study |
| `06_ham_kaynak_eslesme.py` | Traces every SolarFCD file to the raw source archives by MD5 (archives read in memory) |
| `07_saglamlik_analizi.py` | *New.* Robustness of P1: subsets (conflicting, de-duplicated, leak-free), McNemar tests, the thermal images and the effect of checkpoint selection (`--valid-termal`) |
| `08_asama2_manifest.py` | Builds the leak-controlled grouping (release r4.1): strict groups (MD5, dihedral pHash), photo and scene groups from RootSIFT/MAGSAC matching, DINOv2, EXIF and the visual decisions, label conflicts, representatives and scene-grouped folds |
| `08a_gorsel_etiket_derle.py` | Compiles the visual pair decisions of all audit and screening rounds into one table (deposited as `visual_pair_reviews.csv`) |
| `09_asama2_egitim.py` | Runner for the P3 runs: five backbones × three seeds × five folds |
| `10_asama2_sonuclar.py` | Pooled out-of-fold metrics per model and seed with scene-cluster bootstrap CIs, and the comparison with P1 |
| `11_resnet_duyarlilik.py` | ResNetV2-101 sensitivity runs (inner validation with checkpoint selection, 60 epochs, learning rate 3e-4) |
| `12_insan_kontrol_seti.py` | *New.* Builds the hand-check set (115 pair decisions in two parts and 8 EF/SF images) and its forms |
| `12b_secimli_form.py` | Blind click form for the hand check (language-model decision and current label hidden); saves `insan_kararlari.csv` |
| `13_insan_uyum.py` | Agreement between the author's hand decisions and the language-model decisions; it also reads the two forms written by `12`, which are not deposited |
| `14_rastgele_bolme.py` | P2 runner: the same 2,095 images in random stratified five-fold CV (seed 42), five backbones |
| `15_ek_analizler.py` | Paired model comparisons (scene-cluster bootstrap), the pre-specified 224 px ensemble and the five-model ensemble, EF/SF label sensitivity and confusion matrices |
| `16_lr_ayari.py` | Per-model learning-rate tuning with inner validation |
| `17_yeni_model.py` | ConvNeXt V2-T under the P3 protocol, three seeds |
| `19_yuksek_cozunurluk.py` | High-resolution P3 runs: ConvNeXt-T at 384 px and DINOv2 at 392 px, three seeds |
| `20_etiket_denetimi_formu.py` | Selects the 150 images of the expert audit (112 model-flagged candidates, 38 stratified random controls) and writes the blinded expert forms |
| `21_cozunurluk_sonuc.py` | Resolution comparison and a preview ensemble named `Topluluk-3 yuksek` (ConvNeXt-T 384 px + Swin-T 224 px + DINOv2 392 px), which is **not** E3 |
| `22_secici_konformal.py` | Learn-then-Test selective classification and LAC conformal sets over 1,000 scene-level calibration/test splits; its default members are not E3 (see [Ensembles](#ensembles)) |
| `23_etiket_birlestir.py` | Merges the three expert answers by the pre-specified majority rule, computes Fleiss' κ and writes the audited manifest |
| `24_temiz_egitim.py` | Retrains the E3 backbones with the audited labels |
| `25_dis_dogrulama.py` | External validation: label mapping, overlap screening against all SolarFCD images and TTA predictions of the SolarFCD-trained fold models |
| `26_temiz_degerlendirme.py` | Evaluation with the audited labels: retrained versus original models, and the original models without the ambiguous images |
| `27_dis_secici_gurultu.py` | Transfer of the in-domain selective threshold and LAC sets to the external data, recalibration with 50–400 target images, and the dataset-wide label-change estimate from the controls |
| `28_az_ornek_uyarlama.py` | Few-shot adaptation on RF-Fault: 0, 5, 10 and 20% target images, ConvNeXt-T initialised from the SolarFCD fold-0 model or from ImageNet, near-duplicate groups kept together |
| `28b_az_ornek_gruplu.py` | *New.* Repeats `28` with session and sub-collection groups (ten repetitions, two-fold group split) and adds 1-NN and linear-probe baselines on frozen features; outputs in `az_ornek_gruplu.csv` and `az_ornek_gruplu.txt` |
| `29_gradcam.py` | Grad-CAM maps of ConvNeXt-T (224 px, seed 42) on out-of-fold images |
| `29b_gradcam_v2.py` | 300-dpi version of the Grad-CAM figure (maps computed on the original images, the company name of one watermark blurred for display only) |
| `30_makale_sekilleri.py` | *New.* First article figures: accuracy by evaluation setting, accuracy–coverage curve of E3 and few-shot adaptation |
| `31_makale_ek_sekiller.py` | Figures of the article and the numbers quoted in the text |
| `33_ek_materyal.py` | Supplementary tables and figures (replaced by its current version in 1.1.0) |
| `34_grafik_ozet.py` | Graphical abstract |
| `37a_uzman_ornekleri.py` | *New.* Expert-audit figure (original label versus expert consensus and four audited examples with the E3 out-of-fold prediction) and the per-expert answer distribution of Supplementary Fig. S4 |
| `37b_dis_veri_ornekleri.py` | *New.* Example images of RF-Fault and RF-Soiling with their box labels and mapped classes, and one image removed by the overlap screening next to its nearest SolarFCD image |
| `37c_etiket_butcesi.py` | *New.* Number of labelled target images needed to re-certify the 5% error bound with Learn-then-Test, computed from the Clopper–Pearson bound |
| `37d_akis_semasi.py` | *New.* Flow diagram of the classification and triage pipeline |

## Additional analyses

The 19 scripts below and their outputs were added in version 1.1.0. They were run on 24 November 2026 against the folder layout of the original analysis, that is, the pilot folder `C:/solarfcd/pilot` with the subfolders described under [Working folder](#working-folder), so their paths must be adapted before they are re-run. They read existing outputs and deposit files; `d_azornek.py` and `f_kaynak.py` also read the RF-Fault and SolarFCD images to compute frozen features (`d_azornek.py` with the fold-0 ConvNeXt-T weights), and none of them trains a model. In this repository each file carries the prefix `revision_analyses_`; without the prefix it has the name under which it was run and under which the scripts write their outputs. `<DEPOSIT_DIR>` stands for the folder with the files of this repository and `<SOLARFCD_DIR>` for the SolarFCD-Final archive. The feature caches (`*.npy`) are not included; `d_azornek.py` and `f_kaynak.py` recreate them, and `k_recal.py` needs the pHash cache written by `d_azornek.py`. Warning lines that contained local temporary paths were removed from `a_uzman_out.txt`.

| Script (`revision_analyses_` + …) | Output (`revision_analyses_` + …) | What it computes |
|---|---|---|
| `a_uzman.py` | `a_uzman_out.txt` | Expert-audit checks from saved outputs: E3 against the expert consensus for candidates and controls, agreement of each expert with the original labels, stratified accuracy against the consensus |
| `b_secici.py` | `b_secici_out.txt` | Re-run of the `22_secici_konformal.py` procedure for E3 with extra diagnostics: zero-coverage splits, median and IQR of coverage, violation rate among splits that accept images, class-wise and Mondrian LAC coverage, missed faults among accepted images |
| `c_dis.py` | `c_dis_out.txt` | External-validation checks from the deposited predictions: removals by class and screening criterion, E3 accuracy by cosine band, majority baselines |
| `d_azornek.py` | `d_azornek_out.txt`, `d_azornek.csv`, `rf_families.csv` | Rebuilds the few-shot split of `28_az_ornek_uyarlama.py` and computes file-name-family baselines and 1-NN baselines on frozen ImageNet and SolarFCD features |
| `e_protokol.py` | `e_protokol_out.txt` | Protocol checks: share of images with a scene partner in another P2 fold, P2 versus P3 accuracy by subset, E3 metrics with CIs, single-view versus TTA metrics and grouping checks |
| `f_kaynak.py` | `f_kaynak_out.txt` | Training-free check of source cues: how often 1-NN on frozen ImageNet ConvNeXt-T features recovers the source collection within a class |
| `g_egitim_egrisi.py` | `g_egitim_egrisi_out.txt` | Convergence summary (training accuracy and loss at epochs 20–30) from the logs of all P3 runs |
| `h_kappa_ci.py` | `h_kappa_ci_out.txt` | Fleiss' κ for a variable number of answers per image with 95% bootstrap CIs (all images, candidates, controls) |
| `i_p2p3_ci.py` | `i_p2p3_ci_out.txt` | Paired scene-cluster bootstrap CIs for the P2 minus P3 differences in accuracy and macro-F1 (seed 42) |
| `k_recal.py` | `k_recal_out.txt` | RF-Fault recalibration of `27_dis_secici_gurultu.py` repeated with group-level calibration/test splits (613 near-duplicate groups) |
| `r1_a_ic_alan.py` | `r1_a_sonuc.txt`, `ltt_sinif_orij.csv`, `ltt_sinif_orijbel.csv`, `ltt_sinif_temiz.csv` | In-domain checks on the E3 out-of-fold predictions: class-wise results of selective prediction per split under the three label conditions, δ sensitivity, LAC sets as actions, per-fold metrics, the largest scene group, resolution and P1 decomposition |
| `r1_a4_p1.py` | `r1_a4_sonuc.txt` | Decomposition of the P1 accuracies (all, RGB only, RGB without a training copy, test-only photo groups) next to P3 on the same images |
| `r1_b_dis_uzman.py` | `r1_b_sonuc.txt` | Checks from the deposit files on the expert audit, the external data (RF-Fault, RF-Soiling), few-shot adaptation and the pair decisions |
| `r1_c_ek.py` | `r1_c_sonuc.txt` | RF-Fault files named `Imgdirty_*` and the `IMG_####` camera sessions |
| `r1_d_tta.py` | `r1_d_sonuc.txt` | Whether the pooled P2 and P3 metrics are single-view: sums the single-view and TTA confusion matrices of `results.json` over the folds |
| `r1_e_fig4.py` | `r1_e_sonuc.txt` | Bootstrap CIs over source-file stems for the external bars of Fig. 6 (DINOv2 224 px, P3, seed 42, fold ensemble) |
| `r1_f_tablo_sayim.py` | `r1_f_sonuc.txt` | RF-Fault box-class combinations by mapped label for all and for the retained images (the counts in `external_label_mapping.csv`), and word and character counts of the abstract and highlights |
| `r1_h_isyuku.py` | `r1_h_sonuc.txt` | Review workload and errors per 1,000 images from `ltt_sinif_*.csv` |

## Key columns

**`solarfcd_manifest.csv`**

| Column | Description |
|---|---|
| `rel_path` | Path relative to the root of the SolarFCD-Final archive, e.g. `Train/Healthy/Imgclean_1364_0.jpg` |
| `official_split`, `label` | Split and class of the original release |
| `md5` | MD5 hash of the file bytes |
| `source_family` | Origin family inferred from the file name, e.g. `Imgclean`, `Imgdirty`, `Bird`, `Physical` |
| `strict_group`, `photo_group`, `scene_group` | Byte or pHash group, same-photograph group and same-scene group |
| `label_conflict` | The photo group contains different labels (excluded) |
| `in_evaluation_set` | One of the 2,095 RGB representatives used in the article |
| `fold_scene_grouped` | Fold of protocol P3 (scene-grouped five-fold CV), −1 if not in the evaluation set |
| `fold_random` | Fold of protocol P2 (random five-fold CV) |
| `ambiguous_expert` | The expert majority rule declared the image ambiguous (46 images) |
| `label_audited` | Label after the expert audit, empty if not in the evaluation set or ambiguous |

**`expert_audit.csv`.** The columns `expert1` to `expert3` hold the independent, blinded answers, whose possible values are the four classes, `mixed` and `not_sure`. The column `consensus` holds the class chosen by at least two experts and otherwise `ambiguous`, and `status` is `kept`, `changed` or `ambiguous`.

**`oof_predictions.csv` and `p1_predictions.csv`.** The columns are `protocol`, `model`, `img_size`, `seed`, `fold` (out-of-fold file only), `rel_path`, `label`, `prediction` and the class probabilities `p_*`. Each out-of-fold row is the prediction of the model trained on the other four folds. Both files hold single-view probabilities, as written by the training scripts to `test_predictions.csv`.

**`visual_pair_reviews.csv`.** `rel_path_a` and `rel_path_b` are the two images (sorted), `decision` is `same_photo`, `same_scene`, `uncertain` or `unrelated`, `certainty` is `definite` or `probable`, `n_decisions` and `all_decisions` count and list the decisions of all rounds for the pair, and `review_round` lists their sources. `human_checked` is `True` for the 115 pairs that the author checked by hand, and `human_check_id` (`K000`–`K114`) links them to `insan_kararlari.csv`.

**`insan_kararlari.csv`** (kept as recorded by the form, with Turkish column names and values). `tur` is the type of check (`cift` = image pair, `efsf` = label check of an EF/SF image), and `id` is the pair identifier `K000`–`K114` (as in `human_check_id`) or the file name of the SolarFCD image, which is unique in the release. `karar` is the author's decision: `ayni_foto` = same photo, `ayni_sahne` = same scene, `ilgisiz` = unrelated, or the class chosen for an EF/SF image. `not` is an optional note and is empty. The eight EF/SF images are `Electrical (103)`, `(73)`, `(9)`, `(129)`, `(118)` and `Physical (83)` in Train and `Electrical (13)` and `(144)` in Test (all `.jpg`).

**`external_label_mapping.csv`.** `dataset`, `source_class_ids` (YOLO class ids of the boxes in the image, empty for folder labels), `source_classes`, `rule` (the rule that applies, numbered in the order in which the rules are checked), `mapped_label`, `used_in` (the analyses that use the image), `images_all` and `images_retained` (images before and after overlap screening).

### Script columns and deposit columns

The scripts read the internal manifest `asama2/manifest_asama2.csv` and other files with Turkish column names. The deposit files contain the same information under English names.

| Deposit (`solarfcd_manifest.csv`) | Scripts (`asama2/manifest_asama2.csv`) |
|---|---|
| `rel_path` | `path` (absolute path: `<SOLARFCD_DIR>` + `rel_path` with Windows separators; the scripts index by `path`) and `rel_path` |
| `official_split`, `label`, `md5`, `width`, `height` | `split`, `label`, `md5`, `width`, `height` |
| `source_family`, `source`, `thermal` | `aile`, `kaynak`, `termal` |
| `strict_group`, `photo_group`, `scene_group` | `kati_grup`, `foto_grup`, `sahne_grup` |
| `label_conflict`, `eligible`, `representative` | `celiskili`, `uygun`, `temsilci` |
| `in_evaluation_set` | `temsilci` and `kat_v1` ≥ 0 |
| `fold_scene_grouped` | `kat_v1` (−1 outside the evaluation set) |
| `fold_random` | `kat_rastgele` in `asama2_rastgele/manifest_asama2.csv` |
| `ambiguous_expert`, `label_audited` | evaluation images with `kat_v1` = −1, and `label`, in `asama2_temiz/manifest_asama2.csv` |

| Other deposit files | Script columns and values |
|---|---|
| `visual_pair_reviews.csv` | `rel_a`, `rel_b`, `sinif`, `kesinlik`, `karar_sayisi`, `siniflar`, `kaynaklar` in `asama2/gorsel_etiketler.csv`; `ayni_foto`, `ayni_sahne`, `belirsiz`, `ilgisiz` = same_photo, same_scene, uncertain, unrelated; `kesin`, `olasi` = definite, probable |
| `link_edges.csv` | `tur`, `rel_i`, `rel_j`, `S`, `phash_d`, `sahne_grup` in `asama2/kenarlar.csv`; edge types `kati`, `foto_geometrik`, `foto_gorsel`, `sahne_geometrik`, `sahne_gorsel`, `exif`, `dino_mknn` = strict_md5_or_phash, photo_geometric, photo_visual, scene_geometric, scene_visual, scene_exif, scene_dinov2 |
| `expert_audit.csv` | `no`, `path`, `label`, `aday`, `p_etiket`, `model_tahmini`, `uzman1`–`uzman3`, `uzlasi`, `oybirligi`, `durum` in `insan_kontrol/etiket/sonuc.csv`; `karisik`, `emin_degil`, `belirsiz` = mixed, not_sure, ambiguous; `ayni`, `degisti` = kept, changed |
| `external_overlap_*.csv` | `path`, `ust`, `label` (`Ariza` = Fault_unknown_type), `md5`, `md5_eslesme`, `ph_min`, `cos_max`, `en_yakin`, `ortusen` in `ortusme.csv` of `dis_dogrulama/`, `dis_dogrulama/roboflow/` and `dis_dogrulama/melbourne/` |
| `oof_predictions.csv`, `p1_predictions.csv` | `path`, `label`, `tahmin` (= prediction), `p_*` in `test_predictions.csv` of each run folder |
| `few_shot_adaptation.csv` | `oran`, `tekrar`, `baslangic`, `n_egitim`, `f1` = target_fraction, repeat, initialisation, n_train, macro_f1 in `asama2_sonuclar/az_ornek.csv` |
| `original_alpha*.csv`, `original_ambiguous_removed_alpha*.csv`, `audited_alpha*.csv` | `kapsam`, `acc`, `ihlal`, `tam_acc`, `aps_kapsama`, `aps_boy`, `aps_tekil` = coverage, accuracy_accepted, violation, accuracy_all, lac_coverage, lac_mean_set_size, lac_singleton_rate in `asama2_sonuclar/secici_ek/orij_r*.csv`, `orijbel_r*.csv` and `temiz_r*.csv` |

## External data layout

`<EXTERNAL_DATA_DIR>` must contain the three external datasets as distributed by their publishers (Roboflow exports in YOLO format):

```text
<EXTERNAL_DATA_DIR>/
  pv_panel_defect/<train|val|test>/<Bird-drop|Clean|Dusty|Electrical-damage|Physical-Damage|Snow-Covered>/<image>
  roboflow/<train|valid|test>/images/*.jpg                       RF-Fault, YOLO export
  roboflow/<train|valid|test>/labels/*.txt                       class ids: 0 Bird Drop, 1 Defective, 2 Dust, 3 Dusty,
                                                                 4 Non Defective, 5 Physical Damage, 6 Snow
  melbourne/Soiling_DS_v2_14/<train|valid|test>/images/*.jpg     RF-Soiling, YOLO export
  melbourne/Soiling_DS_v2_14/<train|valid|test>/labels/*.txt     class ids: 0 Dust, 1 bird-droppings
```

- **Kaggle PV Panel Defect Dataset** (folder `pv_panel_defect`, 1,574 images in `.jpg`, `.jpeg` or `.png` files): `25_dis_dogrulama.py` searches the folder recursively and takes the class from the name of the parent folder (Clean → Healthy; Dusty, Bird-drop and Snow-Covered → Surface_Obstruction; Electrical-damage → Electrical_Fault; Physical-Damage → Structural_Fault).
- **RF-Fault** (folder `roboflow`, 2,763 images): the box classes of an image are mapped by priority. (1) Any Physical Damage box gives Structural_Fault. (2) Otherwise any Defective box gives a fault of unknown type (`Fault_unknown_type`), which enters only the fault/no-fault analysis. (3) Otherwise any Bird Drop, Dust, Dusty or Snow box gives Surface_Obstruction. (4) Images with only Non Defective boxes are Healthy. Images without boxes would be excluded, but none occurred. 723 of the 1,478 retained images carry boxes of more than one class.
- **RF-Soiling** (folder `melbourne`, the working name of this dataset, 302 images): any box gives Surface_Obstruction, and an image without boxes would be Healthy; all 302 images carry at least one box.
- `external_label_mapping.csv` lists every rule and box-class combination with its image counts. The `rel_path` columns of the external files are relative to `pv_panel_defect/`, `roboflow/` and `melbourne/`.

## Reproducing the analyses

### Environment

The analyses used Python 3.12.16, PyTorch 2.15.0 and torchvision 0.29.2 built for CUDA 13.2 (`+cu132` wheels from the PyTorch index `https://download.pytorch.org/whl/cu132`), timm 1.0.31, OpenCV 5.0.0.93 (`opencv-python-headless`), scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4, imagehash 4.3.2, matplotlib 3.11.2 and Pillow 13.0.0 on one NVIDIA RTX 4080 Laptop GPU (12 GB) under Windows 11. `environment.yml` lists the complete environment; on other platforms the conda build strings must be dropped. In the code, `<SOLARFCD_DIR>` must be replaced with the folder of the SolarFCD-Final archive, `<EXTERNAL_DATA_DIR>` with the folder of the external datasets and `<OUTPUT_DIR>` with the folder that receives the manuscript files. `06_ham_kaynak_eslesme.py` also needs the raw source archives of SolarFCD (`--ham`).

### Working folder

All scripts were run with the working folder as the current directory, and their inputs and outputs are stored in its subfolders: `denetim_final/` (output of `01`, including `manifest.csv` for P1), `asama2/` (internal manifest, edges, visual decisions), `asama2_rastgele/` (manifest with the P2 folds), `asama2_temiz/` (manifest with the audited labels), `runs/` (P1), `runs_asama2/` (P3 and its sensitivity runs), `runs_rastgele/` (P2), `runs_temiz/` (audited labels), `runs_lr/` (learning-rate tuning), `insan_kontrol/` (hand check and expert audit), `dis_dogrulama/` (external data), `asama2_sonuclar/` (result tables) and `makale/` (figures and supplement). A run folder is named `official_<model>_<size>_s<seed>` (P1) or `grouped<fold>_<model>_<size>_s<seed>_<fold column>_<manifest folder>_<inner validation>`, for example `grouped0_convnext_tiny_224_s42_kat_v1_asama2_none`. It holds `config.json`, `log.csv`, `results.json` (metrics, with `test` for single-view and `test_tta` for TTA predictions) and `test_predictions.csv` (single-view probabilities).

### Command lines

The commands below are those recorded in the study notes and in the runner scripts. Every script was run as `python -B <script>` with `PYTHONUTF8=1`; scripts that are not listed take no arguments.

```text
# Audit of the release and P1 (official split)
python 01_veri_denetimi.py --kok <SOLARFCD_DIR> --cikti denetim_final --derin
python 06_ham_kaynak_eslesme.py --final <SOLARFCD_DIR> --ham <RAW_DIR> --cikti denetim_ek
python 05b_label_copy_baseline.py --solarfcd <SOLARFCD_DIR>
python 03_pilot.py --veri-kok <SOLARFCD_DIR> --atla-denetim --batch 32
#   runs, for each backbone and learning rate of 09 below:
#   python 02_egit.py --model <model> --lr <lr> --epochs 30 --tta --manifest denetim_final/manifest.csv --batch-size 32 --workers 8 --img-size 224
#   and the DINOv2 linear probe with --linear-probe --lr 1e-3 --epochs 15
python 04_sonuclari_topla.py
python 05_ek_analiz.py
python 07_saglamlik_analizi.py --valid-termal

# Grouping (release r4.1)
python 08a_gorsel_etiket_derle.py
python 08_asama2_manifest.py

# P3 (75 runs), P2 and sensitivity runs
python 09_asama2_egitim.py
#   runs, for seeds 42, 43, 44 and folds 0-4, with resnetv2_101.a1h_in1k (lr 1e-4), convnext_tiny.fb_in22k_ft_in1k (1e-4),
#   tf_efficientnetv2_s.in21k_ft_in1k (1e-4), swin_tiny_patch4_window7_224.ms_in22k_ft_in1k (5e-5) and
#   vit_base_patch14_dinov2.lvd142m (2e-5):
#   python 02b_egit_asama2.py --manifest asama2/manifest_asama2.csv --protocol grouped --fold-col kat_v1 --group-col sahne_grup
#          --fold <k> --val-mode none --select last --model <model> --lr <lr> --epochs 30 --tta --batch-size 32 --workers 8
#          --seed <seed> --out runs_asama2
python 14_rastgele_bolme.py      # the same with --manifest asama2_rastgele/manifest_asama2.csv --fold-col kat_rastgele
                                 # --group-col rel_path --seed 42 --out runs_rastgele
python 19_yuksek_cozunurluk.py   # the same with --img-size 384 (ConvNeXt-T) or 392 (DINOv2); batch 16 only after an out-of-memory error
python 11_resnet_duyarlilik.py
python 16_lr_ayari.py
python 17_yeni_model.py
python 10_asama2_sonuclar.py
python 15_ek_analizler.py
python 21_cozunurluk_sonuc.py

# Expert audit and audited labels (23 needs the three expert answer files)
python 20_etiket_denetimi_formu.py
python 23_etiket_birlestir.py
python 24_temiz_egitim.py        # E3 backbones with --manifest asama2_temiz/manifest_asama2.csv --out runs_temiz
python 26_temiz_degerlendirme.py

# Selective prediction and conformal sets for E3 (see Ensembles)
python 22_secici_konformal.py --uyeler <E3> --risk <a> --kayit asama2_sonuclar/secici_ek/orij_r<a>.csv --cikti asama2_sonuclar/secici_ek/orij_r<a>.txt
python 22_secici_konformal.py --uyeler <E3> --risk <a> --etiket-dosyasi asama2_sonuclar/orijinal_belirsizsiz_etiketler.csv --kayit asama2_sonuclar/secici_ek/orijbel_r<a>.csv --cikti asama2_sonuclar/secici_ek/orijbel_r<a>.txt
python 22_secici_konformal.py --uyeler <E3> --risk <a> --kok runs_temiz --son kat_v1_asama2_temiz_none --kayit asama2_sonuclar/secici_ek/temiz_r<a>.csv --cikti asama2_sonuclar/secici_ek/temiz_r<a>.txt
#   with <E3> = convnext_tiny_224,swin_tiny_patch4_window7_224_224,vit_base_patch14_dinov2_392 and <a> = 0.02, 0.05, 0.10

# Hand check of the language-model pair decisions
python 12_insan_kontrol_seti.py
python 12b_secimli_form.py
python 13_insan_uyum.py

# External validation and few-shot adaptation
python 25_dis_dogrulama.py                 # Kaggle PV Panel Defect (overlap screening)
python 25_dis_dogrulama.py roboflow        # RF-Fault
python 25_dis_dogrulama.py melbourne       # RF-Soiling
python 27_dis_secici_gurultu.py
python 28_az_ornek_uyarlama.py
python 28b_az_ornek_gruplu.py

# Figures and supplement
python 29_gradcam.py
python 29b_gradcam_v2.py
python 30_makale_sekilleri.py
python 31_makale_ek_sekiller.py
python 37a_uzman_ornekleri.py
python 37b_dis_veri_ornekleri.py
python 37c_etiket_butcesi.py
python 37d_akis_semasi.py
python 33_ek_materyal.py
python 34_grafik_ozet.py
```

The selective-prediction, calibration and model-comparison results can be reproduced from `oof_predictions.csv` alone, without retraining.

### Ensembles

- **E3**, the ensemble reported in the article, averages the class probabilities of ConvNeXt-T at 224 px, Swin-T at 224 px and DINOv2 ViT-B/14 at 392 px over the seeds 42, 43 and 44 (run names `convnext_tiny_224`, `swin_tiny_patch4_window7_224_224` and `vit_base_patch14_dinov2_392`). Its out-of-fold predictions use, for every image and seed, the model of the fold that held the image out. Its external predictions average, for each backbone, the five fold models of the three seeds with TTA, and then the three backbone means. Scripts 24–27, 30 and 31 use these members, and 24, 25 and 27 call them `Topluluk-3 yüksek`.
- The **pre-specified candidate ensemble** of `15_ek_analizler.py` (`Topluluk-3`) uses the same three backbones at 224 px.
- `Topluluk-3 yuksek` in `21_cozunurluk_sonuc.py` uses ConvNeXt-T at **384** px with Swin-T 224 and DINOv2 392. It was a preview of the resolution experiment and is not E3, although the name resembles the one used for E3 in scripts 24, 25 and 27.

### Documented behaviour of the code

These points are documented rather than changed, so that the deposited code remains the code that produced the reported values.

- **Default members of `22_secici_konformal.py`.** The default of `--uyeler` is `convnext_tiny_384,swin_tiny_patch4_window7_224_224,vit_base_patch14_dinov2_392`, the preview ensemble of script 21 and not E3. Run without `--uyeler`, the script reproduces `asama2_sonuclar/secici.txt` (coverage 0.557, violation rate 0.046), a preview that the article does not report. All selective-prediction results in the article were produced with E3 passed explicitly, by the three commands listed above. The label file `orijinal_belirsizsiz_etiketler.csv` assigns the original label to the 2,049 non-ambiguous evaluation images and `belirsiz` to all other files, and with `runs_temiz` the audited labels come from the retrained runs. With α = 0.05 the first and third commands also gave `asama2_sonuclar/secici_E3_orijinal.txt` and `secici_temiz.txt`. Re-running the three commands with the deposited code reproduced all nine deposited split files (`original_alpha*`, `original_ambiguous_removed_alpha*`, `audited_alpha*`) byte for byte.
- **`fleiss()` in `23_etiket_birlestir.py`** takes the number of answers per image from the first image and uses it for all images. Two of the 150 images received only two answers because Expert 1 left them blank (one candidate and one control). On both, the two remaining answers disagree, so their agreement term is zero whatever the denominator, and the function returns exactly the Fleiss' κ for a variable number of answers per image: 0.123 for all 150 images, 0.115 for the 112 candidates and 0.138 for the 38 controls (`revision_analyses_h_kappa_ci.py` computes the variable-number version with bootstrap CIs). For other data, the number of answers should be taken per image (`n = M.sum(1)`).
- **Single-view and TTA predictions.** `02_egit.py` and `02b_egit_asama2.py` write single-view probabilities to `test_predictions.csv` and store the TTA metrics only in `results.json` (`test_tta`). All out-of-fold analyses therefore use single-view predictions (the docstring of `10_asama2_sonuclar.py` calls them TTA predictions, which is inaccurate), whereas the P1 column of Table 3 is taken from `test_tta`. TTA is applied to the external predictions (25) and in the few-shot experiments (28, 28b).

## Note on AI-assisted review

The visual review of all candidate image pairs, recorded in the `review_round` column of `visual_pair_reviews.csv`, was performed by multimodal language-model reviewers: Claude Opus 5.5 (Anthropic) agents run through Claude Code in November 2026. Their prompts are reproduced in `llm_review_prompts.md`. The author checked 115 of their decisions by hand, blind to the reviewers' verdicts (`human_checked` in `visual_pair_reviews.csv`, decisions in `insan_kararlari.csv`). All labels in `expert_audit.csv` were given by human experts.

## Data sources and licences

- **SolarFCD.** M. Ijaz et al., arXiv:2604.23662 (2026). The images are subject to the licence of the original release.
- **Kaggle PV Panel Defect Dataset.** https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
- **RF-Fault.** Roboflow Universe, yolo-fisdp / solar-panel-fault-dataset-nzzte, version 1 (CC BY 4.0).
- **RF-Soiling.** Roboflow Universe, pvsoiling / soiling_dropping-2, version 14 (CC BY 4.0).

The code in this repository is released under the MIT licence (`LICENSE`), and the derived data and result files (CSV and TXT) and `llm_review_prompts.md` are released under CC BY 4.0 (`LICENSE-DATA.md`).

## Version history

- **1.1.0** (25 November 2026), for the manuscript submitted to *Renewable Energy*. Added the P1 predictions (`p1_predictions.csv`), the external label mapping (`external_label_mapping.csv`), the author's hand decisions (`insan_kararlari.csv`) with the columns `human_checked` and `human_check_id` in `visual_pair_reviews.csv`, the environment file, the prompts of the language-model reviewers, the scripts 04, 05, 05b (label-copy baseline), 07, 12, 28b and 30, the additional analyses (`revision_analyses_*`) and the figure scripts 37a–37d. Replaced `33_ek_materyal.py` by its current version. Added to this README the descriptions of all scripts, the external data layout, the column mapping, the command lines and the notes on the ensembles and on the code. Updated the article title.
- **1.0.0–1.0.2** (24 November 2026), for an earlier draft of the manuscript.

## Citation

Please cite the article and this archive (see `CITATION.cff`).

Contact: Enes Halit Aydın, Department of Electricity and Energy, Karadeniz Technical University, Trabzon, Türkiye.
