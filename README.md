# SolarFCD audit: leakage, label ambiguity and domain shift in image-based PV fault detection

This repository accompanies the article:

> E.H. Aydın, *Leak-controlled evaluation and expert label auditing reveal the true reliability of image-based photovoltaic fault detection for operation and maintenance*, submitted to *Renewable Energy* (2026).

Archived on Zenodo: [https://doi.org/10.5281/zenodo.24500535](https://doi.org/10.5281/zenodo.24500535)

It provides everything needed to reproduce the analyses **except the images themselves**:

- the duplicate and scene groups for the SolarFCD benchmark;
- the leak-controlled folds;
- the expert labels;
- the overlap screening of the external datasets;
- all model predictions and the analysis scripts.

The images must be obtained from their original sources (see [Data sources and licences](#data-sources-and-licences)). All files are stored in the repository root.

## Data files

| File | Content |
|---|---|
| `solarfcd_manifest.csv` | All 3,742 files of the SolarFCD release with MD5 hash, source family, photo group, scene group, label conflicts, the 2,095-image evaluation set, folds and expert-audited labels |
| `link_edges.csv` | Edges used to build photo and scene groups (MD5/pHash, geometric matching, DINOv2, EXIF, visual review) |
| `visual_pair_reviews.csv` | Visual decisions on candidate image pairs (same photo, same scene or unrelated) used for grouping and for the residual-leakage audit |
| `expert_audit.csv` | Blinded labels of three PV experts for 150 images (112 model-flagged candidates and 38 random controls) and the majority consensus |
| `external_overlap_kaggle_pv_panel_defect.csv`, `external_overlap_rf_fault.csv`, `external_overlap_rf_soiling.csv` | Overlap screening of the external datasets against SolarFCD (MD5, pHash, DINOv2 cosine) |

## Result files

| File | Content |
|---|---|
| `oof_predictions.csv` | Out-of-fold class probabilities for every model, seed and fold under protocols P2, P3 and P3 with audited labels |
| `external_predictions_rf_fault.csv`, `external_predictions_rf_soiling.csv` | Class probabilities of the SolarFCD-trained models on the screened external images |
| `few_shot_adaptation.csv` | Few-shot adaptation results on RF-Fault |
| `original_alpha*.csv`, `original_ambiguous_removed_alpha*.csv`, `audited_alpha*.csv` | Per-split results of the Learn-then-Test selective classifier and LAC conformal sets (1,000 splits each, α = 2, 5 and 10%) under the three label conditions |

## Code files

The numbered Python scripts (`01_…py` to `34_…py` and `ortak.py`) are the analysis scripts in execution order. Their comments are in Turkish.

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

**`oof_predictions.csv`.** The columns are `protocol`, `model`, `img_size`, `seed`, `fold`, `rel_path`, `label`, `prediction` and the class probabilities `p_*`. Each row is the prediction of the model trained on the other four folds.

## Reproducing the analyses

The analyses used Python 3.12, PyTorch 2.15 (CUDA 13.2), timm 1.0.31, scikit-learn, pandas, OpenCV 5, imagehash and matplotlib on one NVIDIA RTX 4080 Laptop GPU (12 GB). In the code, `<SOLARFCD_DIR>` must be replaced with the folder of the SolarFCD-Final archive and `<EXTERNAL_DATA_DIR>` with the folder of the external datasets. The scripts originally ran in a working folder in which their inputs and outputs were stored in subfolders, as referenced in the code.

The main scripts are, in order:

1. `01_veri_denetimi.py` and `06_ham_kaynak_eslesme.py` for the audit of the release and of duplicates in the raw sources;
2. `08_asama2_manifest.py` and `08a_gorsel_etiket_derle.py` for photo and scene grouping, conflict removal and folds, which produce `solarfcd_manifest.csv`;
3. `02_egit.py` and `03_pilot.py` for protocol P1 (official split);
4. `02b_egit_asama2.py` with `09_asama2_egitim.py`, `14_rastgele_bolme.py`, `16_lr_ayari.py`, `17_yeni_model.py`, `19_yuksek_cozunurluk.py` and `24_temiz_egitim.py` for training under P3, P2 and audited labels;
5. `10_asama2_sonuclar.py`, `15_ek_analizler.py`, `21_cozunurluk_sonuc.py` and `26_temiz_degerlendirme.py` for the model comparison, leakage effect, resolution and audited-label analyses;
6. `20_etiket_denetimi_formu.py` and `23_etiket_birlestir.py` for the expert audit form and consensus;
7. `22_secici_konformal.py` and `27_dis_secici_gurultu.py` for Learn-then-Test selective prediction, LAC sets and transfer to external data;
8. `25_dis_dogrulama.py` and `28_az_ornek_uyarlama.py` for external validation and few-shot adaptation;
9. `29_gradcam.py`, `31_makale_ek_sekiller.py`, `33_ek_materyal.py` and `34_grafik_ozet.py` for the figures and supplementary tables.

The selective-prediction, calibration and model-comparison results can be reproduced from `oof_predictions.csv` alone, without retraining.

## Note on AI-assisted review

Part of the visual review of candidate image pairs, recorded in the `review_round` column of `visual_pair_reviews.csv`, was performed by multimodal language-model reviewers, and the author checked these decisions by hand on a sample. All labels in `expert_audit.csv` were given by human experts.

## Data sources and licences

- **SolarFCD.** M. Ijaz et al., arXiv:2604.23662 (2026). The images are subject to the licence of the original release.
- **Kaggle PV Panel Defect Dataset.** https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
- **RF-Fault.** Roboflow Universe, yolo-fisdp / solar-panel-fault-dataset-nzzte, version 1 (CC BY 4.0).
- **RF-Soiling.** Roboflow Universe, pvsoiling / soiling_dropping-2, version 14 (CC BY 4.0).

The code in this repository is released under the MIT licence (`LICENSE`), and the derived CSV data files are released under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

Please cite the article and this archive (see `CITATION.cff`).

Contact: Enes Halit Aydın, Department of Electricity and Energy, Karadeniz Technical University, Trabzon, Türkiye.
