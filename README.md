# SolarFCD audit: leakage, label ambiguity and domain shift in image-based PV fault detection

This repository accompanies the article:

> E.H. Aydın, *Leak-controlled evaluation and expert label auditing reveal the true reliability of image-based photovoltaic fault detection for operation and maintenance*, submitted to *Renewable Energy* (2026).

Archived on Zenodo: [https://doi.org/10.5281/zenodo.24500535](https://doi.org/10.5281/zenodo.24500535)

It provides everything needed to reproduce the analyses **except the images themselves**:

- the duplicate and scene groups for the SolarFCD benchmark;
- the leak-controlled folds;
- the expert labels;
- the overlap screening of the external datasets;
- all model predictions and analysis scripts.

The images must be obtained from their original sources (see [Data sources](#data-sources-and-licences)).

## Repository contents

| Path | Content |
|---|---|
| `data/solarfcd_manifest.csv` | All 3,742 files of the SolarFCD release with MD5 hash, source family, photo group, scene group, label conflicts, the 2,095-image evaluation set, folds and expert-audited labels |
| `data/link_edges.csv` | Edges used to build photo and scene groups (MD5/pHash, geometric matching, DINOv2, EXIF, visual review) |
| `data/visual_pair_reviews.csv` | Visual decisions on candidate image pairs (same photo / same scene / unrelated) used for grouping and for the residual-leakage audit |
| `data/expert_audit.csv` | Blinded labels of three PV experts for 150 images (112 model-flagged candidates, 38 random controls) and the majority consensus |
| `data/external_overlap_*.csv` | Overlap screening of the Kaggle PV Panel Defect, RF-Fault and RF-Soiling datasets against SolarFCD (MD5, pHash, DINOv2 cosine) |
| `results/oof_predictions.csv` | Out-of-fold class probabilities for every model, seed and fold under protocols P2, P3 and P3 with audited labels |
| `results/external_predictions_*.csv` | Class probabilities of the SolarFCD-trained models on the screened external images |
| `results/few_shot_adaptation.csv` | Few-shot adaptation results on RF-Fault |
| `results/selective_splits/*.csv` | Per-split results of the Learn-then-Test selective classifier and LAC conformal sets (1,000 splits; α = 2, 5, 10%) |
| `code/` | Analysis scripts (numbered in execution order; comments are in Turkish) |

## Key columns

**`solarfcd_manifest.csv`**

| Column | Description |
|---|---|
| `rel_path` | Path relative to the root of the SolarFCD-Final archive, e.g. `Train/Healthy/Imgclean_1364_0.jpg` |
| `official_split`, `label` | Split and class of the original release |
| `md5` | MD5 hash of the file bytes |
| `source_family` | Origin family inferred from the file name, e.g. `Imgclean`, `Imgdirty`, `Bird`, `Physical` |
| `strict_group`, `photo_group`, `scene_group` | Byte or pHash group; same-photograph group; same-scene group |
| `label_conflict` | The photo group contains different labels (excluded) |
| `in_evaluation_set` | One of the 2,095 RGB representatives used in the article |
| `fold_scene_grouped` | Fold of protocol P3 (scene-grouped five-fold CV); −1 if not in the evaluation set |
| `fold_random` | Fold of protocol P2 (random five-fold CV) |
| `ambiguous_expert` | The expert majority rule declared the image ambiguous (46 images) |
| `label_audited` | Label after the expert audit; empty if not in the evaluation set or ambiguous |

**`expert_audit.csv`:** `expert1`–`expert3` are the independent, blinded answers. Possible values are the four classes, `mixed` and `not_sure`. `consensus` is the class chosen by at least two experts, otherwise `ambiguous`. `status` is `kept`, `changed` or `ambiguous`.

**`oof_predictions.csv`:** `protocol`, `model`, `img_size`, `seed`, `fold`, `rel_path`, `label`, `prediction` and the class probabilities `p_*`. Each row is the prediction of the model trained on the other four folds.

## Reproducing the analyses

**Environment.** Python 3.12, PyTorch 2.15 (CUDA 13.2), timm 1.0.31, scikit-learn, pandas, OpenCV 5, imagehash and matplotlib. All experiments ran on one NVIDIA RTX 4080 Laptop GPU (12 GB).

**Placeholders in the code.** Replace `<SOLARFCD_DIR>` with the folder of the SolarFCD-Final archive and `<EXTERNAL_DATA_DIR>` with the folder of the external datasets.

The main scripts, in order:

1. `01_veri_denetimi.py`, `06_ham_kaynak_eslesme.py`: audit of the release and of duplicates in the raw sources.
2. `08_asama2_manifest.py`, `08a_gorsel_etiket_derle.py`: photo and scene grouping, conflict removal and folds. Their result is `data/solarfcd_manifest.csv`.
3. `02_egit.py`, `03_pilot.py`: protocol P1 (official split).
4. `02b_egit_asama2.py` with `09_asama2_egitim.py`, `14_rastgele_bolme.py`, `16_lr_ayari.py`, `17_yeni_model.py`, `19_yuksek_cozunurluk.py` and `24_temiz_egitim.py`: training under P3, P2 and audited labels.
5. `10`, `15`, `21`, `26`: model comparison, leakage effect, resolution and audited-label analyses.
6. `20_etiket_denetimi_formu.py`, `23_etiket_birlestir.py`: expert audit form and consensus.
7. `22_secici_konformal.py`, `27_dis_secici_gurultu.py`: Learn-then-Test selective prediction, LAC sets and transfer to external data.
8. `25_dis_dogrulama.py`, `28_az_ornek_uyarlama.py`: external validation and few-shot adaptation.
9. `29_gradcam.py`, `31_makale_ek_sekiller.py`, `33_ek_materyal.py`, `34_grafik_ozet.py`: figures and supplementary tables.

**Without retraining.** The selective-prediction, calibration and model-comparison results can be reproduced from `results/oof_predictions.csv` alone.

## Note on AI-assisted review

Part of the visual review of candidate image pairs, which `visual_pair_reviews.csv` records in its `review_round` column, was performed by multimodal language-model reviewers. The author checked these decisions by hand on a sample. All labels in `expert_audit.csv` were given by human experts.

## Data sources and licences

- **SolarFCD:** M. Ijaz et al., arXiv:2604.23662 (2026). Images are subject to the licence of the original release.
- **Kaggle PV Panel Defect Dataset:** https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
- **RF-Fault:** Roboflow Universe, yolo-fisdp / solar-panel-fault-dataset-nzzte, version 1 (CC BY 4.0).
- **RF-Soiling:** Roboflow Universe, pvsoiling / soiling_dropping-2, version 14 (CC BY 4.0).

Code in this repository is released under the MIT licence (`LICENSE`). The derived data files (`data/`, `results/`) are released under CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

Please cite the article and this archive (see `CITATION.cff`).

Contact: Enes Halit Aydın, Department of Electricity and Energy, Karadeniz Technical University, Trabzon, Türkiye.
