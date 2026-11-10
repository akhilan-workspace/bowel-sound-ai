# Phase 3: pretrained audio models, external validation and few-subject adaptation

**Project:** Generalizable AI for Automated Bowel-Sound Detection (MD5001, AI in Medicine). Akhilan S K, MD24B001.
**Run:** Kaggle notebook `notebook98cfb36f44`, version 3 ("Phase 3 GPU"), 1× Tesla T4, total time 243 min, finished with no failed blocks.
**Script:** `phase3.py` (md5 052a84446833b8560cc1bd355ee2345b).

## What was tested

**Datasets**

| Dataset | Role | Contents |
|---|---|---|
| Warsaw/Poznań (Kaggle `robertnowak/bowel-sounds`, Ficek et al. 2021) | Internal | 1606 two-second clips from 427 recording prefixes. 6413 annotated events. 44.1 kHz contact microphone. |
| Oldenburg (figshare 10.6084/m9.figshare.28595741, CC BY 4.0) | External | 7 recordings from 4 subjects, 91 min, 2915 events typed SB/MB/CRS/HS. Different device (a wearable). Never used for training or threshold choice in the external test. |

**Models**

| Model | What it is |
|---|---|
| LightGBM | Hand-made log-mel frame features. |
| CRNN | Small CNN plus BiGRU, trained from scratch. |
| CNN14 | PANNs, pretrained on AudioSet. Frequency-only pooling keeps 10 ms resolution. BiGRU head, fine-tuned. |
| HuBERT-base | Self-supervised speech model, fine-tuned. |

**Internal evaluation protocols**

| Protocol | Splitting | Runs |
|---|---|---|
| Grouped (primary) | 5-fold, split by recording, so it's leak-free | 3 seeds |
| Random | 5-fold over clips, which is leaky | seed 0 |
| Official | The authors' train/test split | seed 0 |

**Metrics**
- Event F1 uses one-to-one matching with any overlap.
- Thresholds are always chosen on an inner validation split of the training data.
- Confidence intervals: 1000 bootstrap resamples over recordings, with paired differences between models.

**External test.** The 5 grouped-fold models are averaged, then applied to Oldenburg zero-shot.
- External precision counts a detection as correct if it overlaps any annotation. Recall is one-to-one.
- Also reported: 100 ms segment F1, false alarms per minute and frame AUROC.

**Leave-one-subject-out (LOSO) adaptation on Oldenburg.** One subject is held out each time. CRNN and CNN14 are trained either on the other 3 Oldenburg subjects only, or on Warsaw plus those 3 subjects. 1 seed.

## Main findings

### 1. Internal: pretraining didn't beat the small CRNN

Event F1 under the leak-free grouped protocol (`internal_pooled_ci.csv`, `paired_comparisons.csv`, Fig. 1):

| Model | Event F1 (mean of 3 seeds) | 95% CI (mean of per-seed bounds) | Frame AUROC |
|---|---|---|---|
| CNN14 | 0.696 | 0.679–0.711 | 0.977 |
| CRNN | 0.695 | 0.679–0.711 | 0.979 |
| LightGBM | 0.663 | 0.646–0.678 | 0.960 |
| HuBERT | 0.658 | 0.642–0.672 | 0.962 |

- **CNN14 vs CRNN:** the difference is −0.001 to +0.003 across seeds, and the CI includes 0 every time, so it's a tie.
- **LightGBM vs CRNN:** −0.026 to −0.039.
- **HuBERT vs CRNN:** −0.031 to −0.045.
- The CIs for those two comparisons exclude 0 in all 3 seeds.

**Leakage inflation** (random split minus grouped split, seed 0) was small and differed by model:

| Model | Inflation | CI |
|---|---|---|
| LightGBM | +0.014 | 0.006–0.023 |
| CNN14 | +0.006 | −0.000 to 0.012 |
| CRNN | +0.003 | not significant |
| HuBERT | +0.003 | not significant |

So on this dataset the random-vs-grouped gap is real but modest. It shouldn't be presented as large.

### 2. External: pretrained models hold recall on a new device, at a cost in false alarms

See `external.csv` and Fig. 2. Values are the mean over 7 recordings, then the mean ± SD over 3 seeds.

| Model | Event F1 | Recall | Precision | Segment F1 | False alarms/min | Frame AUROC |
|---|---|---|---|---|---|---|
| CNN14 | 0.755 ± 0.012 | 0.870 | 0.686 | 0.681 | 42 | 0.820 |
| HuBERT | 0.734 ± 0.004 | 0.886 | 0.650 | 0.665 | 47 | 0.812 |
| CRNN | 0.677 ± 0.016 | 0.639 | 0.815 | 0.524 | 24 | 0.803 |
| LightGBM | 0.552 ± 0.020 | 0.508 | 0.767 | 0.418 | 31 | 0.792 |

How to read this honestly:

1. **Two recordings are dominated by long sounds.** In `080524002_2` and `150524003_1`, 55–70% of annotated events are multi-burst, rumbling or harmonic sounds (MB/CRS/HS), and the models flag 27–38% of the time. On those, every model, LightGBM included, scores about 0.79 event F1 with 50–130 false alarms per minute. The event metric doesn't separate models there.
2. **The model differences come from the 5 sparse recordings.**
   - Event F1: CNN14 0.743, HuBERT 0.713, CRNN 0.628, LightGBM 0.455.
   - The gain comes mainly from recall. On sparse recordings, CNN14 recall averages 0.84 against 0.52 for CRNN (Fig. 2).
   - The price is false alarms per minute: 26.7 for CNN14 and 35.4 for HuBERT, against 4.1 for CRNN.
3. **The threshold-free gap is small.**
   - CNN14 vs CRNN per recording: frame AUROC +0.016 on average (CNN14 higher in 5 of 7 recordings). Segment F1: CNN14 higher in 6 of 7.
   - So much of the external advantage looks like a better operating point carried over from the Warsaw threshold, more than a much better representation.
4. **Recall by Oldenburg event type** (all seeds):
   - CNN14: SB 0.85, MB 0.96, CRS 0.84, HS 0.82.
   - CRNN: SB 0.61, MB 0.82, CRS 0.47, HS 0.57.

### 3. Adaptation: target data improves ranking, but event F1 depends on the threshold

See `loso.csv` and Fig. 3. This is 1 seed and only 4 subjects, so treat it as preliminary.

| | Frame AUROC, CRNN | Frame AUROC, CNN14 | Event F1, CRNN | Event F1, CNN14 |
|---|---|---|---|---|
| Zero-shot (Warsaw only, seed 0) | 0.797 | 0.822 | 0.684 | 0.763 |
| 3 Oldenburg subjects only | 0.860 | 0.858 | 0.546 | 0.529 |
| Warsaw + 3 Oldenburg subjects | 0.864 | 0.857 | 0.652 | 0.639 |

- Adding target-device subjects raises AUROC clearly.
- Event F1 with the tuned threshold falls below zero-shot.
- The chosen thresholds varied from 0.325 to 0.95 across LOSO folds, which suggests unstable operating-point selection with few subjects.
- Adding Warsaw to Oldenburg-only training improved event F1 in 5 of 7 held-out recordings for both models (mean +0.11).

### 4. Error analysis (Warsaw, grouped)

See `recall_by_duration.csv` and Fig. 4.
- Recall for events shorter than 15 ms is only 0.46–0.54, against 0.74–0.87 for events over 80 ms. Short events are the main failure mode for every model.
- HuBERT is best on the shortest events (0.54) and weakest on mid-length ones.

## Limitations

- **Small external set.** Only 4 external subjects, 7 recordings.
- **Different labelling conventions.** Oldenburg annotates long multi-burst or rumbling events; Warsaw annotates short frames. External numbers are about agreement with another group's labels, not absolute accuracy.
- **Overlap-based precision.** It rewards fragmented detections inside long annotations. Segment F1, false alarms per minute and AUROC are reported for that reason.
- **LOSO thinness.** One seed. No CIs, because there are too few subjects for a meaningful bootstrap.
- **Untuned pretrained models.** Fixed epochs (CNN14 25, HuBERT 15, CRNN 40) and no hyperparameter search for any model. HuBERT in particular might improve with tuning.
- **Grouping unit.** Warsaw grouping is by recording prefix, because the dataset doesn't expose subject IDs.

## A defensible claim

Under leak-free evaluation, AudioSet pretraining gives no in-domain gain over a small CRNN for bowel-sound event detection, and speech self-supervised pretraining is worse. Under a device and annotation shift, the AudioSet model keeps much higher recall at the source-tuned threshold, but with more false alarms and only a small threshold-free gain. Few-subject target data improves discrimination, yet the event-level benefit is limited by unstable threshold selection. Per-device threshold calibration is therefore the key open step.

## Sensible next experiments (not yet run)

1. Calibrate the threshold on a short labelled slice of each target recording (for example the first 2 minutes), then test on the rest. This directly tests the threshold hypothesis.
2. Run LOSO with 3 seeds to get variance estimates.
3. Add another external dataset if the contacted groups share data.

## Files

| File | Contents |
|---|---|
| `internal_pooled_ci.csv` | Pooled out-of-fold metrics with bootstrap CIs, per model, protocol and seed. |
| `internal_folds.csv` | Per-fold metrics, thresholds, sample sizes and fit times. |
| `paired_comparisons.csv` | Paired bootstrap model differences and leakage inflation. |
| `external.csv` | Per-recording zero-shot Oldenburg metrics, per model and seed. |
| `loso.csv` | Per-recording leave-one-subject-out metrics. |
| `recall_by_duration.csv`, `recall_by_category.csv` | Warsaw recall by event duration bin and by annotation category code as given in the dataset. Counts are summed over 3 seeds. |
| Figs. 1–4 | `fig1_internal_event_f1.png`, `fig2_external.png`, `fig3_loso_adaptation.png`, `fig4_recall_by_duration.png`. In this folder. |

Numbers are rounded to 4 decimals. `phase3.py`, `log.txt`, `SUMMARY.txt` and the raw out-of-fold prediction files are kept in the Kaggle output of version 3.
