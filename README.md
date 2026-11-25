# Two-Teacher Distillation for Plant-Disease Classification

**BioCLIP-2 + DINOv2 → MobileNetV3**

Two frozen Vision-Transformer teachers (BioCLIP-2 ViT-L and DINOv2-B) see the *clean* image once. Their predictions and DINOv2 attention maps are cached. A small CNN student (MobileNetV3, about 5 M parameters) is trained on the same leaf pasted onto a *random new background*, so it cannot rely on the background. Only the student is deployed.

| | Original paper | This project |
|---|---|---|
| Teacher | one frozen ViT | two frozen ViTs, combined per image |
| Student | ConvNeXt / ViT-S (21-50 M) | MobileNetV3 (about 5 M) |
| Input | same image for teacher and student | teacher: clean, student: background-swapped |
| Loss | CE + KD | CE + KD + attention + consistency |

## Datasets

- **PlantVillage** (38 classes): training and in-distribution test (20,000 images used, 80/20 split). Colour and segmented images are paired to extract leaf masks.
- **Describable Textures Dataset (DTD)**: random backgrounds (2,000 train / 500 test).
- **PlantDoc** (field images): test only, on the 2,578 images in classes shared with PlantVillage.

## Repository contents

| File | Description |
|---|---|
| `deep_learning_project.ipynb` | Full pipeline: data prep, teacher feature caching, student training, ablation, results table |
| `plantkd_lib.py` | Library code (also written by the notebook via `%%writefile`) |
| `requirements.txt` | Python dependencies |
| `results/` | `results_table.csv`, `all_runs.csv` |

## How to run

1. Open the notebook in Google Colab (GPU runtime, e.g. L4).
2. Run the sections in order: GPU/paths, install, library, download datasets, preprocess, teachers, student runs, results.
3. The datasets and the cache (about 7 GB) are **not** stored in this repo. They are downloaded and built by the notebook. The cache can optionally be shared through Google Drive (section 8).

```bash
pip install -r requirements.txt
```

## Ablation

9 configurations × 5 seeds = 45 student runs, 12 epochs each:

`ce_clean`, `ce_swap`, `orig_style`, `kd_bioclip`, `kd_dino`, `kd_both`, `kd_both_att`, `kd_both_con`, `full`.

## Results (mean ± std over 5 seeds, %)

| Config | PV clean | PV swapped | PlantDoc |
|---|---|---|---|
| ce_clean | 99.36 ± 0.09 | 59.85 ± 3.93 | 18.69 ± 2.29 |
| ce_swap | 93.30 ± 0.79 | 98.78 ± 0.10 | 25.54 ± 1.42 |
| orig_style | 98.82 ± 0.09 | 67.29 ± 0.56 | 24.47 ± 0.72 |
| kd_bioclip | 94.71 ± 0.44 | 98.36 ± 0.07 | 31.65 ± 1.05 |
| kd_dino | 94.84 ± 0.35 | 98.27 ± 0.09 | 31.21 ± 0.91 |
| kd_both | 95.07 ± 0.48 | 98.44 ± 0.03 | 31.81 ± 1.39 |
| kd_both_att | 95.20 ± 0.38 | 98.54 ± 0.14 | 31.82 ± 0.60 |
| kd_both_con | 95.59 ± 0.43 | 98.58 ± 0.12 | 31.31 ± 1.00 |
| full | 95.45 ± 0.34 | 98.61 ± 0.10 | 31.11 ± 1.22 |

**Teacher reference (linear heads):**

| Teacher | PV test | PlantDoc |
|---|---|---|
| BioCLIP-2 | 97.70 | 56.32 |
| DINOv2 | 98.87 | 46.08 |
| Both | 99.05 | 53.65 |

## Key findings

- Training on background-swapped leaves removes background shortcuts: `ce_clean` falls to 59.9% on swapped images, while swapped training reaches about 98.5%.
- Distillation improves field accuracy on PlantDoc by about 6 points over a plain swapped-background student (25.5% to about 31.8%).
- The attention and consistency losses give small gains on PlantVillage but no measurable difference on PlantDoc (within seed noise).
- PlantDoc accuracy is limited by the teachers themselves (46-56%) and by the lab-to-field domain shift.

## Limitations and future work

- Teachers are linear probes trained on PlantVillage only, which caps field accuracy.
- Stronger augmentation (scale, lighting, cluttered natural backgrounds such as Places365), longer training, a pretrained student initialization and the full PlantVillage set are expected to help.
- The PlantDoc class mapping is rule-based and should be checked.

## References

- Moummad et al., knowledge distillation for plant recognition, [arXiv:2604.27178](https://arxiv.org/abs/2604.27178)
- PlantVillage: [spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- PlantDoc: [pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- DTD: Cimpoi et al., Describable Textures Dataset
