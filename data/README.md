# Data

Images and annotations are not versioned in this repository (see
`.gitignore`). Download them from
**[Zenodo](https://doi.org/10.5281/zenodo.XXXXXXX)** and place them as follows.
The partition manifests in `splits/` are versioned here.

```
data/
├── alternaria_alternata/
│   └── images/                 146 bright-field micrographs, one LabelMe .json per image
├── lasiodiplodia/
│   └── images/
├── colletotrichum_siamense/
│   └── images/
├── neopestalotiopsis/
│   └── images/
├── hortaea_werneckii/
│   └── images/
└── splits/
    ├── alternaria_alternata.csv
    └── fewshot/
        ├── lasiodiplodia/              5shot_s42.csv, 5shot_s7.csv, ..., 10shot_s2024.csv
        ├── colletotrichum_siamense/
        ├── neopestalotiopsis/
        └── hortaea_werneckii/
```

Images may be organized in subfolders: they are searched recursively, and
every image is identified by its file name, which must therefore be unique
within a collection.

## Collections

| Folder | Taxon | Images used | Annotated conidia | Source and license |
|---|---|---|---|---|
| `alternaria_alternata` | *Alternaria alternata* ATCC 20084 | 146 | 24 686 | This study, CC BY-NC 4.0 |
| `lasiodiplodia` | *Lasiodiplodia* spp. (four species) | 170 | 518 | Images: Crespo-Michel et al. (2023), redistribution subject to the rights holders' permission. Annotations: this study, CC BY-NC 4.0 |
| `colletotrichum_siamense` | *Colletotrichum siamense* | 59 | 912 | Images: TgFC, Islam et al. (2025), CC BY 4.0. Annotations: this study, CC BY-NC 4.0 |
| `neopestalotiopsis` | *Neopestalotiopsis* sp. | 83 | 374 | Images: TgFC, Islam et al. (2025), CC BY 4.0. Annotations: this study, CC BY-NC 4.0 |
| `hortaea_werneckii` | *Hortaea werneckii* | 87 | 11 135 | This study, CC BY-NC 4.0 |

Counts refer to the images used in the experiments (for the target domains,
the test set plus the distinct images drawn into any support set).

## Annotation format

Each image `<name>.<ext>` (PNG, JPEG, TIFF or BMP) has a LabelMe JSON file
`<name>.json` next to it, with one closed polygon per conidium in pixel
coordinates of the original image:

```json
{
  "shapes": [
    {"label": "conidium", "points": [[x1, y1], [x2, y2], [x3, y3], ...], "shape_type": "polygon"}
  ],
  "imagePath": "<name>.<ext>", "imageHeight": 2048, "imageWidth": 2448
}
```

Only `shapes[*].points` is read; polygons with fewer than three vertices are
ignored.

## Partition manifests

There is one manifest for the source domain and one sub-folder per target
taxon for the few-shot manifests, with one file per support size and seed
(2 sizes x 4 seeds = 8 files per taxon, 32 in total). Manifests from the
original implementation (e.g. `split_5folds.csv`,
`split_fewshot_5_colletotrichum_siamense_s42.csv`) can be copied into this
layout, with English column names, by `scripts/prepare_splits.py`.

**Source domain** (`splits/alternaria_alternata.csv`): columns `image,fold`.
`fold` is `0`, `1`, `2` or `3` for the four cross-validation folds of the 116
training images and `test` for the 30 held-out test images.

```csv
image,fold
IMG_0001.jpg,0
IMG_0002.jpg,test
```

**Few-shot** (`splits/fewshot/<species>/<N>shot_s<seed>.csv`): columns
`image,species,subset`. `subset` is `train` for the N support images, `test`
for the fixed test set and `unused` for pool images not drawn for this
support set. Support sets of N = 5 and 10 images were drawn with seeds 42, 7,
123 and 2024; for *H. werneckii*, the five-image set is nested within the
ten-image set of the same seed.

```csv
image,species,subset
HW_017.jpg,hortaea_werneckii,train
HW_031.jpg,hortaea_werneckii,test
HW_044.jpg,hortaea_werneckii,unused
```

Manifests written by the original implementation (`imagen`, `especie`,
`conjunto`, `no_usado`) are read transparently.
