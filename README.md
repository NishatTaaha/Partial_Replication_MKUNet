# Partial Replication of MK-UNet for Lightweight Polyp Segmentation

A partial replication of **MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation** (Rahman & Marculescu, 2025) on two polyp segmentation benchmarks, **CLINIC-DB** and **COLON-DB**, run under a reduced training budget on a single Tesla T4 (Kaggle).

This repository contains the notebook, per-seed results, training curves and failure analysis for **Deliverable 1** (setup, training, failure analysis and result analysis).

- Paper: [arXiv:2509.18493](https://arxiv.org/abs/2509.18493)
- Authors' code: <https://github.com/SLDGroup/MK-UNet>


## Environment

- Platform: Kaggle notebook
- Python 3.12, PyTorch 2.10 (CUDA 12.8)
- GPU: Tesla T4 (one GPU used for training)
- Extra packages: `medpy`, `SimpleITK`, `nibabel`, `segmentation-mask-overlay`, `ptflops`, `thop`

The training loop was rewritten inside the notebook. The model (`mkunet_network.py`), the dataloader (`utils/dataloader_polyp.py`) and `utils/utils.py` (`clip_gradient`) come from the authors' repository.


## Data

The notebook expects this layout (80:10:10 train/val/test split, as in the paper):

```
data/polyp/target/
├── ClinicDB/
│   ├── train/{images,masks}   # 489
│   ├── val/{images,masks}     # 61
│   └── test/{images,masks}    # 62
└── ColonDB/
    ├── train/{images,masks}   # 303
    ├── val/{images,masks}     # 38
    └── test/{images,masks}    # 38
```

Images are resized to 352×352 and normalised with ImageNet mean and std. Masks are binarised adaptively (`>20` if the mask maximum is above 127, else `>=1`), as in the authors' dataloader.

The datasets are not redistributed here. Obtain them from the original sources (CVC-ClinicDB and CVC-ColonDB) or from the authors' repository instructions.

## How to run

1. Clone the authors' repository and place it (with the data above) where the notebook expects it. The notebook uses `/kaggle/input/.../MK-UNet` as the source and copies it to `/kaggle/working/MK-UNet`. Change `KAGGLE_ROOT` and `WORK_DIR` in the path cell if you run elsewhere.
2. Install the extra dependencies (the notebook does this in its second code cell).
3. Open `better_mk-unet.ipynb` and run the cells in order. Main configuration cells:

   ```python
   DATASETS    = ["ClinicDB", "ColonDB"]
   SEEDS       = [1, 2, 3, 4, 5]
   IMG_SIZE    = 352
   BATCH_SIZE  = 8
   AUGMENT     = False
   EPOCHS      = 100
   LR          = 1e-4
   WEIGHT_DECAY = 1e-4
   ```

4. Outputs are written per dataset and seed to `step1_<Dataset>_MK_UNet/seed<k>/` (`best.pth`, `last.pth`, `history.json`, `result.json`, `console.txt`). Training can resume from `last.pth` (`RESUME = True`).


## Setup and differences from the paper

| Item | This Code | Author's Code |
|---|---|---|
| **Epochs** | `EPOCHS = 100` | `--epoch default=200` |
| **Batch Size** | `BATCH_SIZE = 8` | `--batchsize default=8` |
| **Learning Rate (LR)** | `1e-4` | `--lr default=5e-4` |
| **Optimizer** | `AdamW, weight_decay=1e-4` | `AdamW, weight_decay=1e-4` |
| **Scheduler** | `CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-6)` | `CosineAnnealingLR(T_max=opt.epoch, eta_min=1e-6)` |
| **Loss** | `structure_loss` (weighted BCE + weighted IoU, w=1) | Same function, w=1 |
| **Gradient Clipping** | `clip_gradient(optimizer, 0.5)` | Same (`--clip 0.5`) |
| **Multi-scale Training** | Not used | Used: `size_rates = [0.75, 1, 1.25]` — each batch trained 3× at different scales |
| **Seeds** | `SEEDS = [1,2,3,4,5]` | `for run in [1,2,3,4,5]` |

## Limitations

- The training setup was reduced to fit the available compute: 100 instead of 200 epochs, batch size 8.
- FLOPs were measured at 352×352 (0.619 G MACs), not at the paper’s 256×256 (0.314 G).
- HD95 were not measured.
- Only ClinicDB and ColonDB were run. The paper’s other four datasets were not.

## Acknowledgements and citation

This work builds on the official MK-UNet implementation by the SLDGroup. Please cite the original paper if you use this code:

```bibtex
@article{rahman2025mkunet,
  title   = {MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation},
  author  = {Rahman, Md Mostafijur and Marculescu, Radu},
  journal = {arXiv preprint arXiv:2509.18493},
  year    = {2025}
}
```

## License

The model and dataloader code originate from the authors' repository; refer to its license. Add your own license for the notebook and analysis code here.
