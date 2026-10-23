# Data

No datasets are included in this repository, either because of size or because their licenses may not allow redistribution. Download each dataset from its official source and place it under `data/` using the layout below. Everything in `data/` except this file is ignored by Git.

The layouts below match what `tadgan/dataload.py` reads.

## NSL-KDD (`--dataset KDD`)

```
data/NSL-KDD/
├── KDDtrain+.txt
└── KDDtest+.txt
```

## TON_IoT Windows 10 (`--dataset TON`)

Download the Windows 10 train/test file (`Train_Test_Windows_10.csv`) from the TON_IoT collection, then generate the split files with the preprocessing script in `tadgan/preprocess/preprocess_ton_swat.py`:

```
data/TON_IoT_windows10/
├── train_data.csv
├── train_labels.csv
├── test_data.csv
└── test_labels.csv
```

## SMD (`--dataset SMD`)

```
data/SMD/
├── SMD_train.pkl
├── SMD_test.pkl
└── SMD_test_label.pkl
```

## Other datasets supported by the loaders

| Dataset | `--dataset` | Expected files |
|---|---|---|
| MSL | `MSL` | `T-9_train.npy`, `T-9_test.npy`, `T-9_labels.npy` |
| SMAP | `SMAP` | `A-6_train.npy`, `A-6_test.npy`, `A-6_labels.npy` |
| PSM | `PSM` | `train.csv`, `test.csv`, `test_label.csv` |
| SWaT | `SWAT` | `train.csv`, `test.csv`, `test_labels.csv` (produced by `tadgan/preprocess/preprocess_ton_swat.py`; SWaT requires applying to the dataset owners) |

## Download sources

<!-- TODO: add official links for each dataset you used (NSL-KDD, TON_IoT, SMD, MSL/SMAP, PSM, SWaT) and note each dataset's license. -->

- Add the official page for each dataset here.
- Cite the original dataset papers in your thesis summary.
