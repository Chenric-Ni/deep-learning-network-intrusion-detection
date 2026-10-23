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

Please check each dataset's license and cite the original papers if you use it.

| Dataset | Official source | Access / license notes |
|---|---|---|
| NSL-KDD | http://nsl.cs.unb.ca/NSL-KDD/ (University of New Brunswick) | Publicly downloadable. Use `KDDTrain+.txt` and `KDDTest+.txt`. Cite Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set" (2009). |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets (UNSW Canberra) | Free for academic research; you must cite the TON_IoT papers listed on that page. The Windows 10 file used here is in the processed Windows datasets. |
| SMD | https://github.com/NetManAIOps/OmniAnomaly (folder `ServerMachineDataset`) | Public. Cite Su et al., KDD 2019 (OmniAnomaly). |
| MSL / SMAP | https://github.com/khundman/telemanom | Public (NASA JPL). Data zip and `labeled_anomalies.csv` are linked in that repo's README. Cite Hundman et al., KDD 2018. |
| PSM | https://github.com/eBay/RANSynCoders/tree/main/data | Public (eBay); the repo is BSD-3 licensed, but check the terms of the sample data. Cite Abdulaal et al., KDD 2021. |
| SWaT | https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat (iTrust, SUTD) | Not freely downloadable: you must request access from iTrust. Do not redistribute. |

Preprocessed copies of SMD, MSL, SMAP and PSM in the Anomaly Transformer format are also linked from the [official Anomaly Transformer repository](https://github.com/thuml/Anomaly-Transformer). The file names expected by `tadgan/dataload.py` for MSL and SMAP differ from that release (see the table above), so check the layout before use.
