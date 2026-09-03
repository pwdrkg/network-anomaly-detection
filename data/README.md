# Data

The raw dataset is **not committed** to keep the repository lightweight. Download
the UNSW-NB15 pre-partitioned CSVs and place them in this folder.

## UNSW-NB15 (train/test partition)

- **Source:** Australian Centre for Cyber Security (ACCS), 2015
- **Kaggle:** https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- **Official:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place these two files here:

```
data/
├── UNSW_NB15_training-set.csv   # 175,341 records (official training split)
└── UNSW_NB15_testing-set.csv    #  82,332 records (official testing split)
```

> **Note:** some mirrors swap the train/test file names. The **larger** file
> (175,341 rows) is the official *training* split. `src/features.py` expects the
> files named as above.

## Generated files (also git-ignored)

Running `python src/features.py` produces cached, model-ready splits:

```
data/train_processed.parquet
data/test_processed.parquet
```
