# igm-ITMIXII-paper

This repository contains the Python scripts used in the paper **XXX**.  
They modify and extend the **IGM 2.2.1** modeling framework for the ITMIX-II experiments.

---

## 📦 Requirements

- **IGM 2.2.1**, commit `030133c` (July 5, 2024)  
- Python ≥ 3.9  
- Dependencies required by IGM (see its wiki)

---

## 🛠 1. Install IGM

The analysis in the paper uses **IGM 2.2.1**, at commit: 030133c

See the commit list here:  
https://github.com/instructed-glacier-model/igm/commits/main/

Following IGM installation's instruction (https://github.com/instructed-glacier-model/igm/wiki/1.-Installation) :
Clone and checkout the correct version:

```bash
git checkout 030133c
```


### 🔧 2. Merge modified scripts with main IGM
This repository provides custom preprocessing scripts for the ITMIX-II experiments.

Copy:

  load_ncdf_ITMIXpaper.py; optimize_ITMIXpaper_tasman.py; optimize_ITMIXpaper_tasman.py

into your local IGM installation:
```bash
igm/modules/preproc/
```


### ▶️ 3. Running the workflow

After obtaining the ITMIX-II datasets (from XXXX.com), create a working directory with a structure similar to:

```bash
your_project/
├─ data/
│ └─ Data_thkinit.nc
├─ params.json
```
with params.json being choosen amongst params_IC.json, params_gla.json and params_gla_tasman.json

