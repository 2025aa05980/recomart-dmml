# Model Performance Report — RecoMart Pipeline
**Task 9 | DMML Assignment 1 | Group 37**
Generated: 2026-07-07 06:32 UTC

---

## Executive Summary

| Model | Type | RMSE | Precision@10 | Recall@10 | NDCG@10 |
|-------|------|------|-------------|-----------|---------|
| SVD | Collaborative Filtering | 0.2865 | 1.0 | 0.0013 | 1.0 |
| TF-IDF Cosine | Content-Based | — | — | — | — |

---

## Model 1: SVD Collaborative Filtering

### Architecture
SVD (Singular Value Decomposition) factorizes the user-item
rating matrix R into latent factor matrices:
`R ≈ U × Σ × V^T`
where U = user factors, V = item factors, Σ = singular values.

### Hyperparameter Tuning (GridSearchCV, 3-fold CV)

| Parameter | Search Space | Best Value |
|-----------|-------------|------------|
| n_factors | [20, 50, 100] | 20 |
| n_epochs  | [15, 20] | 15 |
| lr_all    | [0.005, 0.01] | 0.01 |
| reg_all   | [0.02, 0.1] | 0.1 |

### Evaluation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| RMSE | 0.2865 | Root Mean Squared Error (lower = better) |
| MAE | 0.2185 | Mean Absolute Error (lower = better) |
| Precision@10 | 1.0 | Fraction of top-10 that are relevant |
| Recall@10 | 0.0013 | Fraction of relevant items in top-10 |
| NDCG@10 | 1.0 | Ranking quality (higher = better) |
| CV RMSE | 0.9988 | Cross-validated RMSE |

### MLflow Run ID: `757e9377fb944f3da091de7c3a8e3f55`
Experiment: `recomart_videogames_svd`

---

## Model 2: Content-Based Filtering (TF-IDF + Cosine Similarity)

### Architecture
Builds item profiles from text features (title + category + brand)
using TF-IDF vectorization. Item similarity computed via cosine
similarity. Recommends items similar to those a user has liked.

### Configuration

| Parameter | Value |
|-----------|-------|
| Vectorizer | TF-IDF |
| Max features | 5,000 |
| N-gram range | (1, 2) |
| Similarity | Cosine |
| Products indexed | 45,914 |
| Vocabulary size | 5,000 |

### Evaluation

| Metric | Value | Description |
|--------|-------|-------------|
| Coverage | 100.0% | % of rated items recommendable |
| Avg top-10 similarity | 0.5963 | Item neighborhood cohesion |

### MLflow Run ID: `b50c5f3054c7434c84b6898839379591`

---

## Sample Recommendations (SVD)

Top-10 predicted recommendations for 3 sample users:

### User: `AHJRJCJMK3XVV4BSPBRA...`

| Rank | Product ID | Predicted Rating | Title | Category |
|------|-----------|-----------------|-------|----------|
| 1 | B004HD55VK | 3.17 | Tomb Raider | Video Games > Legacy Systems > PlayStati |
| 2 | B0BWTD299J | 3.152 | KKCOBVR IR Battery Elite Head Strap with 10000mAh  | Video Games > PC > Virtual Reality > Hea |
| 3 | B0025ZRHTA | 3.148 | PlayStation Portable Limited Edition Rock Band Unp | Video Games > Legacy Systems > PlayStati |
| 4 | B001ELJE1K | 3.138 | Grand Theft Auto: The Trilogy (Grand Theft Auto II | Video Games > Legacy Systems > PlayStati |
| 5 | B0C86HSZX9 | 3.114 | TESSGO PS 5 Disc Edition Matte Black Face Plate Co | Video Games > PlayStation 5 > Accessorie |
| 6 | B09FGM3DVQ | 3.108 | Mcbazel Dust Cover for Xbox Series X Console, Easy | Video Games > Legacy Systems > Xbox Syst |
| 7 | B000035Y2Q | 3.102 | Secret of Mana | Video Games > Legacy Systems > Nintendo  |
| 8 | B002BSA388 | 3.093 | Super Mario Galaxy 2 | Video Games > Legacy Systems > Nintendo  |
| 9 | B00S7KMY3Q | 3.062 | Forza Motorsport 6 (Xbox One) | Video Games > Xbox One > Games |
| 10 | B0000ALBWU | 3.057 | Metal Gear Solid 3: Snake Eater | Video Games > Legacy Systems > PlayStati |

### User: `AGMWACNMAG74AXBF7IJ2...`

| Rank | Product ID | Predicted Rating | Title | Category |
|------|-----------|-----------------|-------|----------|
| 1 | B000B6MLTG | 3.608 | Xbox 360 VGA HD AV Cable | Video Games > Legacy Systems > Xbox Syst |
| 2 | B000ARJIBU | 3.376 | Serious Sam 2 - Xbox | Video Games > Legacy Systems > Xbox Syst |
| 3 | B009GE437W | 3.351 | Remote Plus, Mario - Nintendo Wii | Video Games > Legacy Systems > Nintendo  |
| 4 | B07P3WBC6R | 3.349 | Nintendo amiibo - King K. Rool - Super Smash Bros. | Video Games > Nintendo Switch > Accessor |
| 5 | B00009YFU2 | 3.317 | Amped 2 - Xbox | Video Games > Legacy Systems > Xbox Syst |
| 6 | B01JYYWL7C | 3.29 | Exlene Nintendo GBA/SP/DS USB Power Charger Cable  | Video Games > Legacy Systems > Nintendo  |
| 7 | B000MUYV4O | 3.286 | Nintendo DS Lite Armor Lite Case | Video Games > Legacy Systems > Nintendo  |
| 8 | B00JTWUMJY | 3.253 | PlayStation Thumb Grips (for PS4 and PS3 controlle | Video Games > Legacy Systems > Nintendo  |
| 9 | B0BPS14XFS | 3.227 | AKNES Gulikit Switch Joycon Replacement-No Driftin | Video Games > Nintendo Switch > Accessor |
| 10 | B004XABXY0 | 3.22 | PlayStation 3 160GB Call of Duty: Black Ops Bundle | Video Games > Legacy Systems > PlayStati |

### User: `AGIBXD3LM6HNDWWRTIOJ...`

| Rank | Product ID | Predicted Rating | Title | Category |
|------|-----------|-----------------|-------|----------|
| 1 | B00MB53DQ0 | 5 | Aweek Bracket Handgrip Handle Grip Case for Playst | Video Games > Legacy Systems > PlayStati |
| 2 | B01F2AW69A | 5 | Gametown Repair Replacement Button Thumbstick Anal | Video Games > Legacy Systems > PlayStati |
| 3 | B09LC2RDKG | 5 | WFB Wireless Gaming Mouse,Silence Click,Rechargeab | Video Games > PC > Accessories > Gaming  |
| 4 | B01MQSH5IE | 5 | CYBER /push guard (For New 3DS XL) Black | Unknown |
| 5 | B0000AW9RE | 5 | Saitek ST290 Programmable Joystick with Throttle | Video Games > PC > Accessories > Control |
| 6 | B07NTP3MPW | 5 | MOYEEL 2 Pairs Replacement Metal Lock Latches for  | Video Games > Nintendo Switch > Accessor |
| 7 | B07K545RJY | 5 | Atelier Lulua: The Scion of Arland - Nintendo Swit | Video Games > Nintendo Switch > Games |
| 8 | B000132GC6 | 5 | Sony PlayStation 2 Combo Pack | Video Games > Legacy Systems > PlayStati |
| 9 | B0BR4PNZTS | 5 | i Maifu Ray Pre Lubed Switches - 35x PCB Mounted 5 | Video Games > PC > Accessories > Gaming  |
| 10 | B01AC0I84W | 5 | Kapp'n amiibo - Nintendo Wii U | Video Games > Legacy Systems > Nintendo  |

---

## Model Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| SVD model | `models/svd_model.pkl` | Trained SVD model (pickle) |
| CBF similarity | `models/cbf_similarity.npz` | Sparse cosine similarity matrix |
| Product index | `models/cbf_product_index.json` | productId → matrix index |
| Model metadata | `models/model_metadata.json` | Parameters, metrics, run IDs |

## MLflow Experiment
Experiment name: `recomart_videogames_svd`
View UI: `mlflow ui` then open http://localhost:5000