# GuessThatPrice

A financial analytics tool that uses **linear regression** (built from scratch with gradient descent) to model and forecast prices. Designed to help analysts and traders **mitigate financial risk** and gain a sharper read on commodity market trends.

---

## Purpose

Commodity prices — especially natural gas — are notoriously volatile. GuessThatPrice provides a data-driven baseline for:

- **Risk mitigation** — Quantify price uncertainty with confidence intervals (±1 RMSE band) so you never go into a position blind.
- **Market trend analysis** — Visualize the long-run price trajectory to separate signal from noise.
- **Price forecasting** — Enter any date and receive an estimated price along with a likely trading range, with an automatic warning when forecasting beyond the training window (extrapolation risk).

---

## How It Works

| Step | Description |
|------|-------------|
| **Data ingestion** | Reads historical Natural Gas prices from `Nat_Gas.csv` or any valid CSV file |
| **Feature engineering** | Converts calendar dates to numeric (Unix-day) format and applies Z-score scaling |
| **Model training** | Minimises Mean Squared Error via batch **gradient descent** (no ML library black-boxes) |
| **Evaluation** | Reports R², MAE, and RMSE to gauge fit quality |
| **Forecasting** | Inversely maps a user-supplied date through the scaler and outputs a point estimate + ±1 RMSE interval |

---

## Output

The model produces three diagnostic charts and a console summary:

| Output | What It Shows |
|--------|---------------|
| **Figure 1 — Price Trend** | Historical prices, the fitted regression line, and the ±1 RMSE confidence band |
| **Figure 2 — Residual Analysis** | Residuals over time + distribution histogram (checks for model bias) |
| **Figure 3 — Cost Convergence** | Gradient descent cost curve (confirms the model converged properly) |
| **Console Summary** | Dataset stats, R², MAE, RMSE, iteration count, and the forecasted price range |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
python linear_regression_model.py
```

When prompted, enter a date in `mm/dd/yy` format to receive a price estimate and confidence range.

---

## Project Structure

```
GuessThatPrice/
├── linear_regression_model.py   # Core model — data loading, training, charts, forecasting
├── Nat_Gas.csv                  # Historical natural gas price dataset
├── requirements.txt             # Python dependencies
└── Demo Photos/                 # Sample output screenshots
```

---

## Built With

- **Python** — Core language
- **NumPy** — Gradient descent & linear algebra
- **Pandas** — Data ingestion and date handling
- **scikit-learn** — Feature scaling and evaluation metrics
- **Matplotlib** — Charting and visualisation
