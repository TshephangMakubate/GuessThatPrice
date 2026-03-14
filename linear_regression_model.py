"""
Natural Gas Price Prediction — Linear Regression (from scratch)
===============================================================
Model: Single-feature linear regression using gradient descent.
Data : Nat_Gas.csv  (columns: Dates, Prices)
Output:
  • Console summary  — metrics, model equation, forecast
  • Figure 1         — Price trend + prediction line
  • Figure 2         — Residual analysis (scatter + histogram)
  • Figure 3         — Gradient-descent cost curve (convergence)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#2d3040",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.framealpha": 0.2,
    "legend.edgecolor": "#3a3d4d",
    "font.family":      "sans-serif",
})

ACCENT   = "#58a6ff"   # blue  — prediction line
SCATTER  = "#f78166"   # coral — actual data points
RESIDUAL = "#3fb950"   # green — residuals
WARN     = "#d29922"   # amber — warning / annotation


# ── Core Model Functions ──────────────────────────────────────────────────────

def compute_model_output(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """
    Compute the linear model prediction  f(x) = w·x + b.

    Parameters
    ----------
    x : np.ndarray  — input feature vector (scaled)
    w : float       — weight (slope)
    b : float       — bias (intercept)

    Returns
    -------
    np.ndarray  — predicted values
    """
    return w * x + b


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Compute the Mean Squared Error cost  J(w,b) = (1/2m) Σ(f(xᵢ) − yᵢ)².

    Parameters
    ----------
    x, y : np.ndarray — feature and target vectors
    w, b  : float     — current model parameters

    Returns
    -------
    float — scalar cost value
    """
    m = x.shape[0]
    f_wb = compute_model_output(x, w, b)
    return (1 / (2 * m)) * np.sum((f_wb - y) ** 2)


def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float):
    """
    Compute partial derivatives ∂J/∂w and ∂J/∂b.

    Parameters
    ----------
    x, y : np.ndarray — feature and target vectors
    w, b  : float     — current model parameters

    Returns
    -------
    (dj_dw, dj_db) : tuple[float, float]
    """
    m = x.shape[0]
    error = compute_model_output(x, w, b) - y
    dj_dw = (1 / m) * np.dot(error, x)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w_in: float = 0.0,
    b_in: float = 0.0,
    alpha: float = 1.0e-3,
    num_iters: int = 10_000,
    tolerance: float = 1.0e-6,
    log_every: int = 500,
):
    """
    Minimise J(w,b) via batch gradient descent with early stopping.

    Parameters
    ----------
    x, y      : np.ndarray — training feature and target vectors
    w_in, b_in: float      — initial weights (default 0)
    alpha     : float      — learning rate
    num_iters : int        — maximum iterations
    tolerance : float      — convergence threshold (|ΔJ| < tolerance → stop)
    log_every : int        — record cost history every N iterations

    Returns
    -------
    w, b         : float — optimised parameters
    cost_history : list  — cost sampled at each `log_every` step
    iters_run    : int   — actual iterations completed
    """
    w, b = w_in, b_in
    prev_cost = float("inf")
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        current_cost = compute_cost(x, y, w, b)

        if i % log_every == 0:
            cost_history.append(current_cost)

        if abs(prev_cost - current_cost) < tolerance:
            cost_history.append(current_cost)
            return w, b, cost_history, i + 1

        prev_cost = current_cost

    cost_history.append(compute_cost(x, y, w, b))
    return w, b, cost_history, num_iters


# ── Data Loading & Preprocessing ─────────────────────────────────────────────

print("\n" + "═" * 60)
print("  NATURAL GAS PRICE PREDICTION — LINEAR REGRESSION")
print("═" * 60)

nat_gas_df = pd.read_csv("Nat_Gas.csv")
nat_gas_df["Dates"] = pd.to_datetime(nat_gas_df["Dates"], format="%m/%d/%y")
nat_gas_df.sort_values("Dates", inplace=True)
nat_gas_df["Numeric_dates"] = (
    nat_gas_df["Dates"] - pd.Timestamp("1970-01-01")
) // pd.Timedelta("1D")

print(f"\n  Dataset loaded  : {len(nat_gas_df):,} records")
print(f"  Date range      : {nat_gas_df['Dates'].min().date()}  →  {nat_gas_df['Dates'].max().date()}")
print(f"  Price range     : ${nat_gas_df['Prices'].min():.2f}  –  ${nat_gas_df['Prices'].max():.2f}")
print(f"  Mean price      : ${nat_gas_df['Prices'].mean():.2f}")

# Feature scaling (zero-mean, unit-variance)
scaler = StandardScaler()
x_train = scaler.fit_transform(
    nat_gas_df["Numeric_dates"].values.reshape(-1, 1)
).flatten()
y_train = nat_gas_df["Prices"].to_numpy()


# ── Model Training ───────────────────────────────────────────────────────────

print("\n  Training model …")
w, b, cost_history, iters_run = gradient_descent(x_train, y_train, 0, 0)
model_prediction = compute_model_output(x_train, w, b)


# ── Metrics ──────────────────────────────────────────────────────────────────

r2   = r2_score(y_train, model_prediction)
mae  = mean_absolute_error(y_train, model_prediction)
rmse = np.sqrt(mean_squared_error(y_train, model_prediction))
residuals = y_train - model_prediction

print("\n" + "─" * 60)
print("  MODEL PERFORMANCE")
print("─" * 60)
print(f"  R² Score (Goodness of Fit) : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE  (Mean Abs Error)      : ${mae:.2f}")
print(f"  RMSE (Root Mean Sq Error)  : ${rmse:.2f}")
print(f"  Converged in               : {iters_run:,} iterations")
print(f"  Final cost J(w,b)          : {cost_history[-1]:.6f}")
print("─" * 60)


# ── Figure 1 — Price Trend & Prediction ──────────────────────────────────────

fig1, ax = plt.subplots(figsize=(13, 6))
fig1.suptitle("Natural Gas Price — Trend & Linear Prediction", fontsize=15, fontweight="bold", y=1.01)

ax.scatter(nat_gas_df["Dates"], y_train, marker="x", c=SCATTER,
           alpha=0.7, s=40, zorder=3, label="Actual Prices")
ax.plot(nat_gas_df["Dates"], model_prediction, c=ACCENT,
        linewidth=2.5, zorder=4, label="Linear Trend (Prediction)")

# ± 1 RMSE shaded band
ax.fill_between(nat_gas_df["Dates"],
                model_prediction - rmse,
                model_prediction + rmse,
                color=ACCENT, alpha=0.12, label=f"±1 RMSE band (${rmse:.2f})")

ax.set_xlabel("Date", labelpad=8)
ax.set_ylabel("Price (USD)", labelpad=8)
ax.tick_params(axis="x", rotation=35)
ax.grid(True)
ax.legend()

# Annotate R²
ax.annotate(f"R² = {r2:.4f}", xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=11, color=ACCENT,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1d27", ec=ACCENT, alpha=0.8))

fig1.tight_layout()


# ── Figure 2 — Residual Analysis ─────────────────────────────────────────────

fig2 = plt.figure(figsize=(13, 6))
fig2.suptitle("Residual Analysis", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

# Left: Residual scatter over time
ax2a = fig2.add_subplot(gs[0])
ax2a.scatter(nat_gas_df["Dates"], residuals, c=RESIDUAL,
             alpha=0.6, s=30, edgecolors="none", zorder=3)
ax2a.axhline(0, color=WARN, linewidth=1.5, linestyle="--", label="Zero line")
ax2a.set_xlabel("Date", labelpad=8)
ax2a.set_ylabel("Residual (Actual − Predicted)", labelpad=8)
ax2a.set_title("Residuals Over Time")
ax2a.tick_params(axis="x", rotation=35)
ax2a.grid(True)
ax2a.legend()

# Right: Residual distribution
ax2b = fig2.add_subplot(gs[1])
ax2b.hist(residuals, bins=25, color=RESIDUAL, alpha=0.75, edgecolor="#0f1117")
ax2b.axvline(residuals.mean(), color=WARN, linewidth=2,
             linestyle="--", label=f"Mean: {residuals.mean():.3f}")
ax2b.axvline(residuals.mean() + residuals.std(), color=ACCENT,
             linewidth=1.5, linestyle=":", label=f"+1σ: {residuals.std():.2f}")
ax2b.axvline(residuals.mean() - residuals.std(), color=ACCENT,
             linewidth=1.5, linestyle=":")
ax2b.set_xlabel("Residual Value", labelpad=8)
ax2b.set_ylabel("Frequency", labelpad=8)
ax2b.set_title("Residual Distribution")
ax2b.grid(True)
ax2b.legend()

fig2.tight_layout()


# ── Figure 3 — Gradient Descent Convergence ───────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.suptitle("Gradient Descent — Cost Convergence", fontsize=15, fontweight="bold")

iterations = np.arange(len(cost_history))
ax3.plot(iterations, cost_history, c=ACCENT, linewidth=2)
ax3.fill_between(iterations, cost_history, alpha=0.1, color=ACCENT)
ax3.set_xlabel("Iteration Sample", labelpad=8)
ax3.set_ylabel("Cost  J(w,b)", labelpad=8)
ax3.grid(True)
ax3.annotate(f"Final cost: {cost_history[-1]:.5f}",
             xy=(iterations[-1], cost_history[-1]),
             xytext=(-80, 20), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color=WARN),
             color=WARN, fontsize=10)

fig3.tight_layout()

plt.show()


# ── Forecasting ──────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  PRICE FORECASTING")
print("─" * 60)

date_input = input("  Enter a date (mm/dd/yy) for price estimation: ").strip()
try:
    date_obj = pd.to_datetime(date_input, format="%m/%d/%y")
    numeric   = (date_obj - pd.Timestamp("1970-01-01")) // pd.Timedelta("1D")
    scaled    = scaler.transform([[numeric]])[0][0]
    estimate  = compute_model_output(scaled, w, b)

    # ± 1 RMSE confidence interval
    low, high = estimate - rmse, estimate + rmse

    days_from_last = (date_obj - nat_gas_df["Dates"].max()).days
    extrapolation  = " ⚠ Beyond training data — extrapolation" if days_from_last > 0 else ""

    print(f"\n  Date queried   : {date_obj.strftime('%B %d, %Y')}")
    print(f"  Estimated price: ${estimate:.2f}")
    print(f"  Likely range   : ${low:.2f}  –  ${high:.2f}  (±1 RMSE){extrapolation}")

except ValueError:
    print("  ✗ Invalid date format. Please use mm/dd/yy.")

print("\n" + "═" * 60 + "\n")
