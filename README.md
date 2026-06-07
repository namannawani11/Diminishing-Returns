import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Your Data
# ================================================
df = pd.read_csv('media_data.csv', parse_dates=['month'])
df['month'] = pd.to_datetime(df['month'])

print("Data loaded:", df.shape)
print(df.groupby('channel').agg({'ad_spend':'sum', 'rsv':'sum'}))

# Aggregate to total monthly for overall curve (recommended first)
monthly = df.groupby('month').agg({
    'ad_spend': 'sum',
    'rsv': 'sum',
    'attributed_sales': 'sum'
}).reset_index()

# ================================================
# 2. Adstock + Hill Function
# ================================================
def adstock_hill(spend, beta, ec, slope, decay):
    adstocked = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        adstocked[t] = spend[t] + decay * (adstocked[t-1] if t > 0 else 0)
    return beta * (adstocked ** slope) / (ec ** slope + adstocked ** slope)

# Fit on total
popt, _ = curve_fit(
    adstock_hill,
    monthly['ad_spend'].values,
    monthly['rsv'].values,
    p0=[np.max(monthly['rsv'])*2, monthly['ad_spend'].median()*1.5, 0.65, 0.65],
    bounds=([0, 1000, 0.1, 0.1], [np.inf, np.inf, 2.0, 0.95]),
    maxfev=10000
)

print("\nFitted Parameters (Overall):")
print(f"Beta: {popt[0]:,.0f} | EC: {popt[1]:,.0f} | Slope: {popt[2]:.3f} | Decay: {popt[3]:.3f}")

# ================================================
# 3. Generate Curves
# ================================================
spend_grid = np.linspace(monthly['ad_spend'].min()*0.1, monthly['ad_spend'].max()*1.3, 800)

response_grid = adstock_hill(spend_grid, *popt)

eps = 1e-5
mroas_grid = (adstock_hill(spend_grid + eps, *popt) - response_grid) / eps
elasticity_grid = mroas_grid * (spend_grid / response_grid)
elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)

ar_ratio_grid = spend_grid / response_grid

# ================================================
# 4. Plot with Current & Optimal Points
# ================================================
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)', fontsize=13)
ax1.set_ylabel('Marginal ROAS', color='blue', fontsize=13)
ax1.tick_params(axis='y', labelcolor='blue')

ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Break-even (mROAS=1)')
ax1.axhline(2.0, color='green', linestyle=':', lw=2, label='Target mROAS=2')

ax2 = ax1.twinx()
ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', linewidth=2.5, label='Elasticity')
ax2.set_ylabel('Elasticity', color='red', fontsize=13)

# Current point
current_spend = monthly['ad_spend'].mean()
current_rsv = monthly['rsv'].mean()
current_ar = current_spend / current_rsv

# Find closest point on curve
idx_current = np.argmin(np.abs(spend_grid - current_spend))
ax1.scatter([ar_ratio_grid[idx_current]], [mroas_grid[idx_current]], 
            color='purple', s=120, zorder=5, label='Current Investment')

plt.title('RMN Optimization: Marginal ROAS & Elasticity vs A:R Ratio\n(2+ Years Monthly Data)', fontsize=15)

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================
# 5. Recommendations
# ================================================
def recommend_at_mroas(target_mroas=2.0):
    idx = np.argmin(np.abs(mroas_grid - target_mroas))
    print(f"\n=== Optimal at Target mROAS ≈ {target_mroas} ===")
    print(f"A:R Ratio          : {ar_ratio_grid[idx]:.4f}")
    print(f"Recommended Weekly Spend : ${spend_grid[idx]/4:,.0f} (monthly: ${spend_grid[idx]:,.0f})")
    print(f"Expected RSV       : ${response_grid[idx]:,.0f}")
    print(f"Elasticity         : {elasticity_grid[idx]:.3f}")
    print(f"Current A:R        : {current_ar:.4f} → Shift {'up' if ar_ratio_grid[idx] > current_ar else 'down'}")

recommend_at_mroas(2.5)
recommend_at_mroas(2.0)
recommend_at_mroas(1.5)

# Per-channel view (quick check)
print("\n=== Per Channel Performance ===")
channel_perf = df.groupby('channel').agg({
    'ad_spend':'sum',
    'rsv':'sum',
    'attributed_sales':'sum'
}).round(0)
channel_perf['A:R'] = channel_perf['ad_spend'] / channel_perf['rsv']
channel_perf['ROAS'] = channel_perf['rsv'] / channel_perf['ad_spend']
print(channel_perf)

# Save
plt.savefig('rmn_optimization_curves.png', dpi=300, bbox_inches='tight')
monthly.to_csv('monthly_aggregated.csv', index=False)
print("\nPlots and files saved.")
