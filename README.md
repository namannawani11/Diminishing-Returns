import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Your Data
# ================================================
df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])
print("Columns found:", df_wide.columns.tolist())

# Reshape wide → long format
channels = {
    'Banner (Mosaic)': ['Banner_Spend', 'Banner_Impression', 'Banner_Click'],
    'Banner (Slider)': ['Mini_Spend', 'Mini_Impression', 'Mini_Click'],
    'Sponsored Search': ['Sponsored_Search_Spend', 'Sponsored_Search_Impression', 'Sponsored_Search_Click']
}

long_data = []
for channel_name, prefixes in channels.items():
    channel_df = df_wide[['month']].copy()
    for prefix in prefixes:
        if prefix in df_wide.columns:
            metric = prefix.split('_')[-1].lower()  # spend, impression, click
            if 'Spend' in prefix:
                metric = 'ad_spend'
            elif 'Impression' in prefix:
                metric = 'impressions'
            elif 'Click' in prefix:
                metric = 'clicks'
            channel_df[metric] = df_wide[prefix]
    
    channel_df['channel'] = channel_name
    # Add RSV and Events (common columns)
    channel_df['rsv'] = df_wide['RSV_$']
    if 'Events' in df_wide.columns:
        channel_df['events'] = df_wide['Events']
    
    long_data.append(channel_df)

df = pd.concat(long_data, ignore_index=True)

print("\nLong format sample:")
print(df.head())
print("\nChannels:", df['channel'].unique())

# ================================================
# 2. Monthly Total Aggregation (for overall curve)
# ================================================
monthly = df.groupby('month').agg({
    'ad_spend': 'sum',
    'rsv': 'sum',
    'events': 'sum' if 'events' in df.columns else 'first'
}).reset_index().sort_values('month')

print(f"\nMonthly rows: {len(monthly)} | Total Spend: ${monthly['ad_spend'].sum():,.0f} | Total RSV: ${monthly['rsv'].sum():,.0f}")

# ================================================
# 3. Adstock + Hill Response Model
# ================================================
def adstock_hill(spend, beta, ec, slope, decay):
    adstocked = np.zeros_like(spend, dtype=float)
    for t in range(len(spend)):
        adstocked[t] = spend[t] + decay * (adstocked[t-1] if t > 0 else 0)
    return beta * (adstocked ** slope) / (ec ** slope + adstocked ** slope)

popt, _ = curve_fit(
    adstock_hill,
    monthly['ad_spend'].values,
    monthly['rsv'].values,
    p0=[monthly['rsv'].max()*2, monthly['ad_spend'].median()*1.5, 0.65, 0.6],
    bounds=([0, 1000, 0.1, 0.1], [np.inf, np.inf, 2.5, 0.95]),
    maxfev=10000
)

print("\nFitted Parameters:")
print(f"Beta: {popt[0]:,.0f} | EC: {popt[1]:,.0f} | Slope: {popt[2]:.3f} | Decay: {popt[3]:.3f}")

# ================================================
# 4. Generate Curves
# ================================================
spend_grid = np.linspace(monthly['ad_spend'].min()*0.1, monthly['ad_spend'].max()*1.4, 800)
response_grid = adstock_hill(spend_grid, *popt)

eps = 1e-5
mroas_grid = (adstock_hill(spend_grid + eps, *popt) - response_grid) / eps
elasticity_grid = mroas_grid * (spend_grid / response_grid)
elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)

ar_ratio_grid = spend_grid / response_grid

# ================================================
# 5. Plot - Exactly as you wanted
# ================================================
fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)', fontsize=13)
ax1.set_ylabel('Marginal ROAS', color='blue', fontsize=13)
ax1.tick_params(axis='y', labelcolor='blue')

ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Break-even (mROAS = 1)')
ax1.axhline(2.0, color='green', linestyle=':', lw=2, label='Target (mROAS = 2)')

ax2 = ax1.twinx()
ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', linewidth=2.5, label='Elasticity')
ax2.set_ylabel('Elasticity', color='red', fontsize=13)

# Current point
current_spend = monthly['ad_spend'].mean()
current_rsv = monthly['rsv'].mean()
current_ar = current_spend / current_rsv

idx_current = np.argmin(np.abs(spend_grid - current_spend))
ax1.scatter(ar_ratio_grid[idx_current], mroas_grid[idx_current], 
            color='purple', s=180, zorder=5, label='Current Investment Point')

plt.title('RMN Budget Optimization\nMarginal ROAS & Elasticity vs A:R Ratio', fontsize=15, pad=20)
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
ax1.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

# ================================================
# 6. Recommendations (including budget shift insight)
# ================================================
def recommend_at_mroas(target=2.0):
    idx = np.argmin(np.abs(mroas_grid - target))
    print(f"\n=== Optimal at Target mROAS ≈ {target} ===")
    print(f"A:R Ratio                  : {ar_ratio_grid[idx]:.4f}")
    print(f"Recommended Monthly Spend  : ${spend_grid[idx]:,.0f}")
    print(f"Expected Monthly RSV       : ${response_grid[idx]:,.0f}")
    print(f"Elasticity                 : {elasticity_grid[idx]:.3f}")
    print(f"Current Monthly Spend      : ${current_spend:,.0f} (A:R = {current_ar:.4f})")

recommend_at_mroas(2.5)
recommend_at_mroas(2.0)
recommend_at_mroas(1.5)

# Per-channel performance (key for shifting budget to Sponsored Search)
print("\n=== Per Channel Performance (Total Period) ===")
channel_perf = df.groupby('channel').agg({
    'ad_spend': 'sum',
    'rsv': 'sum',
    'events': 'sum' if 'events' in df.columns else 'first'
}).round(0)
channel_perf['A:R'] = (channel_perf['ad_spend'] / channel_perf['rsv']).round(4)
channel_perf['ROAS'] = (channel_perf['rsv'] / channel_perf['ad_spend']).round(2)
print(channel_perf)

# Save
plt.savefig('rmn_optimization_curves.png', dpi=300, bbox_inches='tight')
monthly.to_csv('monthly_aggregated.csv', index=False)
print("\n✅ Saved: rmn_optimization_curves.png and monthly_aggregated.csv")
