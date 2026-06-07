import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Your Data (Wide Format)
# ================================================
df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])
print("Columns:", df_wide.columns.tolist())

# Define channels and their spend columns
channel_map = {
    'Banner (Mosaic)': 'Banner_Spend',
    'Banner (Slider)': 'Mini_Spend',
    'Sponsored Search': 'Sponsored_Search_Spend'
}

# ================================================
# 2. Function to Fit & Plot One Channel
# ================================================
def analyze_channel(channel_name, spend_col):
    # Prepare data for this channel
    data = pd.DataFrame({
        'month': df_wide['month'],
        'ad_spend': df_wide[spend_col],
        'rsv': df_wide['RSV_$']          # Using total RSV (common proxy)
    }).sort_values('month')
    
    if data['ad_spend'].sum() == 0:
        print(f"Skipping {channel_name} - No spend")
        return
    
    print(f"\n=== {channel_name} ===")
    print(f"Total Spend: ${data['ad_spend'].sum():,.0f} | Total RSV: ${data['rsv'].sum():,.0f}")
    
    # Adstock + Hill Model
    def adstock_hill(spend, beta, ec, slope, decay):
        adstocked = np.zeros_like(spend, dtype=float)
        for t in range(len(spend)):
            adstocked[t] = spend[t] + decay * (adstocked[t-1] if t > 0 else 0)
        return beta * (adstocked ** slope) / (ec ** slope + adstocked ** slope)
    
    popt, _ = curve_fit(
        adstock_hill,
        data['ad_spend'].values,
        data['rsv'].values,
        p0=[data['rsv'].max()*1.5, data['ad_spend'].median()*1.3, 0.6, 0.65],
        bounds=([0, 500, 0.1, 0.1], [np.inf, np.inf, 2.5, 0.95]),
        maxfev=10000
    )
    
    print(f"Fitted → Beta: {popt[0]:,.0f} | EC: {popt[1]:,.0f} | Slope: {popt[2]:.3f} | Decay: {popt[3]:.3f}")
    
    # Generate curves
    spend_grid = np.linspace(data['ad_spend'].min()*0.1, data['ad_spend'].max()*1.4, 600)
    response_grid = adstock_hill(spend_grid, *popt)
    
    eps = 1e-5
    mroas_grid = (adstock_hill(spend_grid + eps, *popt) - response_grid) / eps
    elasticity_grid = mroas_grid * (spend_grid / response_grid)
    elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)
    ar_ratio_grid = spend_grid / response_grid
    
    # Current values
    current_spend = data['ad_spend'].mean()
    current_rsv = data['rsv'].mean()
    current_ar = current_spend / current_rsv if current_rsv > 0 else 0
    
    # Optimal at mROAS = 2.0 (you can change this)
    idx_opt = np.argmin(np.abs(mroas_grid - 2.0))
    opt_ar = ar_ratio_grid[idx_opt]
    opt_spend = spend_grid[idx_opt]
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
    ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)', fontsize=12)
    ax1.set_ylabel('Marginal ROAS', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Break-even (mROAS=1)')
    ax1.axhline(2.0, color='green', linestyle=':', lw=2, label='Target mROAS=2')
    
    ax2 = ax1.twinx()
    ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', linewidth=2.5, label='Elasticity')
    ax2.set_ylabel('Elasticity', color='red', fontsize=12)
    
    # Current point
    idx_curr = np.argmin(np.abs(spend_grid - current_spend))
    ax1.scatter(ar_ratio_grid[idx_curr], mroas_grid[idx_curr], 
                color='purple', s=150, zorder=5, label='Current Point')
    
    plt.title(f'{channel_name}\nMarginal ROAS & Elasticity vs A:R Ratio', fontsize=14, pad=20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True, alpha=0.3)
    
    # Text box with Current vs Optimal
    textstr = f"""Current A:R  : {current_ar:.4f}
Optimal A:R  : {opt_ar:.4f}
Current Spend: ${current_spend:,.0f}/month
Optimal Spend: ${opt_spend:,.0f}/month"""
    
    plt.gcf().text(0.15, 0.15, textstr, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(f'{channel_name.replace(" ", "_").replace("(", "").replace(")", "")}_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'channel': channel_name,
        'current_ar': current_ar,
        'optimal_ar': opt_ar,
        'current_spend': current_spend,
        'optimal_spend': opt_spend
    }

# ================================================
# 3. Run for All Channels
# ================================================
results = []
for channel_name, spend_col in channel_map.items():
    res = analyze_channel(channel_name, spend_col)
    if res:
        results.append(res)

# Summary Table
summary = pd.DataFrame(results)
print("\n=== SUMMARY: Current vs Optimal A:R ===")
print(summary.round(4))
summary.to_csv('channel_optimization_summary.csv', index=False)
print("\n✅ All plots saved as separate PNG files!")
