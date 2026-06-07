import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Data
# ================================================
df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])

channel_map = {
    'Banner (Mosaic)': 'Banner_Spend',
    'Banner (Slider)': 'Mini_Spend',
    'Sponsored Search': 'Sponsored_Search_Spend'
}

# ================================================
# 2. Function to Analyze One Channel
# ================================================
def analyze_channel(channel_name, spend_col, adstock_decay=0.7, poly_degree=2):
    data = pd.DataFrame({
        'month': df_wide['month'],
        'ad_spend': df_wide[spend_col],
        'rsv': df_wide['RSV_$']
    }).sort_values('month').reset_index(drop=True)
    
    if data['ad_spend'].sum() == 0:
        print(f"Skipping {channel_name} - No spend")
        return None
    
    print(f"\n=== {channel_name} ===")
    print(f"Total Spend: ${data['ad_spend'].sum():,.0f} | Total RSV: ${data['rsv'].sum():,.0f}")
    
    # === Step 1: Apply Adstock ===
    adstocked = np.zeros(len(data))
    for t in range(len(data)):
        adstocked[t] = data['ad_spend'].iloc[t] + adstock_decay * (adstocked[t-1] if t > 0 else 0)
    data['adstocked_spend'] = adstocked
    
    # === Step 2: Polynomial Regression ===
    X = data[['adstocked_spend']]
    y = data['rsv']
    
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # === Step 3: Generate Curves ===
    spend_grid = np.linspace(data['ad_spend'].min()*0.1, data['ad_spend'].max()*1.5, 600)
    
    # Adstock on grid (approximate)
    adstock_grid = np.zeros_like(spend_grid)
    for t in range(len(spend_grid)):
        adstock_grid[t] = spend_grid[t] + adstock_decay * (adstock_grid[t-1] if t > 0 else 0)
    
    X_grid_poly = poly.transform(adstock_grid.reshape(-1, 1))
    response_grid = model.predict(X_grid_poly)
    
    # Marginal ROAS (numerical derivative)
    eps = 1e-4
    response_plus = model.predict(poly.transform((adstock_grid + eps).reshape(-1, 1)))
    mroas_grid = (response_plus - response_grid) / eps
    
    # Elasticity
    elasticity_grid = mroas_grid * (spend_grid / response_grid)
    elasticity_grid = np.nan_to_num(elasticity_grid, nan=0, posinf=0, neginf=0)
    
    ar_ratio_grid = spend_grid / response_grid
    
    # Current values
    current_spend = data['ad_spend'].mean()
    current_rsv = data['rsv'].mean()
    current_ar = current_spend / current_rsv if current_rsv > 0 else 0
    
    # Optimal at mROAS ≈ 2.0
    idx_opt = np.argmin(np.abs(mroas_grid - 2.0))
    opt_ar = ar_ratio_grid[idx_opt]
    opt_spend = spend_grid[idx_opt]
    
    # ====================== PLOT ======================
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
    ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)', fontsize=12)
    ax1.set_ylabel('Marginal ROAS', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_ylim(0, 5)                    # Y-axis limited to 0-5 as requested
    
    ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Break-even (mROAS=1)')
    ax1.axhline(2.0, color='green', linestyle=':', lw=2, label='Target mROAS=2')
    
    ax2 = ax1.twinx()
    ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', linewidth=2.5, label='Elasticity')
    ax2.set_ylabel('Elasticity', color='red', fontsize=12)
    
    # Current point
    idx_curr = np.argmin(np.abs(spend_grid - current_spend))
    ax1.scatter(ar_ratio_grid[idx_curr], mroas_grid[idx_curr], 
                color='purple', s=150, zorder=5, label='Current Point')
    
    plt.title(f'{channel_name}\nPolynomial Regression on Adstocked Spend', fontsize=14, pad=20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True, alpha=0.3)
    
    # Text box
    textstr = f"""Current A:R  : {current_ar:.4f}
Optimal A:R  : {opt_ar:.4f}
Current Spend: ${current_spend:,.0f}/month
Optimal Spend: ${opt_spend:,.0f}/month
Adstock Decay: {adstock_decay}"""
    
    plt.gcf().text(0.15, 0.15, textstr, fontsize=11, 
                   bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    safe_name = channel_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f'{safe_name}_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'channel': channel_name,
        'current_ar': current_ar,
        'optimal_ar': opt_ar,
        'current_spend': current_spend,
        'optimal_spend': opt_spend,
        'adstock_decay': adstock_decay
    }

# ================================================
# 3. Run for All Channels
# ================================================
results = []
for channel_name, spend_col in channel_map.items():
    res = analyze_channel(channel_name, spend_col, adstock_decay=0.65, poly_degree=2)
    if res:
        results.append(res)

summary = pd.DataFrame(results)
print("\n=== SUMMARY: Current vs Optimal ===")
print(summary.round(4))
summary.to_csv('channel_optimization_summary.csv', index=False)

print("\n✅ Done! Check the three PNG files.")
