import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Data
# ================================================
df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])

channel_map = {
    'Banner (Mosaic)': {'spend': 'Banner_Spend', 'imp': 'Banner_Impression', 'click': 'Banner_Click'},
    'Banner (Slider)': {'spend': 'Mini_Spend', 'imp': 'Mini_Impression', 'click': 'Mini_Click'},
    'Sponsored Search': {'spend': 'Sponsored_Search_Spend', 'imp': 'Sponsored_Search_Impression', 'click': 'Sponsored_Search_Click'}
}

# ================================================
# 2. Hybrid Hill + Controls Model
# ================================================
def analyze_channel(channel_name, cols):
    data = df_wide[['month', cols['spend'], cols['imp'], cols['click'], 'RSV_$', 'Events']].copy()
    data = data.rename(columns={cols['spend']: 'ad_spend', cols['imp']: 'impressions', 
                                cols['click']: 'clicks', 'RSV_$': 'rsv'})
    data = data.sort_values('month').reset_index(drop=True)
    
    if data['ad_spend'].sum() == 0:
        print(f"Skipping {channel_name}")
        return None
    
    print(f"\n=== {channel_name} ===")
    print(f"Total Spend: ${data['ad_spend'].sum():,.0f} | Total RSV: ${data['rsv'].sum():,.0f}")
    
    # Scale for stability
    spend_scale = data['ad_spend'].mean()
    imp_scale = data['impressions'].mean()
    rsv_scale = data['rsv'].mean()
    
    spend_norm = data['ad_spend'].values / spend_scale
    imp_norm = data['impressions'].values / imp_scale
    rsv_norm = data['rsv'].values / rsv_scale
    
    # Hybrid Hill + Adstock on Spend + Linear Controls
    def hybrid_model(spend, imp, beta, ec, slope, decay, imp_coef):
        adstocked_spend = np.zeros_like(spend)
        for t in range(len(spend)):
            adstocked_spend[t] = spend[t] + decay * (adstocked_spend[t-1] if t > 0 else 0)
        
        hill_part = beta * (adstocked_spend ** slope) / (ec ** slope + adstocked_spend ** slope)
        return hill_part + imp_coef * imp   # Impressions as additional driver
    
    # Auto best decay
    best_r2 = -np.inf
    best_decay = 0.6
    best_popt = None
    
    for decay in np.arange(0.3, 0.91, 0.1):
        try:
            p0 = [np.max(rsv_norm)*1.1, np.median(spend_norm)*1.1, 0.7, decay, 0.3]
            popt, _ = curve_fit(
                lambda x, b, e, sl, ic: hybrid_model(x, imp_norm, b, e, sl, decay, ic),
                spend_norm, rsv_norm,
                p0=p0, bounds=([0.1, 0.01, 0.1, 0.1, -2], [10, 5, 2.5, 0.95, 5]),
                maxfev=10000
            )
            pred = hybrid_model(spend_norm, imp_norm, *popt, decay)
            r2 = 1 - np.sum((rsv_norm - pred)**2) / np.sum((rsv_norm - rsv_norm.mean())**2)
            
            if r2 > best_r2:
                best_r2 = r2
                best_decay = decay
                best_popt = popt
        except:
            continue
    
    print(f"Best Adstock Decay: {best_decay:.2f} (R² = {best_r2:.4f})")
    
    # Generate grid
    spend_grid = np.linspace(data['ad_spend'].min()*0.05, data['ad_spend'].max()*1.6, 800)
    imp_grid = np.linspace(data['impressions'].min()*0.8, data['impressions'].max()*1.3, 800)
    spend_norm_grid = spend_grid / spend_scale
    imp_norm_grid = imp_grid / imp_scale
    
    # Predict
    response_norm = hybrid_model(spend_norm_grid, imp_norm_grid, *best_popt, best_decay)
    response_grid = response_norm * rsv_scale
    
    # Marginal ROAS (derivative w.r.t. spend)
    eps = 1e-5
    response_plus = hybrid_model(spend_norm_grid + eps, imp_norm_grid, *best_popt, best_decay) * rsv_scale
    mroas_grid = (response_plus - response_grid) / eps
    
    elasticity_grid = mroas_grid * (spend_grid / response_grid)
    elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)
    ar_ratio_grid = spend_grid / response_grid
    
    # Current values
    current_spend = data['ad_spend'].mean()
    current_rsv = data['rsv'].mean()
    current_ar = current_spend / current_rsv if current_rsv > 0 else 0
    
    # Optimal
    valid = mroas_grid > 0
    idx_opt = np.argmin(np.abs(mroas_grid[valid] - 2.0))
    idx_opt = np.where(valid)[0][idx_opt]
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(13, 8))
    ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
    ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)')
    ax1.set_ylabel('Marginal ROAS', color='blue')
    ax1.set_ylim(0, 5)
    ax1.axhline(1.0, color='red', linestyle='--', label='Break-even')
    ax1.axhline(2.0, color='green', linestyle=':', label='Target')
    
    ax2 = ax1.twinx()
    ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', label='Elasticity')
    ax2.set_ylabel('Elasticity', color='red')
    
    idx_curr = np.argmin(np.abs(spend_grid - current_spend))
    ax1.scatter(ar_ratio_grid[idx_curr], mroas_grid[idx_curr], color='purple', s=180, label='Current')
    
    plt.title(f'{channel_name}\nHill + Adstock + Impressions (Best Decay = {best_decay:.2f})')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True, alpha=0.3)
    
    textstr = f"""Current A:R   : {current_ar:.4f}
Optimal A:R   : {ar_ratio_grid[idx_opt]:.4f}
Current Spend : ${current_spend:,.0f}
Optimal Spend : ${spend_grid[idx_opt]:,.0f}
Best Decay    : {best_decay:.2f}"""
    
    plt.gcf().text(0.15, 0.15, textstr, bbox=dict(facecolor='white', alpha=0.85))
    plt.tight_layout()
    safe_name = channel_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f'{safe_name}_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================================
# 3. Run
# ================================================
for name, cols in channel_map.items():
    analyze_channel(name, cols)
