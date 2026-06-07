import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])
print("Data Shape:", df_wide.shape)

channel_map = {
    'Banner (Mosaic)': {'spend': 'Banner_Spend', 'imp': 'Banner_Impression', 'click': 'Banner_Click'},
    'Banner (Slider)': {'spend': 'Mini_Spend', 'imp': 'Mini_Impression', 'click': 'Mini_Click'},
    'Sponsored Search': {'spend': 'Sponsored_Search_Spend', 'imp': 'Sponsored_Search_Impression', 'click': 'Sponsored_Search_Click'}
}

def analyze_channel(channel_name, cols):
    data = df_wide[['month', cols['spend'], cols['imp'], cols['click'], 'RSV_$']].copy()
    data = data.rename(columns={
        cols['spend']: 'ad_spend', 
        cols['imp']: 'impressions', 
        cols['click']: 'clicks', 
        'RSV_$': 'rsv'
    })
    data = data.sort_values('month').reset_index(drop=True)
    
    if data['ad_spend'].sum() == 0:
        print(f"Skipping {channel_name} - No spend")
        return
    
    print(f"\n=== {channel_name} ===")
    print(f"Total Spend: ${data['ad_spend'].sum():,.0f} | Total RSV: ${data['rsv'].sum():,.0f}")
    
    # Scaling
    spend_scale = data['ad_spend'].mean()
    imp_scale = data['impressions'].mean()
    rsv_scale = data['rsv'].mean()
    
    spend_norm = data['ad_spend'].values / spend_scale
    imp_norm = data['impressions'].values / imp_scale
    rsv_norm = data['rsv'].values / rsv_scale
    
    def hybrid_model(spend, imp, beta, ec, slope, decay, imp_coef):
        adstocked = np.zeros_like(spend, dtype=float)
        for t in range(len(spend)):
            adstocked[t] = spend[t] + decay * (adstocked[t-1] if t > 0 else 0)
        hill = beta * (adstocked ** slope) / (ec ** slope + adstocked ** slope)
        return hill + imp_coef * imp
    
    best_r2 = -np.inf
    best_decay = 0.6
    best_popt = None
    
    for decay in [0.4, 0.5, 0.6, 0.7, 0.8]:
        try:
            def model_to_fit(s, b, e, sl, ic):
                return hybrid_model(s, imp_norm, b, e, sl, decay, ic)
            
            popt, _ = curve_fit(
                model_to_fit,
                spend_norm, rsv_norm,
                p0=[1.2, 1.0, 0.7, 0.4],
                bounds=([0.1, 0.01, 0.1, -5], [10, 5, 2.5, 10]),
                maxfev=10000
            )
            pred = hybrid_model(spend_norm, imp_norm, *popt, decay)
            r2 = 1 - np.sum((rsv_norm - pred)**2) / np.sum((rsv_norm - rsv_norm.mean())**2)
            
            print(f"  Decay={decay:.1f} | R²={r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_decay = decay
                best_popt = np.append(popt, decay)
        except Exception as e:
            print(f"  Decay={decay:.1f} failed")
            continue
    
    if best_popt is None:
        print("⚠️  Hill fitting failed. Using simple adstocked spend model.")
        best_decay = 0.6
        # Simple fallback
        adstocked = np.zeros(len(spend_norm))
        for t in range(len(spend_norm)):
            adstocked[t] = spend_norm[t] + best_decay * (adstocked[t-1] if t > 0 else 0)
        coef = np.polyfit(adstocked, rsv_norm, 1)
        mroas_grid = np.full(600, coef[0] * rsv_scale / spend_scale)
        spend_grid = np.linspace(data['ad_spend'].min()*0.1, data['ad_spend'].max()*1.5, 600)
        response_grid = (coef[0] * (spend_grid/spend_scale) + coef[1]) * rsv_scale
    else:
        print(f"✅ Best Decay = {best_decay:.2f} (R² = {best_r2:.4f})")
        spend_grid = np.linspace(data['ad_spend'].min()*0.1, data['ad_spend'].max()*1.5, 600)
        spend_norm_grid = spend_grid / spend_scale
        imp_norm_grid = np.full_like(spend_norm_grid, np.mean(imp_norm))
        
        response_norm = hybrid_model(spend_norm_grid, imp_norm_grid, *best_popt[:-1], best_decay)
        response_grid = response_norm * rsv_scale
        
        eps = 1e-5
        response_plus = hybrid_model(spend_norm_grid + eps, imp_norm_grid, *best_popt[:-1], best_decay) * rsv_scale
        mroas_grid = (response_plus - response_grid) / eps
    
    elasticity_grid = mroas_grid * (spend_grid / response_grid)
    elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)
    ar_ratio_grid = spend_grid / response_grid
    
    current_spend = data['ad_spend'].mean()
    current_rsv = data['rsv'].mean()
    current_ar = current_spend / current_rsv if current_rsv > 0 else 0
    
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
    ax1.scatter(ar_ratio_grid[idx_curr], mroas_grid[idx_curr], color='purple', s=180, label='Current Point')
    
    plt.title(f'{channel_name}\nHill + Adstock + Impressions (Decay={best_decay:.2f})')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True, alpha=0.3)
    
    textstr = f"Current A:R : {current_ar:.4f}\nCurrent Spend: ${current_spend:,.0f}/mo"
    plt.gcf().text(0.15, 0.15, textstr, fontsize=11, bbox=dict(facecolor='white', alpha=0.85))
    
    plt.tight_layout()
    safe_name = channel_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f'{safe_name}_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run for all channels
for name, cols in channel_map.items():
    analyze_channel(name, cols)
