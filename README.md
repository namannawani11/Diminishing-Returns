import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 1. Load Data
# ================================================
df_wide = pd.read_csv('media_data.csv', parse_dates=['month'])
print("Columns:", df_wide.columns.tolist())

channel_map = {
    'Banner (Mosaic)': 'Banner_Spend',
    'Banner (Slider)': 'Mini_Spend',
    'Sponsored Search': 'Sponsored_Search_Spend'
}

# ================================================
# 2. Function for Simple Regression per Channel
# ================================================
def analyze_channel(channel_name, spend_col, adstock_decay=0.65):
    data = df_wide[['month', spend_col, 'Events', 'RSV_$']].copy()
    data = data.rename(columns={spend_col: 'ad_spend', 'RSV_$': 'rsv'})
    data = data.sort_values('month').reset_index(drop=True)
    
    if data['ad_spend'].sum() == 0:
        print(f"Skipping {channel_name} - No spend")
        return None
    
    # Calculate ASP
    data['asp'] = data['rsv'] / data['Events'].replace(0, np.nan)
    data['asp'] = data['asp'].fillna(data['asp'].median())
    
    print(f"\n=== {channel_name} ===")
    print(f"Total Spend: ${data['ad_spend'].sum():,.0f} | Total RSV: ${data['rsv'].sum():,.0f}")
    
    # === Adstock ===
    adstocked = np.zeros(len(data))
    for t in range(len(data)):
        adstocked[t] = data['ad_spend'].iloc[t] + adstock_decay * (adstocked[t-1] if t > 0 else 0)
    data['adstocked_spend'] = adstocked
    
    # === Simple Multiple Linear Regression ===
    X = data[['adstocked_spend', 'Events', 'asp']]
    y = data['rsv']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Model Coefficients:")
    print(f"  Adstocked Spend: {model.coef_[0]:.4f}  (this is approx average mROAS)")
    print(f"  Events: {model.coef_[1]:.4f}")
    print(f"  ASP: {model.coef_[2]:.4f}")
    
    # === Generate Prediction Grid ===
    spend_grid = np.linspace(data['ad_spend'].min()*0.1, data['ad_spend'].max()*1.5, 600)
    
    # Approximate adstock on grid
    adstock_grid = np.zeros_like(spend_grid)
    for t in range(len(spend_grid)):
        adstock_grid[t] = spend_grid[t] + adstock_decay * (adstock_grid[t-1] if t > 0 else 0)
    
    # Use mean values for other variables (standard approach)
    mean_events = data['Events'].mean()
    mean_asp = data['asp'].mean()
    
    X_grid = pd.DataFrame({
        'adstocked_spend': adstock_grid,
        'Events': mean_events,
        'asp': mean_asp
    })
    
    response_grid = model.predict(X_grid)
    
    # Marginal ROAS = coefficient of adstocked_spend (constant in linear model)
    mroas_grid = np.full_like(response_grid, model.coef_[0])
    
    # Elasticity
    elasticity_grid = mroas_grid * (spend_grid / response_grid)
    elasticity_grid = np.nan_to_num(elasticity_grid, nan=0)
    
    ar_ratio_grid = spend_grid / response_grid
    
    # Current values
    current_spend = data['ad_spend'].mean()
    current_rsv = data['rsv'].mean()
    current_ar = current_spend / current_rsv if current_rsv > 0 else 0
    
    # For linear model, optimal is where mROAS = target (flat line)
    target_mroas = 2.0
    print(f"\nConstant Marginal ROAS from model: {model.coef_[0]:.3f}")
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    ax1.plot(ar_ratio_grid, mroas_grid, 'b-', linewidth=3, label='Marginal ROAS')
    ax1.set_xlabel('Advertising Spend / Total RSV (A:R Ratio)', fontsize=12)
    ax1.set_ylabel('Marginal ROAS', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_ylim(0, 5)
    
    ax1.axhline(1.0, color='red', linestyle='--', lw=2, label='Break-even (mROAS=1)')
    ax1.axhline(2.0, color='green', linestyle=':', lw=2, label='Target mROAS=2')
    
    ax2 = ax1.twinx()
    ax2.plot(ar_ratio_grid, elasticity_grid, 'r--', linewidth=2.5, label='Elasticity')
    ax2.set_ylabel('Elasticity', color='red', fontsize=12)
    
    # Current point
    idx_curr = np.argmin(np.abs(spend_grid - current_spend))
    ax1.scatter(ar_ratio_grid[idx_curr], mroas_grid[idx_curr], 
                color='purple', s=150, zorder=5, label='Current Point')
    
    plt.title(f'{channel_name}\nSimple Linear Regression on Adstocked Spend + Events + ASP', fontsize=13, pad=20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    ax1.grid(True, alpha=0.3)
    
    textstr = f"""Current A:R     : {current_ar:.4f}
Model mROAS     : {model.coef_[0]:.3f}
Current Spend   : ${current_spend:,.0f}/month
Recommended     : Increase if mROAS > 2"""
    
    plt.gcf().text(0.15, 0.15, textstr, fontsize=11, 
                   bbox=dict(facecolor='white', alpha=0.85, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    safe_name = channel_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f'{safe_name}_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'channel': channel_name,
        'current_ar': current_ar,
        'model_mroas': model.coef_[0],
        'current_spend': current_spend
    }

# ================================================
# 3. Run Analysis
# ================================================
results = []
for channel_name, spend_col in channel_map.items():
    res = analyze_channel(channel_name, spend_col, adstock_decay=0.65)
    if res:
        results.append(res)

summary = pd.DataFrame(results)
print("\n=== SUMMARY ===")
print(summary.round(4))
summary.to_csv('channel_optimization_summary.csv', index=False)

print("\n✅ Done! Three separate graphs created.")
