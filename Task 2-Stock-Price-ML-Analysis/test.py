import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Setup & Data Fetching
ticker = "TSLA"
print(f" Fetching Data for {ticker} ")
df = yf.download(ticker, period="2y")

# 2. Feature Engineering
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['Volatility'] = (df['High'] - df['Low']) / df['Close']
df['Target'] = df['Close'].shift(-1) 
df = df.dropna()

# 3. Model Training
features = ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA50', 'Volatility']
X = df[features]
y = df['Target']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 4. PRO-LEVEL VISUALIZATION
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                               gridspec_kw={'height_ratios': [3, 1]})

# Top Chart: Price Analysis 
ax1.plot(y_test.index, y_test.values, label='Market Reality', color='#00FFCC', alpha=0.8, linewidth=1.5)
ax1.plot(y_test.index, predictions, label='AI Forecast', color='#FF3366', linestyle='--', alpha=0.9)
ax1.set_title(f'{ticker} Price Prediction: Random Forest Analysis', fontsize=16, pad=25)
ax1.set_ylabel("Price in USD ($)")
ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# LEGEND: Moved to lower right to avoid overlapping with signal box
ax1.legend(loc='lower right', facecolor='black', edgecolor='white')

# BOX 1: The Accuracy Stats (Top Left)
mae = mean_absolute_error(y_test, predictions)
ax1.text(0.02, 0.95, f'Avg Error: ${mae:.2f}', transform=ax1.transAxes, fontsize=11,
         color='#FF3366', fontweight='bold', verticalalignment='top',
         bbox=dict(facecolor='black', alpha=0.7, edgecolor='#FF3366'))

# BOX 2: The Actionable Signal (Top Right)
latest_prediction = model.predict(X.tail(1))[0]
# Use .iloc[-1] and handle potential series issues
last_close_val = df['Close'].iloc[-1]
if isinstance(last_close_val, pd.Series):
    last_close_val = last_close_val.item()

if latest_prediction > last_close_val:
    signal, signal_col = " BULLISH", "#00FFCC"
else:
    signal, signal_col = " BEARISH", "#FF3366"

ax1.text(0.98, 0.95, f'Next Day Signal: {signal}\nTarget: ${latest_prediction:.2f}', 
         transform=ax1.transAxes, fontsize=11, ha='right', va='top',
         color=signal_col, fontweight='bold', 
         bbox=dict(edgecolor=signal_col, facecolor='black', alpha=0.8))

#  Bottom Chart: Feature Importance
importances = model.feature_importances_
y_pos = np.arange(len(features))
ax2.barh(y_pos, importances, color='#8884d8', align='center')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(features)
ax2.set_title('Model Logic: Predictive Power of Features', fontsize=12)
ax2.set_xlabel("Relative Importance (0.0 to 1.0)")
ax2.grid(axis='x', color='gray', linestyle='--', alpha=0.3)

# Final spacing adjustment
plt.tight_layout(pad=4.0)
plt.subplots_adjust(hspace=0.5) 
plt.show()

# 5. FINAL TERMINAL LOGGING
print("\n" + "="*30)
print(f"FINAL REPORT FOR {ticker}")
print("="*30)
print(f"Latest Close:    ${last_close_val:.2f}")
print(f"AI Prediction:   ${latest_prediction:.2f}")
print(f"Signal:          {signal}")
print(f"Avg Model Error: ${mae:.2f}")
print("="*30)