# Short-Term Stock Price Predictor (Random Forest)

## Project Overview
This project uses Machine Learning to predict the next day's closing price for **$TSLA** (Tesla) using historical data from Yahoo Finance. Instead of simple linear trends, it utilizes an ensemble **Random Forest Regressor** to capture non-linear market behaviors.

## Key Features
- **Technical Indicators:** Engineered features including 10-day & 50-day Moving Averages and Daily Volatility.
- **Ensemble Learning:** Uses 100 decision trees to reach a consensus on price movements.
- **Visual Analytics:** A dual-plot dashboard showing price forecasts and feature importance.

## How to Run`
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python data_analysis.py`

## Results
*Mean Absolute Error (MAE): ~$15.65*

## Insights
The model identified the **Daily Low** and **Daily High** prices as the most significant predictors, suggesting that intra-day support/resistance levels are more informative than the opening price for this specific ticker.
