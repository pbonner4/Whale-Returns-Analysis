# Whale-Returns-Analysis

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
%matplotlib inline

#### Reading the whale returns CSV
whale_returns_csv = Path('Whale_Returns.csv')
whale_returns = pd.read_csv(whale_returns_csv, index_col='Date', infer_datetime_format = True, parse_dates = True)
whale_returns = whale_returns.sort_index()
whale_returns.head()

#### Count Nulls
whale_returns.isnull().sum()

#### Drop Nulls
whale_returns.dropna(inplace=True)
whale_returns.isnull().sum()

#### Reading algorithmic returns
algo_returns_csv = Path('algo_returns.csv')
algo_returns = pd.read_csv(algo_returns_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
algo_returns = algo_returns.sort_index()
algo_returns.head()

#### Count nulls
algo_returns.isnull().sum()

#### Drop nulls
algo_returns.dropna(inplace=True)
algo_returns.isnull().sum()

#### Read sp500 returns
sp500_history_csv = Path('sp500_history.csv')
sp500_history = pd.read_csv(sp500_history_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
sp500_history = sp500_history.sort_index()
sp500_history.head()

#### Check Data Types
sp500_history.dtypes

#### Fix Data Types
sp500_history['Close'] = sp500_history['Close'].str.replace('$','')
sp500_history['Close'] = sp500_history['Close'].astype('float')
sp500_history.dtypes

#### Calculate Daily Returns
sp500_returns = sp500_history.pct_change()
sp500_returns.head()

#### Drop Nulls
sp500_returns.dropna(inplace=True)
sp500_returns.head()

#### Rename Column
sp500_returns.rename(columns={'Close' : 'S&P 500'}, inplace=True)
sp500_returns.head()

#### Combine Whale, Algorithmic, and S&P 500 Returns
all_portfolios = pd.concat([whale_returns, algo_returns, sp500_returns], axis='columns', join='inner')
all_portfolios.head()

#### Calculate and Plot the daily returns and cumulative returns
all_portfolios.plot(figsize=(19,10),title='Daily Returns')

#### Algo 1 clearly performing the best with the S&P500 in the middle of the group. Paulson & C. INC. actually losing money falling from 1.0 to about 0.8.
cumulative_returns =(1 + all_portfolios).cumprod()
cumulative_returns.plot(figsize=(20,10), title='Cumulative Returns')

#### Box Plot to visually show risk
#### The Daily volatility of an investment in Berkshire Hathaway is much greater than that of Algorithmic Portfolio 1 or the S&P 500 overall
all_portfolios.plot.box(figsize=(20,10), title='Portfolio Risk')

#### Daily Standard Deviations
#### Daily standard deviation of Tiger Global Management LLC and Berkshire Hathaway INC are the highest.
all_portfolios.std()

#### Compare the standard deviations of the entire portfolio to the S&P 500. We find that only Tiger and Berkshire have higher risk than the market as a whole.
sp500_risk = all_portfolios["S&P 500"].std()
all_portfolios.std() > sp500_risk

#### Calculate the annualized standard deviation (252 trading days)
all_portfolios.std() * np.sqrt(252)

#### Calculate and plot the rolling standard deviation for the S&P 500 using a 21 day window
#### Note: Risk of all portfolios seem to have increased toward the end of 2018 and beginning of 2019.
all_portfolios.rolling(window=21).std().plot(figsize=(20, 10), title='21 Day Rolling Standard Deviation')

#### Correlation
#### Lighter colors denote the stronger correlation
#### Algo 2 and Soros Fund Management LLC have the strongest correlation with the market index. So their returns would not be much different than if you invested in a S&P 500 ETF.
#### Algo 1 has the lowest correlation with all the Hedge Funds and the S&P 500, suggesting the most potential for diversification benefits
corr_df = all_portfolios.corr()
corr_df.style.background_gradient(cmap='summer')

#### Calculate Beta for a single portfolio compared to the total market (S&P 500)
covariance = all_portfolios["BERKSHIRE HATHAWAY INC"].rolling(window=60).cov(all_portfolios['S&P 500'])
variance = all_portfolios["S&P 500"].rolling(60).var()
(covariance/variance).plot(figsize=(20,10), title="Berskshire Hathaway Inc. Beta")

#### Calculate a rolling window using the exponentially weighted moving average. 
all_portfolios.ewm(halflife=21).std().plot(figsize=(20, 10), title="Exponentially Weighted Average")

#### Annualize Sharpe Ratios
#### Paulson & Co.INC. do not and Tiger Global Management LLC do not compensate for the level of risk that they take.
#### Only portfolio to outperform the benchmark is Algo 1
sharpe_ratios = (all_portfolios.mean() * 252) / (all_portfolios.std() * np.sqrt(252))
sharpe_ratios

#### Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind="bar", title="Sharpe Ratios")

## Custom Portfolio - Chosen stocks (Google, Apple, and Costco)

#### Reading the Google Csv
google_historical_csv = Path('goog_historical.csv')
google_historical = pd.read_csv(
    google_historical_csv, index_col = 'Trade DATE', infer_datetime_format=True, parse_dates=True)
google__historical = google_historical.sort_index()
google_historical.head()

#### Reading the Apple Csv
apple_historical_csv = Path('aapl_historical.csv')
apple_historical = pd.read_csv(
    apple_historical_csv, index_col = 'Trade DATE', infer_datetime_format=True, parse_dates=True)
apple__historical = apple_historical.sort_index()
apple_historical.head()

#### Reading the Costco Csv
costco_historical_csv = Path('cost_historical.csv')
costco_historical = pd.read_csv(
    costco_historical_csv, index_col = 'Trade DATE', infer_datetime_format=True, parse_dates=True)
costco__historical = costco_historical.sort_index()
costco_historical.head()

#### Combine Google, Apple, and Costco returns
all_stocks = pd.concat([google_historical, apple_historical, costco_historical], axis="rows", join="inner")
all_stocks.head()

all_stocks = all_stocks.reset_index()
all_stocks.head()

portfolio = all_stocks.pivot_table(values="NOCP", index="Trade DATE", columns="Symbol")
portfolio.head()

daily_returns = portfolio.pct_change().dropna()
daily_returns.head()

#### Caculating the weighted returns for the portfolio assuming an equal number of shares for each stock
weights = [1/3, 1/3, 1/3]
portfolio_returns = daily_returns.dot(weights)
portfolio_returns.head()

#### Joining my portfolio to the data frame that contains all of the portfolio returns
all_portfolios["Custom"] = portfolio_returns
all_portfolios.tail()

#### Only comapare dates where the custom portfolio has dates
all_portfolios.dropna(inplace=True)

#### Rolling statistics
all_portfolios.rolling(window=21).std().plot(figsize=(20, 10), title="21 Day Rolling Standard Deviation")

#### Annualized Sharpe Ratios
#### Custom portfolio has done well. It has outperformed, on a risk adjusted basis, the benchmark index. 
#### Note: Still lower than Algo 1
sharpe_ratios = (all_portfolios.mean() * 252) / (all_portfolios.std() * np.sqrt(252))
sharpe_ratios

#### Visualize the Sharpe ratios as a bar plot
sharpe_ratios.plot(kind="bar", title="Sharpe Ratios")

#### Correlation Analysis to determine which stocks are correlated
#### Custom portfolio has a fairly high correlation with the S&P 500
#### Compared to Algo 1, Custom Portfolio does not provide as much diversification potential
df = all_portfolios.corr()
df.style.background_gradient(cmap="summer")

#### Beta of Custom Portfolio
covariance = all_portfolios["Custom"].rolling(window=60).cov(all_portfolios["S&P 500"])
variance = all_portfolios["S&P 500"].rolling(60).var()
(covariance / variance).plot(figsize=(20, 10), title="Custom Portfolio Beta")
