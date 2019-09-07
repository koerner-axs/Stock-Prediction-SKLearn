from math import ceil
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLars, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.pipeline import make_pipeline

import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

def doAnalysis(stock):
	df = pd.read_csv('./datasets/' + stock + '.csv')
	mpl.rc('figure', figsize=(16, 12))

	def convert_dates(row):
		row['Date'] = datetime.datetime(*[int(x) for x in row['Date'].split('-')])
		return row

	# Convert string dates in YYYY-MM-DD-Format to datetime dates.
	df = df.apply(convert_dates, axis=1)

	# Select and manufacture new features.
	rollingMeanWindowSize = 200
	dfreg = df.loc[:, ['Date', 'Adj Close', 'Volume']]
	dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
	dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
	dfreg['AVG_rolling_PCT_change'] = dfreg['PCT_change'].rolling(window=10).mean()
	dfreg['AVG_rolling_eigth'] = df['Adj Close'].rolling(window=rollingMeanWindowSize//8).mean()
	dfreg['AVG_rolling_half'] = df['Adj Close'].rolling(window=rollingMeanWindowSize//2).mean()
	dfreg['AVG_rolling_full'] = df['Adj Close'].rolling(window=rollingMeanWindowSize).mean()

	# Index by date
	dfreg.set_index('Date', inplace=True)

	# Drop the first $rollingMeanWindowSize$ dps as they won't have rolling averages.
	dfreg = dfreg[rollingMeanWindowSize:]
	# Sanitize
	dfreg.fillna(value=-99999, inplace=True)

	# Split dataset into a training set and one for inference
	forecast_length = 50 # Trading days.
	dfreg['label'] = dfreg['Adj Close'].shift(-forecast_length)
	X = np.array(dfreg.drop('label', 1))
	X = scale(X)

	X_train = X[:-forecast_length]
	Y_train = np.array(dfreg['label'])[:-forecast_length]

	X_pred = X[-forecast_length:]

	# Model initialiation and training.
	linreg = LinearRegression(n_jobs=-1)
	linreg.fit(X_train, Y_train)

	quadreg = make_pipeline(PolynomialFeatures(2), LassoLars(alpha=0.01, max_iter=10000))
	quadreg.fit(X_train, Y_train)

	bayesreg = make_pipeline(PolynomialFeatures(2), BayesianRidge(n_iter=10000, compute_score=True))
	bayesreg.fit(X_train, Y_train)

	# Inference
	lin_forecast = linreg.predict(X_pred)
	quad_forecast = quadreg.predict(X_pred)
	bayes_forecast = bayesreg.predict(X_pred)

	# Visualization via matplotlib
	dfreg['Lin Forecast'] = np.nan
	dfreg['Quad Forecast'] = np.nan
	dfreg['Bayes Forecast'] = np.nan

	# Insert predictions right after the last dp.
	last_date = dfreg.iloc[-1].name
	last_unix = last_date
	next_unix = last_unix + datetime.timedelta(days=1)
	for index in range(len(lin_forecast)):
		next_date = next_unix
		next_unix += datetime.timedelta(days=1)

		dfreg.loc[next_date, ['Lin Forecast', 'Quad Forecast', 'Bayes Forecast']] = [lin_forecast[index], quad_forecast[index], bayes_forecast[index]]

	# Plot the stock price, the rolling averages and the three forecasts.
	showN = 200
	dfreg['Adj Close'].tail(showN).plot()
	dfreg['Lin Forecast'].tail(showN).plot(style='--')
	dfreg['Quad Forecast'].tail(showN).plot(style='-')
	dfreg['Bayes Forecast'].tail(showN).plot(style='--')
	dfreg['AVG_rolling_eigth'].tail(showN).plot(style='--')
	dfreg['AVG_rolling_half'].tail(showN).plot(style='--')
	dfreg['AVG_rolling_full'].tail(showN).plot(style='-')
	plt.legend(loc=2)
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stock)
	plt.show()

if __name__ == '__main__':
	stocks = ['SAP.DE', 'AMD', 'GOOG']
	for stock in stocks:
		doAnalysis(stock)