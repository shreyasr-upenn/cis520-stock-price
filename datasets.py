# Dataset entry

import os
os.system("python libraries.py")
from libraries import *



def dataset_to_csv_1(ticker):

    dataset = yf.download(ticker,'2010-01-01', '2021-12-01')
    filename = '%s_datafile_1.csv' % (ticker)
    new_dataset = pd.DataFrame()
    new_dataset['Close'] = dataset['Close']
    for i in range(50):
        column_name = 'Shift_%d' % (int(i))
        new_dataset[column_name] = dataset['Close'].shift(int(i))
    new_dataset.dropna()
    temp = new_dataset
    new_dataset.to_csv(filename)
    return temp


def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max() 
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr

def CCI(close, high, low, n, constant): 
    TP = (high + low + close) / 3 
    CCI = pd.Series((TP - TP.rolling(n).mean()) / (constant * TP.rolling(n).std()), name = 'CCI_' + str(n)) 
    return CCI


def dataset_to_csv_2(ticker):
    dataset = yf.download(ticker,'2010-01-01', '2021-12-01')
    filename = '%s_datafile_2.csv' % (ticker)
    
    new_dataset = pd.DataFrame()
    new_dataset['Date'] = dataset['Date'] 
    
    new_dataset['Close'] = dataset['Close']
    
    new_dataset['SMA_10'] = dataset['Close'].rolling(window=10).mean()

    new_dataset['WMA_10'] = dataset['Close'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    
    momentum = dataset['Close'].shift().rolling(10).apply(lambda prices: prices[9] - prices[0], raw=True)
    new_dataset['Momentum_10'] = momentum

    high_10 = dataset['High'].shift().rolling(10).max()
    low_10 = dataset['Low'].shift().rolling(10).min()

    new_dataset['%K'] = (dataset['Close'] - low_10)*100/(high_10 - low_10)
    new_dataset['%D'] = new_dataset['%K'].rolling(window=5).mean()

    difference = dataset['Close'].diff()
    up = difference.clip(lower=0)
    down = -1*difference.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()  # Exponential weighted mean, com - Specify decay in terms of center of mass
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down

    new_dataset['RSI'] = 100 - (100/(1 + rs))

    exp1 = dataset["Close"].ewm(span=12, adjust=False).mean()
    exp2 = dataset["Close"].ewm(span=26, adjust=False).mean()
    macd = exp1-exp2


    new_dataset['MACD'] = (macd) - ( macd.ewm(span=9, adjust=False).mean() )
    
    new_dataset['wr_14'] = get_wr(dataset['High'], dataset['Low'], dataset['Close'], 10)

    new_dataset['CCI'] = CCI(dataset['Close'], dataset['High'], dataset['Low'], 10, 0.015)

    new_dataset.dropna()
    temp = new_dataset
    new_dataset.to_csv(filename)
    return temp



def read_from_csv(ticker):
    return 0

def plot_datasets(dataset):
    dummy = dataset
    # dummy['Close'] = jpm_dataset['Close']

    dummy.iloc[:,1:].plot(kind='line', figsize=(20,20), subplots= True)

    plt.figure(figsize = (16,5))
    sns.heatmap(dummy.corr(),annot=True)
    plt.title('Correlation between the generated features and closing price between JPMORGAN CHASE & CO.')
    # Plots the graphs

