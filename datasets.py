# Dataset entry

import os

from numpy.core.arrayprint import format_float_scientific
os.system("python libraries.py")
from libraries import *



def dataset_to_csv_1(ticker):

    dataset = yf.download(ticker,'2010-01-01', '2021-12-01')
    filename = '%s_datafile_1.csv' % (ticker)
    dataset['Date'] = dataset.index
    new_dataset = pd.DataFrame()
    new_dataset['Date'] = dataset['Date']
    new_dataset['Close'] = dataset['Close']
    for i in range(1,51):
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

    weights = np.arange(1,11)
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
    
    new_dataset['wr_14'] = get_wr(dataset['High'].shift(10), dataset['Low'].shift(10), dataset['Close'].shift(10), 10)

    new_dataset['CCI'] = CCI(dataset['Close'].shift(10), dataset['High'].shift(10), dataset['Low'].shift(10), 10, 0.015)

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

# For LSTM model
def create_dataset(in_data, days=1):
    X=[]
    Y=[]
    for i in range(len(in_data)-days-1):
        temp_X = in_data[i:i+days,0]
        temp_Y = in_data[i+days,0]

        X.append(temp_X)
        Y.append(temp_Y)
    return np.array(X), np.array(Y).reshape((-1,1))

def test_train_splitting_scaling(dataset,lstm_flag = False):
    dataset = dataset.dropna()
    X = dataset.drop(['Date','Close'], axis=1).to_numpy()
    # X = dataset.drop(['Date','Close'], axis=1)
    # print(X)
    # exit()
    Y = dataset['Close'].to_numpy()
    minmax = MinMaxScaler()
    if lstm_flag == True:
        minmax = MinMaxScaler(feature_range=(0,1))
        temp_dataset = pd.DataFrame()
        temp_dataset = dataset['Close'].reset_index(drop=True)
        close_dataset = pd.DataFrame()
        close_dataset = minmax.fit_transform(np.array(temp_dataset).reshape(-1,1))
        X,Y = create_dataset(close_dataset,days=50)
        # print(temp_dataset.head)
        # temp_dataset = temp_dataset.reset_index()
        # temp_dataset[temp_dataset.columns] = minmax.fit_transform(temp_dataset[temp_dataset.columns])
        # X = temp_dataset.drop(['Date','Close'], axis=1).to_numpy()
        # temp_dataset = minmax.fit_transform(temp_dataset)
        # X = minmax.fit_transform(X)
        print("--- Creatig datasets ---")
        # print(X.shape)
        # Y = dataset['Close'].to_numpy()
        # Y = minmax.transform(Y.reshape(-1,1))
        print(X.shape)
        print(Y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.05,random_state=None,shuffle=False)
        test_date = dataset.index[-1*len(y_test):]
        return minmax, X_train, X_test, y_train, y_test,test_date
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.05,random_state=None,shuffle=False)
    X_train = minmax.fit_transform(X_train,y_train)
    X_test = minmax.transform(X_test)
    test_date = dataset.index[-1*len(y_test):]

    return X_train, X_test, y_train, y_test,test_date

def get_Tensor_Dataloader(X_train, X_test, y_train, y_test):
    train_tensor = torch.tensor(X_train)
    test_tensor = torch.tensor(X_test)
    train_close_tensor = torch.tensor(y_train)
    test_close_tensor = torch.tensor(y_test)
    train_tensor_dataset = TensorDataset(train_tensor,train_close_tensor)
    test_tensor_dataset = TensorDataset(test_tensor,test_close_tensor)
    train_loader = DataLoader(train_tensor_dataset,batch_size=25,shuffle=False)
    test_loader = DataLoader(test_tensor_dataset,batch_size=25,shuffle=False)
    return train_loader, test_loader