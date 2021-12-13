# Models
from libraries import *
class Apple:
    def __init__(self):
        self.model=[]
        self.MSE = []
        self.MAE = []
        self.R2 = []
    
    def metrics(self, model,mse,mae,r2):
        self.model.append(model)
        self.MSE.append(mse)
        self.MAE.append(mae)
        self.R2.append(r2)

class Tesla:
    def __init__(self):
        self.model=[]
        self.MSE = []
        self.MAE = []
        self.R2 = []
    
    def metrics(self, model,mse,mae,r2):
        self.model.append(model)
        self.MSE.append(mse)
        self.MAE.append(mae)
        self.R2.append(r2)

class JPM:
    def __init__(self):
        self.model=[]
        self.MSE = []
        self.MAE = []
        self.R2 = []
    
    def metrics(self, model,mse,mae,r2):
        self.model.append(model)
        self.MSE.append(mse)
        self.MAE.append(mae)
        self.R2.append(r2)

apple = Apple()
tesla = Tesla()
jpm = JPM()


class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.linear1 = nn.Linear(in_features=50, out_features= 75, bias=True)
        self.linear2 = nn.Linear(in_features=75,out_features=100,bias=True)
        self.linear3 = nn.Linear(in_features=100, out_features=100, bias=True)
        self.linear4 = nn.Linear(in_features=100, out_features=10, bias=True)
        self.linear5 = nn.Linear(in_features=10, out_features=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self,x):
        # x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        # x = self.relu(x)
        return x




def plot_model(model, X_test, y_test,test_date):
    plt.figure(figsize=(10,5))
    plt.plot(test_date, y_test, color = 'r', label = 'Actual Value registered as per Yahoo! Finance')
    plt.plot(test_date, model.predict(X_test), color = 'g', label = 'Values Predicted by Model')
    plt.xlabel('')
    plt.ylabel('The Price of the Stock in dollars')
    plt.legend()
    plt.show()

def compute_metrics(y_test, y_pred):
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    RMSE = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    print('Mean Absolute Error: ', MAE)
    print('Root Mean Squared Error: ', RMSE )
    print('R Squared Score: ', R2)

    return MAE, RMSE, R2

def LinearRegressionModel(ticker,X_train, y_train, X_test, y_test,test_date):
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    plot_model(lr, X_test,y_test,test_date)
    # print(lr.predict(X_test))
    MAE, RMSE, R2 = compute_metrics(y_test, lr.predict(X_test))

    if ticker == 'AAPL':
        apple.metrics('LinearReg', RMSE, MAE, R2)
    elif ticker == 'TSLA':
        tesla.metrics('LinearReg', RMSE, MAE, R2)
    elif ticker == 'JPM':
        jpm.metrics('LinearReg', RMSE, MAE, R2)

def RFRModel(ticker,X_train, y_train, X_test, y_test,test_date):
    for i in [5,10,20,50,100]:
        print(f'--------------- n_estimator = {i}------------')
        rfr = RandomForestRegressor(n_estimators=i, random_state=0,)
        rfr.fit(X_train,y_train)
        plot_model(rfr,X_test,y_test,test_date)
        MAE, RMSE, R2 = compute_metrics(y_test, rfr.predict(X_test))

        if ticker == 'AAPL':
            apple.metrics('RFR', RMSE, MAE, R2)
        elif ticker == 'TSLA':
            tesla.metrics('RFR', RMSE, MAE, R2)
        elif ticker == 'JPM':
            jpm.metrics('RFR', RMSE, MAE, R2)

def LassoModel(ticker,X_train, y_train, X_test, y_test,test_date):
    for i in [1,0.1,0.01,0.001]:
        print(f'--------------- alpha = {i}------------')
        lass = Lasso(alpha=i)
        lass.fit(X_train,y_train)
        plot_model(lass,X_test,y_test,test_date)
        MAE, RMSE, R2 = compute_metrics(y_test, lass.predict(X_test))

        if ticker == 'AAPL':
            apple.metrics('Lasso', RMSE, MAE, R2)
        elif ticker == 'TSLA':
            tesla.metrics('Lasso', RMSE, MAE, R2)
        elif ticker == 'JPM':
            jpm.metrics('Lasso', RMSE, MAE, R2)



def NeuralNetwork(ticker,X_train, y_train, X_test, y_test,test_date):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_tensor = torch.tensor(X_train)
    test_tensor = torch.tensor(X_test)
    train_close_tensor = torch.tensor(y_train.reshape(-1,1))
    test_close_tensor = torch.tensor(y_test.reshape(-1,1))
    train_tensor_dataset = TensorDataset(train_tensor,train_close_tensor)
    test_tensor_dataset = TensorDataset(test_tensor,test_close_tensor)
    train_loader = DataLoader(train_tensor_dataset,batch_size=25,shuffle=False)
    test_loader = DataLoader(test_tensor_dataset,batch_size=25,shuffle=False)
    
    nn_model = ANN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr = 1e-3)

    overall_step = 0

    epochs = 100

    train_loss = []
    val_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0
        acc = 0
        total = 0
        num_batches = 0
        nn_model.train()
        for _,(X,y) in enumerate(train_loader):
                    
            optimizer.zero_grad()
            X = X.float()
            # print(X)
            y_pred = nn_model.forward(X)
            # print(y_pred)
            y = y.float()
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            #Backprop
            loss.backward()
            optimizer.step()
            num_batches += 1
        
        running_loss = running_loss / num_batches
        train_loss.append(running_loss)
        
        running_val_loss = 0 
        num_batches = 0
        nn_model.eval()
        for _,(X,y) in enumerate(test_loader):
            X = X.float()
            y_pred = nn_model(X).to(device)
            y = y.float()
            loss = criterion(y_pred, y)
            running_val_loss += loss.item()
            num_batches += 1

        running_val_loss = running_val_loss/num_batches
        val_loss.append(running_val_loss)

        # print("#################################################################################")
        print(f'epoch : {epoch} train_loss : {train_loss[epoch]}, val_loss: {val_loss[epoch]}')
        # print("#################################################################################")
    
    
    plt.plot(test_date,y_test,color = 'r', label = 'Actual Value registered as per Yahoo! Finance')
    y_pred = nn_model(test_tensor.float())
    plt.plot(test_date,y_pred.detach().numpy(), color = 'g', label = 'Values Predicted by Model')
    plt.xlabel('')
    plt.ylabel('The Price of the Stock in dollars')
    plt.legend()
    plt.show()
    MAE, RMSE, R2 = compute_metrics(y_test,y_pred.detach().numpy())
    
    if ticker == 'AAPL':
        apple.metrics('NN', RMSE, MAE, R2)
    elif ticker == 'TSLA':
        tesla.metrics('NN', RMSE, MAE, R2)
    elif ticker == 'JPM':
        jpm.metrics('NN', RMSE, MAE, R2)


def LSTM_Model (ticker, minmax, X_train, y_train, X_test, y_test,test_date):

    # Define Model

    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(50,1)))
    # model.add(Dropout(0.2))
    model.add(LSTM(50,return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.summary()

    model.reset_states()
    model.reset_metrics()
    history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)
    y_test_pred = model.predict(X_test)
    
    # print(y_test_pred.shape)
    y_test_pred = minmax.inverse_transform(y_test_pred.reshape((y_test_pred.shape[0],y_test_pred.shape[1])))

    plt.plot(test_date,minmax.inverse_transform(y_test),color='r', label='Actual')
    plt.plot(test_date, y_test_pred,color = 'g', label = 'Prediction')
    plt.ylabel('Price in USD')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs Test loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    
    MAE, RMSE, R2 = compute_metrics(minmax.inverse_transform(y_test),y_test_pred)
    
    if ticker == 'AAPL':
        apple.metrics('LSTM', RMSE, MAE, R2)
    elif ticker == 'TSLA':
        tesla.metrics('LSTM', RMSE, MAE, R2)
    elif ticker == 'JPM':
        jpm.metrics('LSTM', RMSE, MAE, R2)


def TPOT_model(ticker,X_train, y_train, X_test, y_test,test_date):
    tpot = TPOTRegressor(generations= 10, population_size = 50, verbosity = 2)

    tpot.fit(X_train,y_train)
    print(tpot.score(X_test,y_test))

    plot_model(tpot, X_test,y_test,test_date)
    # print(lr.predict(X_test))
    
    MAE, RMSE, R2 = compute_metrics(y_test, tpot.predict(X_test))

    if ticker == 'AAPL':
        apple.metrics('AutoML', RMSE, MAE, R2)
    elif ticker == 'TSLA':
        tesla.metrics('AutoML', RMSE, MAE, R2)
    elif ticker == 'JPM':
        jpm.metrics('AutoML', RMSE, MAE, R2)

