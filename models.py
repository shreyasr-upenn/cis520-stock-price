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

def plot_model(model, X_test, y_test,test_date):
    plt.figure(figsize=(10,5))
    plt.plot(y_test, color = 'r', label = 'Actual Value registered in NASDAQ as per Yahoo! Finance')
    plt.plot(model.predict(X_test), color = 'g', label = 'Values Predicted by Model')
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
    return 0
    
    


