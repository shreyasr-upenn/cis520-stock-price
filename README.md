### CIS 520 Final Project 

# Stock Price Prediction using Machine Learning Algorithms

## Authors
[Shreyas Ramesh](mailto:shreyasr@seas.upenn.edu)\
[Adwayt Nadakarni](mailto:adwayt@seas.upenn.edu)\
[Debadeepta Tagore](mailto:tagore@seas.upenn.edu)\
\
University of Pennsylvania School of Engineering and Applied Science

## Abstract
Stock or equity market is the aggregation of buyers and sellers of stock, representing the ownership claims in the business listed on the market. The high returns earned from bullish trends attract new investors into investing in the stock market despite the market risks; hence a machine learning model that forecasts the trend of a share is crucial to protect the investors, aid the investors in maximizing profit and predict a fall in the market before it actually happens. In this project, we extract open-source data of three different companies - Apple Inc., Tesla Inc., and JPMorgan & Co., from different sectors using Yahoo! Finance, and use this data to train, evaluate and compare multiple machine learning models in the hopes of finding a model that can best help the investors. As the Open, Close, High, and Low values of the share are highly correlated with each other, the trained model will have low bias (or high variance) which will lead to over-fitting. Hence, we take two approaches to train our model. The first approach is to drop all features from the extracted data and use the present Close values along with Close values shifted over a period of time to predict the future Close values. The second approach is to retain independent features such as date and volume along with technical indices to predict the future Close values. The tabulated results were compared with Auto ML and a final conclusion was recorded. The results showed that Lasso Regression performed the best for Apple Inc. using the second approach, Multiple Linear Regression performed the best for Tesla Inc. using the second approach and as per Auto-ML, a cross-validated Lasso using the LARS algorithm performed best for JPMorgan & Co. using the second approach. The neural network model provided the best overall prediction for all three shares using the first approach.



## License
[MIT](https://choosealicense.com/licenses/mit/)