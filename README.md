# DS301_Final
This repo contains the final project code for DS301 in spring 2024 in NYU
# Project Title
NVIDIA Prediction with LSTM and Various Hybrid Models
# Project Member
Jiewu Rao, Jordon Xue, and Jeremy Zhu
# Project Description and Motivation
Our group members are all finance related background and all know the importance of stock prediction and accurate valuation in financial anaylsis. We note that many popualr stock prediction models are basing on LSTM. Meanwhile, we are inspired by research done on integrating LSTM with other models. In this project, we are interested in exploring accurate prediction of stock price by building up hybrid models and compare them with base LSTM. Stocks’ volatility is usually high across time, so we want to create a better-predicting model to capture the variation. We believe that LSTM will be a good base model that fit for stock prediction as LSTMs are designed to remember and utilize important information over long sequences of time and therefore can capture the historical relevance of stocks. 

However, other models like GRU, CNN, BiLSTM, FasterRNN, SVM, etc. may present different strengths. They might not look directly as useful as LSTM does in stock prediction. However, we assume by integrating model together, the hybrid model can leverage the strength of all methods, obtain highly expressive features, and efficiently learn trends and interactions of multivariate time series, which therefore improves the base LSTM model that many people adopt. 

This reports will tune, implement, examine, and compare across baseline LSTM, CNN-GRU, GRU-CNN-LSTM, BiLSTM-GRU-DenseNetwork, FasterRNN-CNN-LSTM, CNN-LSTM and BiLSTM. We also experiment with GRU-CNN-LSTM, BiLSTM-GRU-DenseNetwork, GRU-LSTM-Attention layers, DenseNetwork-Conv1D-LSTM, TensorFlow-LSTM models, and SVM-LogisticRegression, although their performances are not comparably good enough. To combat overfitting and enhance generalization, we employed dropout strategies, EarlyStopping to halt training when validation loss ceased improving, and ReduceLROnPlateau to adjust learning rates. The stock we predict is NVIDIA as it experience both stable and volatile period during the past 14 years (period we adopt). Feature stocks are upstream and downstream stakeholder companies of NVIDIA. The evaluation metrics we adopt is RMSE.

# Repository and Code Structure
For this repository, we include a .ipynb file "301_Project.ipynb" for all the coding and graph we have. The file contains all the model we experimented with and contain all analysis and preprocessing we've done. We implemented the graph within the .ipynb file as well.

Data imported are from Yahoo Finance. We imported 'AMZN', 'GOOG', 'ON', 'FTNT', 'IBM', 'CSCO', and 'NVDA' real stock price from 2010.01.01 to 2024.01.01. While NVIDIA is the target, other stocks are upstream or downstream stakeholders companies that work as feature in this task. Then we build up and tune our base LSTM model first. Followed by that, we experimented with  CNN-GRU, GRU-CNN-LSTM, BiLSTM-GRU-DenseNetwork, FasterRNN-CNN-LSTM, CNN-LSTM and BiLSTM, GRU-CNN-LSTM, BiLSTM-GRU-DenseNetwork, GRU-LSTM-Attention layers, DenseNetwork-Conv1D-LSTM, TensorFlow-LSTM models, and SVM-LogisticRegression models. We tune the models with the same technique and tested on the same data. To handle overfitting and enhance generalization, we employed dropout strategies, EarlyStopping to halt training when validation loss ceased improving, and ReduceLROnPlateau to adjust learning rates. We aim to compare the best performance of all those models. We also plotted the predicted price in comparison the actual price to visualize model performance.

# Example commands to execute the code

# Results
