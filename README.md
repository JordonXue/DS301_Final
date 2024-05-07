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
Our python code is initiated in Google Colab(cloud based). In order to execute the code, it's suggested to download the file， import and run them on Google Colab in order to ensure all models functions without any errors. 

# Results
To correlation between feature and target:
![image](https://github.com/JordonXue/DS301_Final/assets/118228743/99ad8d19-7544-4154-9c58-af253fe0baf6)

The best performance of models:
<img width="297" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/6923861d-f5af-4434-a084-9165f95f3173">

Some other models we tested:
SVM RMSE: 7.774468191169713 (although high, may be overfitting)
GRU-LSTM-CNN RMSE: 37.52522609638047
Bidirectional LSTM (BiLSTM), GRU, and a simple Dense network RMSE: 33.49966430726417
GRU-LSTM Attention Layer RMSE: 57.31670565826679
Dense network, Conv1D, and LSTM RMSE: 55.26480947035524
Transformer Layer with LSTM RMSE: 47.75423378382045

Visualization of performance:
<img width="416" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/970c1f63-dabb-4221-88b2-f76d2d7009e6">
<img width="413" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/a71d6fb3-2985-42cb-b561-f784dac01cac">
<img width="382" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/b1ebd0a5-e035-42e0-a573-0ed789b1e950">
<img width="384" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/1d021d5d-c102-4825-a06a-8402045862cf">
<img width="383" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/a8c72b04-82f0-4e74-8175-15fdf56ee487">
<img width="380" alt="image" src="https://github.com/JordonXue/DS301_Final/assets/118228743/f07b586a-5f0a-49f4-a9a4-6459731ce576">






