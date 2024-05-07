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

The LSTM-GRU model emerged as the best performing model with the lowest RMSE of 8.149. This indicats that the hybrid model effectively captures both short-term and long-term dependencies in the dataset. The superior performance of the model can be attributed to the synergistic integration of LSTM and GRU layers. LSTMs excel in retaining long-term historical data, crucial for recognizing underlying patterns and trends that influence future prices over extended periods. On the other hand, GRUs provide the model with the flexibility to quickly adapt to new, short-term market dynamics and anomalies, ensuring responsiveness to immediate financial events without the computational complexity of traditional LSTMs. This combination enables a robust predictive model that is both deep in historical insight and agile in updating with new information, leading to high accuracy in predicting volatile context of stock market predictions.

The BiLSTM and FasterRNN-CNN-BiLSTM models outperform the base LSTM due to their enhanced handling of complex data dependencies as well. The BiLSTM utilizes an extended look-back mechanism to capture long-term temporal patterns and trends more effectively, which is critical in stock price forecasting. The FasterRNN-CNN-BiLSTM combines RNNs, CNNs, and BiLSTMs to efficiently process sequences, extract spatial features, and understand both past and future contexts simultaneously, which enables a deeper and more comprehensive analysis of the data. Therefore, these models particularly adept at navigating the complexities of stock market data.

On ther other hand, the comparative underperformance (compared with base LSTM) of other models, such as the CNN-LSTM or CNN-GRU models, underscores the complexity of the prediction task. The CNN-GRU model displayed higher errors with an RMSE of 12.325, and the CNN-LSTM is with a RMSE of 12.100. It suggests that combining convolutional layers with recurrent units might not be as effective in this scenario. The convolutional layers, although proficient at extracting spatial features, may not align well with the sequential nature of stock price data, which is heavily dependent on temporal relationships. These models may suffer from an imbalance in effectively handling the temporal dynamics after spatial processing, potentially leading to a loss of critical time-series information.

# Conclusions and Future Works
Our project has extensively explored various hybrid machine learning models to predict NVIDIA's stock prices, based on the historical stock prices of multiple major companies. We aim to find the best models by comparing it with the base LSTM model (the most popular model in stock prediction task). We focused on LSTM-based models, enhancing them with other neural network architectures like GRU and CNN to capture both temporal and spatial patterns in stock market data effectively. We also experimented with and tuned many other models we found.

The major strength of our approach lies in its innovative integration of different neural network types to tackle the complex nature of financial time series data. We find out that integrating different models can leverage the strengths of each architecture. Our best model, combining LSTM and GRU, effectively utilized the strengths of both architectures, providing superior performance in capturing the long-term and short-term dependencies necessary for accurate stock price predictions. 

One significant limitation was the inherent volatility and unpredictability of the stock market, which often introduces noise that can affect model performance. Additionally, our models primarily relied on historical stock prices, excluding other potentially influential factors like market sentiment or macroeconomic indicators. Furthermore, the introduction of an attention layer in the GRU + LSTM architecture did not yield the expected improvements, largely because attention mechanisms are better suited for tasks with long-range dependencies, unlike the patterns typically found in stock price movements. Moreover, the complexity added by attention layers can increase sensitivity to noise. It is important to note that our model is designed to predict current stock prices based on existing data and does not forecast future stock price movements.

Future work  lies on 1) Enhancing the model with a broader set of features, including macroeconomic indicators, market sentiment data, and technical indicators, to capture a more comprehensive array of factors influencing stock prices; 2) Developing real-time data analysis capabilities to adapt the models for use in dynamic trading environments, thereby increasing their practical utility in real-world scenarios; 3) Expanding the dataset to include a wider variety of stocks from multiple sectors and geographies to enhance the robustness and generalizability of the models; 4) Investigating the effects of significant market movements, such as large-scale buys and sells, on stock prices to enhance predictive accuracy under volatile conditions.
