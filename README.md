# Final-project-


The main objective of the project is to predict the abnormal heartbeat from the time series ECG signal using the concept of the autoencoder. With the help of an autoencoder in LSTM models in deep learning , the anomaly in the heart signals can be predicted accurately in this project. It can effectively reduce the false alarm rate.The accuracy of the anomaly detection method directly reflects the result of the cardiac disease detection 



# DATASET
http://timeseriesclassification.com/description.php?Dataset=ECG5000![image](https://user-images.githubusercontent.com/61631098/113885211-4a986500-97dd-11eb-888f-a54b2d84c079.png)

# Language 

Python

## Libraries used

- Pytorch
- Streamlit
- Matplotlib
- Seaborn
- Pandas
- Numpy
- Plotly

# Algorithm

![image](https://user-images.githubusercontent.com/61631098/113886197-1b362800-97de-11eb-82cf-25e2eff6433d.png)


- Using LSTM encoder to encode the data.
- Getting the compressed data from encoder .
- Using LSTM decoder to decode the compressed data.
- This will give the general learned representation from the data used.
- This forms the auto encoder model.
- The model can be used to predict the patterns of future data.
- The data can be passed to the model for checking normality / anomality.

# why LSTM ?

![image](https://user-images.githubusercontent.com/61631098/113886019-f80b7880-97dd-11eb-99d5-9fa5b0d2ea3e.png)


- LSTM introduces long-term memory into recurrent neural networks.

- It mitigates the vanishing gradient problem

- It is designed to handle sequence dependence

-LSTM networks have memory blocks that are connected through layers. 


# Normal ECG signal:

![image](https://user-images.githubusercontent.com/61631098/113880896-87625d00-97d9-11eb-9cb9-4f0971fe3da5.png)

# Abnormal ECG signal:

![image](https://user-images.githubusercontent.com/61631098/113880983-9f39e100-97d9-11eb-9f88-4b108fa50128.png)


# NEED OF AUTOENCODER

•	The aim of an auto-encoder is to learn a compressed, distributed representation (encoding) for a set of data.

•	Auto encoding is useful in the sense that it allows us to compress the data in an optimal way.

•	It can be used to represent the input data, as observed by the decoding layer.


![image](https://user-images.githubusercontent.com/61631098/113881330-f5a71f80-97d9-11eb-9588-64fb6fcdd825.png)


# SCREENSHOTS 


![image](https://user-images.githubusercontent.com/61631098/113881467-153e4800-97da-11eb-8d7c-fdf4f3c45c58.png)

![image](https://user-images.githubusercontent.com/61631098/113881499-1bccbf80-97da-11eb-833d-18a7621cb9a3.png)

![image](https://user-images.githubusercontent.com/61631098/113881530-2424fa80-97da-11eb-966f-97cba3208657.png)

![image](https://user-images.githubusercontent.com/61631098/113881563-2ab37200-97da-11eb-90e6-40bce4839b5b.png)

