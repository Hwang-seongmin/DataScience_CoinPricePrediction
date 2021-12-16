
# coding: utf-8

# In[1]:


# !pip install finance-datareader
# pip install Prophet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FinanaceDataReader를 통해주가 데이터를 가져옴
import FinanceDataReader as fdr

# 이더리움 코인
# 2020-01-01 부터의 자료를 가져옴
eth = fdr.DataReader('ETH/KRW', '2020-01-01')
eth


# In[3]:


# 페이스북 prophet import
from fbprophet import Prophet


# In[4]:


from fbprophet.plot import plot_plotly, plot_components_plotly


# In[5]:



# fbprophet 라이브러리를 사용하기 위해선 column 명을 y와 ds로 바꿔야함
eth['y'] = eth['Close']
eth['ds'] = eth.index

eth


# In[6]:


# prophet 객체 선언 및 학습
# changepoint_prior_scale이 클수록 모델은 과적합에 가까워짐
prophet = Prophet(seasonality_mode = 'multiplicative',
                 yearly_seasonality = True,
                 weekly_seasonality = True, daily_seasonality = True,
                 changepoint_prior_scale = 0.5)

prophet.fit(eth)


# In[7]:



# periods : 예측하고 싶은 기간
future = prophet.make_future_dataframe(periods=10)
future.tail(5)


# In[8]:


# ds: Date, yhat : 예측값, yhat_lower & yhat_upper : uncertainty intervals
forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)


# In[9]:


# 시각화
fig1 = prophet.plot(forecast)


# In[10]:


# 트렌드 정보 시각화
fig2 = prophet.plot_components(forecast)

