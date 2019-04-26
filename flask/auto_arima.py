 #Before implementing ARIMA, you need to make the series stationary, and determine the values of p and q 
 #using the plots we discussed above. Auto ARIMA makes this task really simple for us as it eliminates 
#Making series stationary,determining the values of p,d,q and creating the ACF and PACF plots.
import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima
import json
from json import dumps

data = pd.read_csv('p2.csv')
data=data[['_time','Type','Message']]
data.loc[data.Type =='Error', 'Type'] = 1
data.loc[data.Type =='Warning', 'Type'] = 0
data.loc[data.Type =='Information', 'Type'] = 0
data.loc[(data.Type.str.contains('Exception'))==True, 'Type'] = 1

data.loc[(data.Message.str.contains('DB error'))==True, 'Message'] = 1
data.loc[(data.Message.str.contains('APIs Error'))==True, 'Message'] = 2
data.loc[(data.Message.str.contains('Shared repository'))==True, 'Message'] = 3

data['_time']=pd.to_datetime(data['_time'], format="%Y/%m/%dT%H:%M:%S")
least_recent_date = data['_time'].min()
recent_date = data['_time'].max()
data['cycles']=(data['_time']-least_recent_date)
data['cycles']=(data['cycles'].astype('timedelta64[s]'))
data=data.sort_values(by='cycles') 
#data=data[data.Type != 0]
data=data[data.Message==3]
data=data[['_time','cycles']]
original=data['cycles']
n=len(data)
print(n)
forecast_out=int(math.ceil(0.1*(n)))
print(forecast_out)
data['label']=data['cycles'].shift(-forecast_out)
print("length of data is",len(data))
data.dropna(inplace=True)
original=data['cycles']
original=original.to_frame()
print("length of data is after dropping ",len(data))


d1=data['label'] #this step changesa dataframe object into that of a series.series object
d1=d1.to_frame(name='label') #thus need to convert it back into a dataframe object
print(type(d1))
data=data['label']
print(len(d1))
data=data.to_frame(name='label')
print(type(data))
print(data.iat[len(data)-1,0])

#divide into train and validation set
train = data[:int(0.8*(len(data)))]
test = data[int(0.8*(len(data))):]

model = auto_arima(train, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=3, max_q=3, max_P=3, max_Q=3, seasonal=True,
                  stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = False)
                  #change 3 to 10
#fitting model
model.fit(train)
print(model.summary())
#the one with the lower AIC is generally “better”. 
y_pred = model.predict(n_periods=len(test))
#y_pred = model.predict(n_periods=len(test))
y_pred = pd.DataFrame(y_pred,columns=['label'])

plt.figure(1)
plt.plot(original[:len(train)],train)
plt.plot(original[-len(test):],test,'b')
plt.plot(original[-len(y_pred):],y_pred,'r')
#plt.savefig("/Users/Arunima_Sharma/Desktop/py/flask/static/fig1.png")
plt.show()
plt.figure(2)
plt.plot(original[-len(test):],test,'b')
plt.plot(original[-len(y_pred):],y_pred,'r')
plt.show()
#plt.savefig("/Users/Arunima_Sharma/Desktop/py/flask/static/fig2.png")

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
acc = r2_score(test, y_pred)
print(acc)
mse = mean_squared_error(test, y_pred)
print('MSE: %f' % mse)
rmse = math.sqrt(mse)
print('RMSE: %f' % rmse)
mae = mean_absolute_error(test, y_pred)
print('MAE: %f' % mae)

model = auto_arima(data, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=3, max_q=3, max_P=3, max_Q=3, seasonal=True,
                  stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = False)
                  #change 3 to 10
#fitting model
model.fit(data)
print(model.summary())
y_pred = model.predict(n_periods=forecast_out)
y_pred = pd.DataFrame(y_pred,columns=['label'])
conn = pd.concat([d1, y_pred], axis=0)
n=conn.size-forecast_out
plt.figure(0)
plt.plot(original[:n],conn[:n],'y')
diff=original.iat[len(original)-1,0]-original.iat[len(original)-forecast_out,0]
plt.plot(original[-forecast_out:]+diff,conn[-forecast_out:],'r')
plt.show()
#plt.savefig("/Users/Arunima_Sharma/Desktop/py/flask/static/fig3.png")
print(type(least_recent_date))

print("recent_date is",recent_date)
date_list=[]
for i in range(len(y_pred)):
    sec=y_pred.iat[i,0]
    x=least_recent_date+datetime.timedelta(seconds=sec)
    if(max(x,recent_date)==x):
            {
                #print(x)
                date_list.append(x)
            }

date_result = pd.DataFrame(date_list)
date_result = date_result.astype(str)
#d1=data_result.to_csv('result.csv', encoding='utf-8', index=False)
date_result.to_csv('result.csv', encoding='utf-8', index=False)
#date_result = date_result.to_json()
#with open('final.json', 'w') as outfile:
#    json.dump(date_result, outfile)
