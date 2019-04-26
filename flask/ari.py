#Making series stationary,determining the values of p,d,q and creating the ACF and PACF plots.
import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima
import json
from json import dumps

data = pd.read_csv('p3.csv')
data = data.dropna(axis='columns', how='all')
data.loc[data.Type =='Error', 'Type'] = 1
data.loc[data.Type =='Warning', 'Type'] = 0
data.loc[data.Type =='Information', 'Type'] = 0
data.loc[(data.Type.str.contains('Exception'))==True, 'Type'] = 1
'''
data.loc[(data.Message.str.contains('DB error'))==True, 'Message'] = "DB Error"
data.loc[(data.Message.str.contains('APIs Error'))==True, 'Message'] = "APIs Error Request"
data.loc[(data.Message.str.contains('Shared repository'))==True, 'Message'] = "Shared repository"
'''
#print(data['Message'])
#data.loc[(data.Message.str.contains('Exception'))==True, 'Message'] = 1
data.loc[(data.Message.str.contains('DB error'))==True, 'Message'] = 2
data.loc[(data.Message.str.contains('APIs Error'))==True, 'Message'] = 3
data.loc[(data.Message.str.contains('Shared repository'))==True, 'Message'] = 1

#print(data['Message'])
data['_time']=pd.to_datetime(data['_time'], format="%Y/%m/%dT%H:%M:%S")
least_recent_date = data['_time'].min()
recent_date = data['_time'].max()
data['cycles']=(data['_time']-least_recent_date)
data['cycles']=(data['cycles'].astype('timedelta64[s]'))
data=data.sort_values(by='cycles') 
data=data[data.Type != 0]
dt1=data[data.Message==1]
dt2=data[data.Message==2]
dt3=data[data.Message==3]
#dt4=data[data['Message']==4]
print(len(dt1))
print(len(dt2))
print(len(dt3))
#print(len(dt4))


date_list=[]
src_list=[]

def predicting(data,pltname):
    data=data[['_time','cycles']]
    original=data['cycles']
    n=len(data)
    #print("length of datta",n)
    forecast_out=int(math.ceil(0.2*(n)))
    #print("forecat_out",forecast_out)
    data['label']=data['cycles'].shift(-forecast_out)
    data.dropna(inplace=True)
    original=data['cycles']
    original=original.to_frame(name='cycles')
    d1=data['label'] #this step changesa dataframe object into that of a series.series object
    d1=d1.to_frame(name='label') #thus need to convert it back into a dataframe object
    data=data['label']
    data=data.to_frame(name='label')
    #print("last value maam",data.iat[len(data)-1,0])

    #divide into train and validation set
    #train = data[:int(0.8*(len(data)))]
    #test = data[int(0.8*(len(data))):]

    model = auto_arima(data, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
    max_p=3, max_q=3, max_P=3, max_Q=3, seasonal=True,
    stepwise=False, suppress_warnings=True, D=1, max_D=10,
    error_action='ignore',approximation = False)
    #change 3 to 10
    #fitting model
    model.fit(data)
    #print(model.summary())
    y_pred = model.predict(n_periods=forecast_out)
    y_pred = pd.DataFrame(y_pred,columns=['label'])
    #print("first element",y_pred.iat[0,0])
    conn = pd.concat([d1, y_pred], axis=0)
    n=conn.size-forecast_out
    plt.figure(0)
    plt.plot(original[:n],conn[:n],'y')
    diff=original.iat[len(original)-1,0]-original.iat[len(original)-forecast_out,0]
    plt.plot(original[-forecast_out:]+diff,conn[-forecast_out:],'r')
    plt.savefig("/Users/Arunima_Sharma/Desktop/py/flask/static/"+pltname)
    plt.show()
    return y_pred


y_pred1=predicting(dt1,'fig1.png')
y_pred2=predicting(dt2,'fig2.png')
y_pred3=predicting(dt3,'fig3.png')
#y_pred_train=predicting(train,figtr.png)
#y_pred_total=predicting(data,figto.png)

#plt.savefig("/Users/Arunima_Sharma/Desktop/py/flask/static/fig3.png")

def stringing(y_pred,name):
    for i in range(len(y_pred)):
        sec=y_pred.iat[i,0]
        x=least_recent_date+datetime.timedelta(seconds=sec)
    #date_list=[]
        if(max(x,recent_date)==x):
            date_list.append(x)
            src_list.append(name)

stringing(y_pred1,"Gcm notification Exception")
stringing(y_pred2,"DB Error")
stringing(y_pred3,'APIs Error Request')
#stringing(y_pred3,'Shared repository')
print(y_pred3)

date_result=pd.DataFrame()
#print(date_result)

date_result['Predicted Date'] = pd.DataFrame(date_list)
#date_result['Date'] =pd.to_datetime(date_result.Date)
date_result['Type of Error'] = pd.DataFrame(src_list)
date_result=date_result.sort_values(by=["Predicted Date"])
#print(date_result)
date_result = date_result.astype(str)
date_result.to_csv('result.csv', encoding='utf-8', index=False)