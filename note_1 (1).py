#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #data frame library.csv,excel file import by pandas.csv create by pandas.
import numpy as np #mathemathecal library.need for linear algebra,large array.
from matplotlib import pyplot as plt #data visualization libraray.use to draw plot,graph.


# In[4]:





# In[ ]:





# In[2]:


df = pd.read_excel("Rent.xlsx")


# In[3]:


###Data Analysis
df.head()


# In[4]:


df.tail()


# In[5]:


df.head(9)


# In[6]:


df.tail(9)


# In[7]:


df[5:11]


# In[8]:


df['rent'].head()


# In[9]:


df.rent.head()


# In[10]:


df['area'].tail()


# In[11]:


df.area.tail()


# In[12]:


df.mean()


# In[13]:


df['rent'].mean()


# In[14]:


df.area.mean()


# In[6]:


df.rent[5:11]


# In[9]:


df['rent'][5:11]        # df['rent'] = df.rent


# In[19]:


df.area[19:28]


# In[20]:


df.describe()


# In[21]:


df.area.describe()


# In[22]:


df['rent'].describe()


# In[23]:


df.area.std()


# In[24]:


df['rent'].min()


# In[26]:


df.shape


# In[27]:


row,col = df.shape


# In[28]:


row


# In[29]:


col


# In[30]:


df.isnull()


# In[31]:


df.isnull().sum()


# In[32]:


###Visualization
plt.figure(figsize=(12,8))


# In[33]:


plt.scatter(df.area, df.rent, marker="+", color="red")


# In[34]:


plt.figure(figsize=(12,8))
plt.scatter(df.area, df.rent, marker="+", color="red")


# In[35]:


plt.xlabel("Area", color="blue")


# In[36]:


plt.ylabel("Rent", color="blue")


# In[37]:


plt.title("Rent Plot", color="red")


# In[38]:


plt.figure(figsize=(12,8))
plt.scatter(df.area, df.rent, marker="+", color="red")
plt.xlabel("Area", color="blue")
plt.ylabel("Rent", color="blue")
plt.title("Rent Plot", color="red")


# In[39]:


plt.figure(figsize=(12,8))
plt.scatter(df.area, df.rent, marker=".", color="blue")
plt.xlabel("Area", color="blue")
plt.ylabel("Rent", color="blue")
plt.title("Rent Plot", color="red")


# In[40]:


plt.figure(figsize=(15,10))
plt.scatter(df.area,df.rent,marker="^",color="green")
plt.xlabel("Area",color="Red")
plt.ylabel("Rent",color="Red")
plt.title("Rent Plot",color="Red")


# In[41]:


x = df[['area']]   #independent variable x.dependent variable y.independent vaviable ke feature bole.feature ke amra single bracket er modde likhte pari na.so somoy double bracket er modde likhte hoi.jeta input oita holo feature.ar jeta predictkorbo oita holo label.Feature = input.Label = jeta ber korbo...


# In[42]:


x.head()


# In[43]:


y=df['rent']


# In[44]:


y.head()


# In[45]:


### ml a data ke 2 vage vag kora hoi...train data(70%) ar test data(30%).


# In[46]:


from sklearn.model_selection import train_test_split as tts


# In[53]:


xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.3,random_state=5) #random_satae dara data suffol ta control kora hoi.jar fole poti bar aki data test a ar aki data train a jai.


# In[54]:


xtrain.head()


# In[49]:


xtest.head()


# In[50]:


ytrain.head()


# In[51]:


ytest.head()


# In[52]:


### Linear Regression


# In[55]:


from sklearn.linear_model import LinearRegression 


# In[56]:


reg = LinearRegression()


# In[57]:


reg.fit(xtrain,ytrain) #data gulo ke train kora hoi fit function dara. linear regression algo tar sahajje machine ke train korailam fit dara.


# In[58]:


m = reg.coef_    #y= mx + c


# In[59]:


m


# In[60]:


c = reg.intercept_


# In[61]:


c


# In[62]:


#y = mx + c
y = m*3000 + c
y


# In[63]:


reg.predict([[3000]])


# In[64]:


reg.predict(xtest)


# In[65]:


pred = reg.predict(xtest)
pred


# In[66]:


df["Predicted Result"] = reg.predict(x) # new colom add


# In[67]:


df.head()


# In[68]:


from sklearn.metrics import mean_absolute_error, mean_squared_error #error ber kora....yt dia concept clear korte hbe error mane asole ki...


# In[69]:


mse = mean_squared_error(ytest,pred) # ytest = actual result(data),pred = predict resulr(ber kora data)


# In[70]:


mse


# In[71]:


mae = mean_absolute_error(ytest,pred)
mae


# In[72]:


### Best Fit Line


# In[73]:


plt.plot(xtrain,reg.predict(xtrain),color="red")  #best fit line sei sokol data point dia jai jekan dia gele error kom ase...


# In[74]:


plt.figure(figsize=(10,8))
plt.xlabel("Area")
plt.ylabel("Rent")
plt.title("Rent Plot")
plt.scatter(xtrain,ytrain)
plt.plot(xtrain,reg.predict(xtrain))
plt.scatter(xtrain,reg.predict(xtrain),color="red")


# In[75]:


from sklearn.metrics import r2_score  # accuracy by r2_score
r2_score(ytest,pred)


# In[ ]:





# In[76]:


xtest.to_csv("xtest.csv")   #data set create kora 


# In[77]:


xtest.head()


# In[78]:


len(xtest)


# In[79]:


### save Model


# In[80]:


import pickle as pk  #besi use hoi
import joblib as jb


# In[81]:


pk.dump(reg,open("first","wb")) #model ta reg er modde e ase. wb mane write mood a ...write binary


# In[82]:


reg2 = pk.load(open("first","rb")) # model load kora


# In[83]:


reg2.predict([[2000]])


# In[84]:


import warnings
warnings.filterwarnings("ignore")  # This code disables all warning messages in Python, often used in ML to keep output clean.


# In[85]:


jb.dump(reg, "second")  # dump function use for save model


# In[86]:


second.head()


# In[87]:


rr = jb.dump(reg, "second")  # dump function use for save model


# In[88]:


rr.head()


# In[89]:


rr.predict([[2000]])


# In[ ]:




