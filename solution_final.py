import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import matplotlib.pyplot as mt


folder=os.listdir('CSV')

#Preparing the training set
train_X=pd.read_csv("CSV/1.csv")
train_X=train_X['Power'][:500]
train_X=np.array(train_X)

for csv in folder:
    df = pd.read_csv('CSV\\'+csv,delimiter=(","))
    df=df['Power'][:500]
    df=np.array(df)
    train_X=np.vstack([train_X,df])

train_y=pd.read_csv('train_y.csv')
train_y=train_y['Result'][:34]
train_y=np.array(train_y)

#Preparing test set
test_file=input('Enter the name of the file: ')
test=pd.read_csv('Test/'+test_file)
test=test['Power'][:500]
test=np.array(test)

#Building ML model
model=LogisticRegression(solver='lbfgs',max_iter=1000)
model.fit(train_X, train_y)

#Predicting results
result=model.predict(test.reshape(1,-1))
score=model.score(train_X,train_y)

#Function to return process name
def process_name(result):
    if result==0:
        return("Single Plunge")
    elif result==1:
        return("Face-Multi Plunge")
    elif result==2:
        return("Face-Single Plunge")
    
print('The process is ',process_name(result),'grinding.')
print('The accuracy of the model is ',score*100.00)



#Plotting the graph
mt.plot(test)
mt.title(process_name(result))
mt.ylabel("Power")
mt.xlabel("Samples")


