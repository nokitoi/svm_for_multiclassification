#STUDENT: 3820211081 农安
#LECTURER: Li Kan
#COURSE: Machine Learning and Knowledge Discovery
#HOMEWORK: SVM for multi-classification

from sklearn import datasets
import numpy as np
from sklearn.svm import SVC

def accuracy(actual,predicted):
  c = 0
  for i in range(len(actual)):
      if(actual[i]!=predicted[i]):
          continue
      else:    
          c = c + 1
  return c/len(actual)


# SVM Class defined by user 

class SVM():

  def __init__(self,C=1,kernel='linear',method='binary',max_iter=-1,g=0.3):
    self.kernel = kernel
    self.method = method
    self.C = C
    self.max_iter = max_iter
    self.g = g
    
  def Fit(self,XTrain,YTrain):
    
    if self.method == 'binary':    
      s = SVC(C = self.C ,kernel = self.kernel ,max_iter = self.max_iter , random_state=10,gamma = self.g)
      s.fit(XTrain,YTrain)
      if self.kernel == 'linear':
        self.coff = s.coef_
        self.bias = s.intercept_
      elif self.kernel == 'rbf':
        self.obj = s
    
    elif self.method =='ovr':
      s = SVC(C = self.C ,kernel = self.kernel ,max_iter = self.max_iter , random_state=10,gamma = self.g)
      s.fit(XTrain,YTrain)
      if self.kernel == 'linear':
        self.coff = s.coef_
        self.bias = s.intercept_
      elif self.kernel == 'rbf':
        self.obj = s

    return s

  def  Predict(self,XTest):
    if self.method == 'binary' and self.kernel == 'linear':
      out = np.dot(XTest,self.coff.T) + self.bias
      out =  np.where(out>=0,1,0) 
    elif self.method == 'binary' and  self.kernel == 'rbf':
      out = self.obj.decision_function(XTest)
      out =  np.where(out>=0,1,0) 
    elif self.method =='ovr':
      out = self.obj.decision_function(XTest)
      out = np.exp(-out) + 1
      out = 1.0/out
    return out

# Load the data. Dataset: Iris
Data = datasets.load_iris()
X = Data['data']
y = Data['target']

# SVM object creation and model fitting
s = SVM(max_iter=-1,kernel = 'rbf',method ='ovr')
ss = s.Fit(X,y)

# Printing the accuracy
print('Accuracy : ',accuracy(y,ss.predict(X)))