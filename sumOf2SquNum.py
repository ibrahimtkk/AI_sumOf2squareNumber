import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits import mplot3d
from sklearn.metrics import mean_absolute_error


def sonucBul(inputs):
    row, col = np.shape(inputs)
    expected_output = np.zeros((row,1))
    for i in range (row):
        expected_output[i] = inputs[i][0] * inputs[i][0] + inputs[i][1] * inputs[i][1] 
    return expected_output

x = np.arange(-10, 10, 0.2)  
y = np.arange(-10, 10, 0.2)
 
random.shuffle(x)           
random.shuffle(y)
inputs = np.array((x,y)).T
gercekSonuclar = sonucBul(inputs)


X = inputs
y = gercekSonuclar

xMax = np.amax(X, axis=0)
yMax = np.amax(y, axis=0)
#X = X/np.amax(X, axis=0) 
y = y/np.amax(y, axis=0)  



# 8-6 iyi sonuc verdi
class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 8
    self.hidden2Size = 6
    self.sonucList = []
    #self.errorList = []
    self.XList, self.YList, self.TargetList = [], [], []
    self.lr = 0.1
    self.hataYuzdesi = 0.1

    with open('input.txt', 'r') as dosya:
      hepsi = dosya.readlines()

    parcaliHepsi = [x.strip() for x in hepsi[1:]]

    for i in range(len(parcaliHepsi)):
      vSiz = parcaliHepsi[i].split(',')
      self.XList.append(float(vSiz[0]))
      self.YList.append(float(vSiz[1]))
      self.TargetList.append(float(vSiz[2]))

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.hidden2Size) # (3x1) weight matrix from hidden to output layer
    self.W3 = np.random.randn(self.hidden2Size, self.outputSize)

  def plot2D_Graph(self, errList):
    xAxis = np.arange(0, len(errList) ,1)
    plt.plot(xAxis, errList)
    plt.show()


  def plotGraph(self, X, y):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xLine = X[:,0]
    yLine = X[:,1]
    zLine = y.ravel()

    tso = gercekSonuclar.ravel()
    ax.plot_trisurf(xLine, yLine, tso, linewidth=0.5, antialiased=True,color='green', alpha=0.5)
    ax.plot_trisurf(xLine, yLine, zLine, linewidth=0.5, antialiased=True,color='red', alpha=0.95)

    plt.show()

  def predict(self, X):
    o = self.forward(X)
    return o

  def lastForward(self, X):
    self.z = np.dot(X, self.W1) # dot product of X (input=20x2) and first set of 2x3 weights = 20x3
    self.z2 = self.sigmoid(self.z) # activation function = 20x3
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2=20x3) and second set of 3x1 weights = 20x1
    self.z4 = self.sigmoid(self.z3) # final activation function = 20x1
    self.z5 = np.dot(self.z4, self.W3)
    o = self.sigmoid(self.z5)
    self.sonucList = o
    return o

  def forward(self, X):
    #forward propagation through our network
    # X = 20x2 matrix
    self.z = np.dot(X, self.W1) # dot product of X (input=20x2) and first set of 2x3 weights = 20x3
    self.z2 = self.sigmoid(self.z) # activation function = 20x3
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2=20x3) and second set of 3x1 weights = 20x1
    self.z4 = self.sigmoid(self.z3) # final activation function = 20x1
    self.z5 = np.dot(self.z4, self.W3)
    o = self.sigmoid(self.z5)
    #print o[0]
    return o 

  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o, i):
    # backward propgate through the network
    # X = 20x2, y= 20x1, o = 20x1
    # o = predicted cikti katmani = 20x1
    self.o_error = (y - o) # error in output
    self.o_delta = self.lr * self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.lr * self.z2_error*self.sigmoidPrime(self.z4) # applying derivative of sigmoid to z2 error

    self.z1_error = self.z2_delta.dot(self.W2.T)
    self.z1_delta = self.lr * self.z1_error*self.sigmoidPrime(self.z2)

    # W1 = 2x3, W2 = 3x1
    self.W1 += X.T.dot(self.z1_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.z2_delta) # adjusting second set (hidden --> output) weights
    self.W3 += self.z3.T.dot(self.o_delta)

  def train (self, X, y, i):
    # X = 20x2, y= 20x1
    o = self.forward(X)
    self.backward(X, y, o, i)

NN = Neural_Network()

def mean_absolute_percentage_error(y_true, y_pred): 
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


epoch = 0
NN.train(X, y, epoch)
errList = []
err = mean_absolute_error(y, NN.forward(X))
while ( err > NN.hataYuzdesi):
  print( epoch, err )
  errList.append(err)
  epoch += 1
  NN.train(X, y, epoch)
  err = mean_absolute_error(y, NN.forward(X))


print('iterasyon: ', epoch)

### predict Islemi
predX = np.array([6,4])

# ##  2. kisa, net, adam gibi adam yol:
tahmin = NN.predict(predX) * yMax
print("tahmin: ", tahmin)


print("\n")
NN.lastForward(X)
for i in range(len(NN.sonucList)):
  NN.sonucList[i] *= yMax

errorFin = 0
print("\n\n\n---")
for i in range(len(NN.sonucList)):
  errorFin += abs(gercekSonuclar[i] - NN.sonucList[i])
  print(gercekSonuclar[i], NN.sonucList[i])

# ## PLOT ISLEMI
NN.plot2D_Graph(errList)

NN.plotGraph(X, NN.sonucList)

