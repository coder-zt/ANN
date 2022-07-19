from operator import ilshift
import numpy as np
import random
import mnist_loader

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# zip数据打乱
def shuffleZipData(zipData):
    listData = list(zipData)
    random.shuffle(listData)
    # a = []
    # b = []
    # for i in listData:
    #     a.append(i[0])
    #     b.append(i[1])
    return listData

  
class DatePackage(object):
    
    def __init__(self,data):
        inputData, resultData = zip(*data)
        self.input = inputData
        self.result = resultData
        
    def zipData(self):
        return zip(self.input, self.result)
    
    def unZipData(self, data):
        inputData, resultData = zip(*data)
        self.input = inputData
        self.result = resultData
        return (self.input, self.result)


class NetWork(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

    
    def feedForward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    
    
    def update_mini_batch(self, mini_batche, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print(mini_batche)
        for x,y in mini_batche:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            a,b = zip(*mini_batche)
            size = len(a)
            self.weights = [w - (eta/size) * nw for w,nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta/size) * nb for b,nb in zip(self.biases, nabla_b)]
            
    def backprop(self, x, y):
        # print("x ===> " + str(x))
        # print("y ===> " + str(y))
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        # 前向计算
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # print("z ===> " + str(activations[-1]))
        # activations = np.array(activations)
        # 后向计算 
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        # print("delta ===> "  + str(delta))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            # delta  = np.dot(delta, self.weights[-l + 1]) * sp
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            # print("activations[-l-1] ===> " + str(activations[-l-1]))
            nabla_w[-l] = np.dot(delta, np.array(activations[-l-1]).T)
            # nabla_w[-l] = np.dot(activations[-l - 1], delta)
        return (nabla_b,nabla_w)
        

    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data.zipData()]
        return sum(int(x == y) for (x, y) in test_results)
    # 
    # 
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data.input)
        n = len(training_data.input)
        zipTrainingData =training_data.zipData()
        for j in range(epochs):
            # 打乱训练数据
            zipTrainingData = shuffleZipData(zipTrainingData)
            # 取部分数据
            # mini_batches = [zipTrainingData[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            mini_batches = []
            count = 0
            mini_batche = []
            for zipData in zipTrainingData:
                # print("zipdata foreach")
                count += 1
                mini_batche.append(zipData)
                if count == mini_batch_size:
                    # 将列表转为zip
                    a = []
                    b = []
                    for item in mini_batche:
                        a.append(item[0])
                        b.append(item[1])
                    mini_batches.append(zip(a,b))
                    mini_batche.clear()
                    count = 0
            
            print("本次训练" + str(len(mini_batches)))
            for mini_batche in mini_batches:
                self.update_mini_batch(mini_batche,eta)
            if test_data:
                print(("Epoch {0} : {1} / {2}").format(j,self.evaluate(test_data), n_test))
            else:
                self.feedForward(training_data.input[0])
                print(("Epoch {0} complete").format(j))

def printNumPic(numArray):
    lineCount = 0
    for i in numArray:
        if i == 0:
            print("⬤⬤", end="")
        else:
            print("◎◎", end = "")
        lineCount += 1
        if(lineCount == 28):
            lineCount = 0
            print("")
    

# test()
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# trainingData = DatePackage(training_data)
# testData = DatePackage(test_data) #72104
# # training_X =  [[[1],[2]],[[2],[3]],[[4],[5]],[[1],[2]],[[2],[3]],[[4],[5]],[[1],[2]],[[2],[3]],[[4],[5]],[[1],[2]],[[2],[3]],[[4],[5]],[[1],[2]],[[2],[3]],[[4],[5]]]
# # training_Y =  []
# # for x in training_X:
# #     training_Y.append([[x[0][0] / x[1][0]]])
n = NetWork([784,30,10])
for i in n.weights:
    print(len(i))
    for j in i:
        print(len(j))

# print(n.biases)
# n.SGD(trainingData, 40,10,2.0, testData)
# for i in range(0,10):
#     printNumPic(testData.input[i * i * i])
#     res = n.feedForward(testData.input[i * i * i])
#     print(list(res).index(max(res)))
# for r in res:

    # print("%f"%r)
# # index = 0
# for i in training_data[0][3]:
#     index += 1
#     if i>0 :
#         print(1, end=" ")
#     else:
#         print(0, end=" ")
#     if index == 28:
#         print()
#         index = 0
