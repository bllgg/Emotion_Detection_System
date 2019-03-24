import nn
import datetime

start_time = datetime.datetime.now()

topology = []
topology.append(3)
topology.append(3)
topology.append(2)
net = nn.Network(topology)
nn.Neuron.eta = 0.5
nn.Neuron.alpha = 0.015
while True:
    err = 0
    inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1 ,0 ,0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs = [[0, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [1, 1]]
    for i in range(len(inputs)):
        net.setInput(inputs[i])
        net.feedForword()
        net.backPropagate(outputs[i])
        err = err + net.getError(outputs[i])
    print ("error: ", err)
    if err < 0.01:
        break

stopped_time = datetime.datetime.now()

print("time to train = ",stopped_time-start_time)

while True:
    a = input("type 1st input : ")
    b = input("type 2nd input : ")
    c = input("type 3rd input : ")
    net.setInput([a, b, c])
    net.feedForword()
    print (net.getThResults())


