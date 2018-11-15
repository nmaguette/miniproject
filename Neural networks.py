import numpy as np
from numpy import matrix
from math import exp, log2, log10


inputMatrix = np.matrix('0.5, 0.2, 0.4')
biaisLayer1 = np.matrix('1.0, 1.0, 1.0')
biaisLayer2 = np.matrix('1.0, 1.0, 1.0')
biaisLayer3 = np.matrix('1.0, 1.0, 1.0')
weightLayer1 = np.matrix('0.3 0.5 0.6; 0.2 0.1 0.8; 0.4 0.7 0.9')
weightLayer2 = np.matrix('0.4 0.7 0.9;0.7 0.3 0.2;0.4 0.6 0.8')
weightLayer3 = np.matrix('0.7 0.3 0.2; 0.2 0.1 0.8; 0.2 0.7 0.1')
outputMatrix = np.matrix('1.0, 0.0, 0.0')

def relu(value):
    return max(0.0,value)
def derivativeRelue(value):
    if value>0:
        return 1
    return 0


def sigmoid(value):
    return (1/(1-exp(-value)))
def derivativeSigmoid(value):
    return (sigmoid(value)*(1-sigmoid(value)))


def softmax(v1, v2, v3):
    return (exp(v1)/(exp(v1) + exp(v2) + exp(v3)))
def derivativeSoftmax(v1, v2, v3):
    return ((exp(v1) * (exp(v2) + exp(v3)))/((exp(v1) + exp(v2) + exp(v3))**2))


hInLayer1 = inputMatrix * weightLayer1 + biaisLayer1
print ('\nhInLayer1 : ', hInLayer1, ' \n')

hOutLayer1 = np.matrix([[relu(hInLayer1.item((0,0))), relu(hInLayer1.item((0,1))), relu(hInLayer1.item((0,2)))]])
print ('hOutLayer1 : ', hOutLayer1, ' \n')

hInLayer2 = hOutLayer1 * weightLayer2 + biaisLayer2
print ('hInLayer2 : ', hInLayer2, ' \n')

hOutLayer2 = np.matrix((1/(1+exp(-hInLayer2.item((0,0)))), 1/(1+exp(-hInLayer2.item((0,1)))), 1/(1+exp(-hInLayer2.item((0,2))))))
print ('hOutLayer2 : ', hOutLayer2, ' \n')

hInLayer3 = hOutLayer2 * weightLayer3 + biaisLayer3
print ('hInLayer3 : ', hInLayer3, ' \n')

sumInput = exp(hInLayer3.item((0,0))) + exp(hInLayer3.item((0,1))) + exp(hInLayer3.item((0,2)))
hOutLayer3 = np.matrix((exp(hInLayer3.item((0,0)))/sumInput,
                        exp(hInLayer3.item((0,1)))/sumInput,
                        exp(hInLayer3.item((0,2)))/sumInput))
print ('hOutLayer3 : ', hOutLayer3, hOutLayer3.item((0,2)), ' \n')


sumOutput = (outputMatrix.item((0,0)) * log10(hOutLayer3.item((0,0)))) \
            +((1-outputMatrix.item((0,0))) * log10((1 - hOutLayer3.item((0,0))))) \
            +(outputMatrix.item((0,1)) * log10(hOutLayer3.item((0,1)))) \
            +((1-outputMatrix.item((0,1))) * log10((1 - hOutLayer3.item((0,1))))) \
            +(outputMatrix.item((0,2)) * log10(hOutLayer3.item((0,2)))) \
            +((1-outputMatrix.item((0,2))) * log10((1 - hOutLayer3.item((0,2)))))

error = -sumOutput
print ('error : ', error, ' \n')

#Considering a learning rate of 0.01
learningRate = 0.01
#Modified weights of neurons after backprop
def appliedLR(value, derivateValue):
    return (value -(learningRate*derivateValue))

#BackPropagating the error in the Hidden Layer 2 and the Output Layer

derivateErrorFromOutput1 = -1 * ((outputMatrix.item((0,0)) * (1/hOutLayer3.item((0,0)))) \
            +((1-outputMatrix.item((0,0))) * (1/(1 - hOutLayer3.item((0,0))))))
derivateErrorFromOutput2 = -1 * ((outputMatrix.item((0,1)) * (1/hOutLayer3.item((0,1)))) \
            +((1-outputMatrix.item((0,1))) * (1/(1 - hOutLayer3.item((0,1))))))
derivateErrorFromOutput3 = -1 * ((outputMatrix.item((0,2)) * (1/hOutLayer3.item((0,2)))) \
            +((1-outputMatrix.item((0,2))) * (1/(1 - hOutLayer3.item((0,2))))))

errorDerivativeInLayer3 = np.matrix([[derivateErrorFromOutput1, derivateErrorFromOutput2, derivateErrorFromOutput3]])
print ('errorDerivativeInLayer3 : ', errorDerivativeInLayer3, ' \n')

derivateOutput1fromInputLayer3_1 = (exp(hInLayer3.item((0,0))) * (exp(hInLayer3.item((0,1)))+exp(hInLayer3.item((0,2)))))\
                                   /((exp(hInLayer3.item((0,0)))+exp(hInLayer3.item((0,1)))+exp(hInLayer3.item((0,2))))**2)
derivateOutput2fromInputLayer3_2 = (exp(hInLayer3.item((0,1))) * (exp(hInLayer3.item((0,0)))+exp(hInLayer3.item((0,2)))))\
                                   /((exp(hInLayer3.item((0,0)))+exp(hInLayer3.item((0,1)))+exp(hInLayer3.item((0,2))))**2)
derivateOutput3fromInputLayer3_3 = (exp(hInLayer3.item((0,2))) * (exp(hInLayer3.item((0,1)))+exp(hInLayer3.item((0,0)))))\
                                   /((exp(hInLayer3.item((0,0)))+exp(hInLayer3.item((0,1)))+exp(hInLayer3.item((0,2))))**2)

layer3Derivative = np.matrix([[derivateOutput1fromInputLayer3_1, derivateOutput2fromInputLayer3_2, derivateOutput3fromInputLayer3_3]])
print ('layer3Derivative : ', layer3Derivative, ' \n')


derivateInput1Layer3fromWeightLayer3_1 = hOutLayer2.item((0,0))
derivateInput1Layer3fromWeightLayer3_2 = hOutLayer2.item((0,1))
derivateInput1Layer3fromWeightLayer3_3 = hOutLayer2.item((0,2))

derivateInput2Layer3fromWeightLayer3_1 = hOutLayer2.item((0,0))
derivateInput2Layer3fromWeightLayer3_2 = hOutLayer2.item((0,1))
derivateInput2Layer3fromWeightLayer3_3 = hOutLayer2.item((0,2))

derivateInput3Layer3fromWeightLayer3_1 = hOutLayer2.item((0,0))
derivateInput3Layer3fromWeightLayer3_2 = hOutLayer2.item((0,1))
derivateInput3Layer3fromWeightLayer3_3 = hOutLayer2.item((0,2))

inputToOutputDerivativeMatrix = np.matrix([[derivateInput1Layer3fromWeightLayer3_1, derivateInput1Layer3fromWeightLayer3_2, derivateInput1Layer3fromWeightLayer3_3]])
print ('inputToOutputDerivativeMatrix : ', inputToOutputDerivativeMatrix, ' \n')

derivateErrorFromWeightLayer3_1_1 = derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * derivateInput1Layer3fromWeightLayer3_1
derivateErrorFromWeightLayer3_1_2 = derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * derivateInput2Layer3fromWeightLayer3_1
derivateErrorFromWeightLayer3_1_3 = derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * derivateInput3Layer3fromWeightLayer3_1

derivateErrorFromWeightLayer3_2_1 = derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * derivateInput1Layer3fromWeightLayer3_2
derivateErrorFromWeightLayer3_2_2 = derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * derivateInput2Layer3fromWeightLayer3_2
derivateErrorFromWeightLayer3_2_3 = derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * derivateInput3Layer3fromWeightLayer3_2

derivateErrorFromWeightLayer3_3_1 = derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * derivateInput1Layer3fromWeightLayer3_3
derivateErrorFromWeightLayer3_3_2 = derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * derivateInput2Layer3fromWeightLayer3_3
derivateErrorFromWeightLayer3_3_3 = derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * derivateInput3Layer3fromWeightLayer3_3

derivativeWeightLayer3 = np.matrix([[derivateErrorFromWeightLayer3_1_1, derivateErrorFromWeightLayer3_1_2, derivateErrorFromWeightLayer3_1_3]\
                                 ,[derivateErrorFromWeightLayer3_2_1, derivateErrorFromWeightLayer3_2_2, derivateErrorFromWeightLayer3_2_3]\
                                 ,[derivateErrorFromWeightLayer3_3_1, derivateErrorFromWeightLayer3_3_2, derivateErrorFromWeightLayer3_3_3]])
print ('derivativeWeightLayer3 : ', derivativeWeightLayer3, ' \n')

finalWeightMatrixLayer3 = np.matrix([[appliedLR(weightLayer3.item((0,0)),derivateErrorFromWeightLayer3_1_1), appliedLR(weightLayer3.item((0,1)),derivateErrorFromWeightLayer3_1_2), appliedLR(weightLayer3.item((0,2)),derivateErrorFromWeightLayer3_1_3)]\
                                 ,[appliedLR(weightLayer3.item((1,0)),derivateErrorFromWeightLayer3_2_1), appliedLR(weightLayer3.item((1,1)),derivateErrorFromWeightLayer3_2_2), appliedLR(weightLayer3.item((1,2)),derivateErrorFromWeightLayer3_2_3)]\
                                 ,[appliedLR(weightLayer3.item((2,0)),derivateErrorFromWeightLayer3_3_1), appliedLR(weightLayer3.item((2,1)),derivateErrorFromWeightLayer3_3_2), appliedLR(weightLayer3.item((2,2)),derivateErrorFromWeightLayer3_3_3)]])
print ('finalWeightMatrixLayer3 : ', finalWeightMatrixLayer3, ' \n')



#BackPropagating the error in the Hidden Layer 1 and the Hidden Layer 2

derivateErrorFromOutputLayer2_1 = (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((0,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((0,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((0,2)))

derivateErrorFromOutputLayer2_2 =  (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((1,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((1,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((1,2)))

derivateErrorFromOutputLayer2_3 =  (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((2,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((2,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((2,2)))

errorDerivativeInLayer2 = np.matrix([[derivateErrorFromOutputLayer2_1, derivateErrorFromOutputLayer2_2, derivateErrorFromOutputLayer2_3]])
print ('errorDerivativeInLayer2 : ', errorDerivativeInLayer2, ' \n')


derivateOutputLayer2fromInputLayer2_1 = derivativeSigmoid(hInLayer2.item((0,0)))
derivateOutputLayer2fromInputLayer2_2 = derivativeSigmoid(hInLayer2.item((0,1)))
derivateOutputLayer2fromInputLayer2_3 = derivativeSigmoid(hInLayer2.item((0,2)))

outputLayer2Derivative = np.matrix([[derivateOutput1fromInputLayer3_1, derivateOutput2fromInputLayer3_2, derivateOutput3fromInputLayer3_3]])
print ('outputLayer2Derivative : ', outputLayer2Derivative, ' \n')

derivateInput1Layer2fromWeightLayer2_1 = hOutLayer1.item((0,0))
derivateInput1Layer2fromWeightLayer2_2 = hOutLayer1.item((0,1))
derivateInput1Layer2fromWeightLayer2_3 = hOutLayer1.item((0,2))

derivateInput2Layer2fromWeightLayer2_1 = hOutLayer1.item((0,0))
derivateInput2Layer2fromWeightLayer2_2 = hOutLayer1.item((0,1))
derivateInput2Layer2fromWeightLayer2_3 = hOutLayer1.item((0,2))

derivateInput3Layer2fromWeightLayer2_1 = hOutLayer1.item((0,0))
derivateInput3Layer2fromWeightLayer2_2 = hOutLayer1.item((0,1))
derivateInput3Layer2fromWeightLayer2_3 = hOutLayer1.item((0,2))

inputLayer2Derivative = np.matrix([[derivateInput1Layer2fromWeightLayer2_1, derivateInput1Layer2fromWeightLayer2_2, derivateInput1Layer2fromWeightLayer2_3]])
print ('inputLayer2Derivative : ', inputLayer2Derivative, ' \n')

derivateErrorFromWeightLayer2_1_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_1
derivateErrorFromWeightLayer2_1_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_1
derivateErrorFromWeightLayer2_1_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_1

derivateErrorFromWeightLayer2_2_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_2
derivateErrorFromWeightLayer2_2_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_2
derivateErrorFromWeightLayer2_2_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_2

derivateErrorFromWeightLayer2_3_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_3
derivateErrorFromWeightLayer2_3_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_3
derivateErrorFromWeightLayer2_3_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_3

derivativeWeightLayer2 = np.matrix([[derivateErrorFromWeightLayer2_1_1, derivateErrorFromWeightLayer2_1_2, derivateErrorFromWeightLayer2_1_3]\
                                 ,[derivateErrorFromWeightLayer2_2_1, derivateErrorFromWeightLayer2_2_2, derivateErrorFromWeightLayer2_2_3]\
                                 ,[derivateErrorFromWeightLayer2_3_1, derivateErrorFromWeightLayer2_3_2, derivateErrorFromWeightLayer2_3_3]])
print ('derivativeWeightLayer2 : ', derivativeWeightLayer2, ' \n')

finalWeightMatrixLayer2 = np.matrix([[appliedLR(weightLayer2.item((0,0)),derivateErrorFromWeightLayer2_1_1), appliedLR(weightLayer2.item((0,1)),derivateErrorFromWeightLayer2_1_2), appliedLR(weightLayer2.item((0,2)),derivateErrorFromWeightLayer2_1_3)]\
                                 ,[appliedLR(weightLayer2.item((1,0)),derivateErrorFromWeightLayer2_2_1), appliedLR(weightLayer2.item((1,1)),derivateErrorFromWeightLayer2_2_2), appliedLR(weightLayer2.item((1,2)),derivateErrorFromWeightLayer2_2_3)]\
                                 ,[appliedLR(weightLayer2.item((2,0)),derivateErrorFromWeightLayer2_3_1), appliedLR(weightLayer2.item((2,1)),derivateErrorFromWeightLayer2_3_2), appliedLR(weightLayer2.item((2,2)),derivateErrorFromWeightLayer2_3_3)]])
print ('finalWeightMatrixLayer2 : ', finalWeightMatrixLayer2, ' \n')



#BackPropagating the error in the Input Layer and the Hidden Layer 1

derivateErrorFromOutputLayer2_1 = (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((0,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((0,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((0,2)))

derivateErrorFromOutputLayer2_2 =  (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((1,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((1,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((1,2)))

derivateErrorFromOutputLayer2_3 =  (derivateErrorFromOutput1 * derivateOutput1fromInputLayer3_1 * weightLayer3.item((2,0))) \
                                + (derivateErrorFromOutput2 * derivateOutput2fromInputLayer3_2 * weightLayer3.item((2,1))) \
                                + (derivateErrorFromOutput3 * derivateOutput3fromInputLayer3_3 * weightLayer3.item((2,2)))

errorDerivativeInLayer2 = np.matrix([[derivateErrorFromOutputLayer2_1, derivateErrorFromOutputLayer2_2, derivateErrorFromOutputLayer2_3]])
print ('errorDerivativeInLayer2 : ', errorDerivativeInLayer2, ' \n')


derivateOutputLayer1fromInputLayer1_1 = derivativeRelu(hInLayer1.item((0,0)))
derivateOutputLayer1fromInputLayer1_2 = derivativeRelu(hInLayer1.item((0,1)))
derivateOutputLayer1fromInputLayer1_3 = derivativeRelu(hInLayer1.item((0,2)))

outputLayer1Derivative = np.matrix([[derivateOutput1fromInputLayer3_1, derivateOutput2fromInputLayer3_2, derivateOutput3fromInputLayer3_3]])
print ('outputLayer1Derivative : ', outputLayer1Derivative, ' \n')

derivateInput1Layer1fromWeightLayer1_1 = inputMatrix.item((0,0))
derivateInput1Layer1fromWeightLayer1_2 = inputMatrix.item((0,1))
derivateInput1Layer1fromWeightLayer1_3 = inputMatrix.item((0,2))

derivateInput2Layer1fromWeightLayer1_1 = inputMatrix.item((0,0))
derivateInput2Layer1fromWeightLayer1_2 = inputMatrix.item((0,1))
derivateInput2Layer1fromWeightLayer1_3 = inputMatrix.item((0,2))

derivateInput3Layer1fromWeightLayer1_1 = inputMatrix.item((0,0))
derivateInput3Layer1fromWeightLayer1_2 = inputMatrix.item((0,1))
derivateInput3Layer1fromWeightLayer1_3 = inputMatrix.item((0,2))

inputLayer2Derivative = np.matrix([[derivateInput1Layer2fromWeightLayer2_1, derivateInput1Layer2fromWeightLayer2_2, derivateInput1Layer2fromWeightLayer2_3]])
print ('inputLayer2Derivative : ', inputLayer2Derivative, ' \n')

derivateErrorFromWeightLayer2_1_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_1
derivateErrorFromWeightLayer2_1_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_1
derivateErrorFromWeightLayer2_1_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_1

derivateErrorFromWeightLayer2_2_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_2
derivateErrorFromWeightLayer2_2_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_2
derivateErrorFromWeightLayer2_2_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_2

derivateErrorFromWeightLayer2_3_1 = derivateErrorFromOutputLayer2_1 * derivateOutputLayer2fromInputLayer2_1 * derivateInput1Layer2fromWeightLayer2_3
derivateErrorFromWeightLayer2_3_2 = derivateErrorFromOutputLayer2_2 * derivateOutputLayer2fromInputLayer2_2 * derivateInput2Layer2fromWeightLayer2_3
derivateErrorFromWeightLayer2_3_3 = derivateErrorFromOutputLayer2_3 * derivateOutputLayer2fromInputLayer2_3 * derivateInput3Layer2fromWeightLayer2_3

derivativeWeightLayer2 = np.matrix([[derivateErrorFromWeightLayer2_1_1, derivateErrorFromWeightLayer2_1_2, derivateErrorFromWeightLayer2_1_3]\
                                 ,[derivateErrorFromWeightLayer2_2_1, derivateErrorFromWeightLayer2_2_2, derivateErrorFromWeightLayer2_2_3]\
                                 ,[derivateErrorFromWeightLayer2_3_1, derivateErrorFromWeightLayer2_3_2, derivateErrorFromWeightLayer2_3_3]])
print ('derivativeWeightLayer2 : ', derivativeWeightLayer2, ' \n')

finalWeightMatrixLayer2 = np.matrix([[appliedLR(weightLayer2.item((0,0)),derivateErrorFromWeightLayer2_1_1), appliedLR(weightLayer2.item((0,1)),derivateErrorFromWeightLayer2_1_2), appliedLR(weightLayer2.item((0,2)),derivateErrorFromWeightLayer2_1_3)]\
                                 ,[appliedLR(weightLayer2.item((1,0)),derivateErrorFromWeightLayer2_2_1), appliedLR(weightLayer2.item((1,1)),derivateErrorFromWeightLayer2_2_2), appliedLR(weightLayer2.item((1,2)),derivateErrorFromWeightLayer2_2_3)]\
                                 ,[appliedLR(weightLayer2.item((2,0)),derivateErrorFromWeightLayer2_3_1), appliedLR(weightLayer2.item((2,1)),derivateErrorFromWeightLayer2_3_2), appliedLR(weightLayer2.item((2,2)),derivateErrorFromWeightLayer2_3_3)]])
print ('finalWeightMatrixLayer2 : ', finalWeightMatrixLayer2, ' \n')


