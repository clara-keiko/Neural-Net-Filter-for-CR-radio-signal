import numpy as np
import scipy.signal
import scipy.fftpack
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#function to convert CGS to SI of the electrin field
def to_SI(e):
	e_si = e*29979199999.34
	return e_si

#to nanosecond
def to_ns(t):
	t_ns = t*100000000
	return t_ns 

#function to filter the signal with the natena response
#GRAND case 50Mhz to 200Mhz
#FFT BANDPASS
def filter_frequency(s,t,f_min,f_max):
	y_fft = scipy.fftpack.fft(s)
	W = scipy.fftpack.fftfreq(s.size, d=t[1]-t[0])

	# If our original signal time was in seconds, this is now in Hz    
	cut_f_signal = y_fft.copy()

	cut_f_signal[(W<f_min*1000000)] = 0
	cut_f_signal[(W>f_max*1000000)] = 0
	cut_signal = scipy.fftpack.ifft(cut_f_signal) 
	
	return cut_signal 

#function to add noise to input signal and filter the signal with the antena response
#input is the index of raw_antena<index>.dat
#output is the filtered in bandpass signal, filtered in bandpass with noise and time
def pre_process_signal(index):
	#signal
	filename_dat = 'raw_antena' + str(index) + '.dat' 
	t, x, y, z = np.loadtxt(filename_dat, delimiter ='\t', usecols =(0, 1, 2, 3), unpack = True)
	
	#noise 
	#GRAND antena is 15uV
	mu, sigma = 0, 15

	# creating a noise with the same dimension as the dataset
	noise_addx = np.random.normal(mu, sigma, x.size)
	noise_addy = np.random.normal(mu, sigma, y.size)
	noise_addz = np.random.normal(mu, sigma, z.size)

	#Gaussian noise
	signalx = filter_frequency(to_SI(x) + noise_addx,t,50,200)
	signaly = filter_frequency(to_SI(y) + noise_addy,t,50,200)
	signalz = filter_frequency(to_SI(z) + noise_addz,t,50,200)

	signalxx = filter_frequency(to_SI(x),t,50,200)
	signalyy = filter_frequency(to_SI(y),t,50,200)
	signalzz = filter_frequency(to_SI(z),t,50,200)

	signal = np.sqrt(signalx*signalx + signaly*signaly + signalz*signalz)
	signal2 = np.sqrt(signalxx*signalxx + signalyy*signalyy + signalzz*signalzz)

	signalx = np.float32(signalx.real)
	signalxx = np.float32(signalxx.real)
	
	signal = np.float32(signal.real)
	signal2 = np.float32(signal2.real)

	return signalx, signalxx, t


i=raw_input("Antenna number: ")

#retrieve signal 
signal_with_noise,signal_without_noise,t = pre_process_signal(i)

#convert to NN input type
inputX = np.asmatrix(signal_with_noise)
inputY = np.asmatrix(signal_without_noise)

#reshape in order to train using each trace individually
inputX = inputX.reshape(300,2082)
inputY = inputY.reshape(300,2082)

#percetage of data that will be trained
train_size = 1

train_cnt = int(np.floor(inputX.shape[0] * train_size))
x_train = inputX[:train_cnt]
y_train = inputY[:train_cnt]
x_test = inputX[train_cnt:]
y_test = inputY[train_cnt:]

#define the NEURAL NETWORK
def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

#number of hidden layer
n_hidden_1 = 5
n_input = x_train.shape[1]
n_classes = y_train.shape[1]

#weights and biases
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Neural Net training
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_input])

predictions = multilayer_perceptron(x, weights, biases)

# loss : sum of the squares of y0 - y_out
loss = tf.reduce_sum(tf.square(y - predictions))

# training step : gradient decent (1.0) to minimize loss
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for step in range(0,300):
		sess.run(train, feed_dict={x: x_train[step], y: y_train[step]})
	result = sess.run([weights,biases])

#output weights and bias
w = result[0]
b = result[1]

#SNR
def snr(entrada, saida):
	ratio = 0
	for i in range(0,len(entrada)-1):
		ratio += entrada[i]/(np.absolute(saida[i] - entrada[i]))
	snri = 10 * np.log(np.absolute(ratio))
	return snri

#cross correlation
def crc(a,b):
	c = np.correlate(a, b, 'full')
	return c
	
#test on new shower
i=raw_input("Validate antenna number: ")

signal_with_noise,signal_without_noise,t = pre_process_signal(i)

inputX = np.asmatrix(signal_with_noise)
inputY = np.asmatrix(signal_without_noise)

inputX = inputX.reshape(1,2082)
inputY = inputY.reshape(1,2082)

predictions = multilayer_perceptron(inputX, w, b)

with tf.Session() as sess2:
	sess2.run(tf.global_variables_initializer())
	results2 = sess2.run(predictions)

nn_filter = results2.flatten()

#Wiener Fiter
wiener_signal = scipy.signal.wiener(signal_with_noise.real)

print "Neural network filter SNR: ", snr_nn
print "Wiener filter SNR: ", snr_wf

print "Neural network filter cross correlation: ", crc_nn
print "Wiener filter cross correlation: ", crc_wf

#Plots
plt.subplot(2,2,1)
plt.plot(to_ns(t),signal_without_noise)
plt.title('Filtered signal')
plt.ylabel('Ex [uV/m]')

plt.subplot(2,2,2)
plt.plot(to_ns(t),signal_with_noise)
plt.title('Filtered signal with noise')
plt.ylabel('Ex [uV/m]')

plt.subplot(2,2,3)
plt.plot(to_ns(t),nn_filter)
plt.title('NN filter')
plt.ylabel('Ex [uV/m]')

plt.subplot(2,2,4)
plt.plot(to_ns(t),wiener_signal)
plt.title('Wiener filter')
plt.ylabel('Ex [uV/m]')
title = 'Antena' + str(i)
#
# plt.title(title)
plt.savefig(title + 'nn_wf.png')

plt.show()
