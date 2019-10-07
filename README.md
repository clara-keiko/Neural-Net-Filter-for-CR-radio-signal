# Neural-Net-Filter-for-CR-radio-signal
#Version 1 - 07/10/2019
#Code to filter a signal using neural networks
#NOTE: this code uses COREAS cosmic rays files as inputs: Ex,Ey,Ez,t in CGS units

#The NN consists of an input layer of (1 x 2082), hidden layer, sigmoind type with 5 units, and a output linear layer
#To train this NN is expected a COREAS simulated shower, from GRAND array configuration with 300 antenas traces.

#input type:
#raw_antena666.dat (file with all antenas traces together)
#Note: To join multiple files cat raw_antena*.dat >> raw_antena666.dat

#validation antena:
#raw_antena6.dat (file with a single antena trace)

