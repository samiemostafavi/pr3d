# Network Calculus

1. The dataset is generated from the bitdomain MATLAB project. 
2. Bitdomain project simulation is timeslot-based. Hence the delay values are only integers e.g. 0, 1, 2, 3, etc which makes the ML model prone to overfitting. They have a lot of reduntancy as well. Hence, two preprocessing tasks are necessary in order to prepare the data:
    1. Adding Gaussian noise to the delay values with the standard deviation of ~0.7.
    **IMPORTANT**: do not remove or zero the negative delays after adding the Gaussian noise. Let them be and train the model with them. Otherwise the ML model cannot be trained.
    2. Downsample the dataset in terms of network state. e.g. when there are samples for the states x=[0,1,2,3,4,5,6,7,8,9,10], only keep x=[0,5,10] and delete the rest.
    

