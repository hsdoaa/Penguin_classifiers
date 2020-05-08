# Penguin_classifiers
Applying ML on the output of nanopolish eventalign for Pseudouridine detection 


# About  Penguin_classifiers
Penguin_classifiers are a set of predictors to identify pseudouridine sites presented in direct Nanopore RNA sequencing reads. Those predictors are integrated in the pipeline of Penguin (https://github.com/daniel235/Penguin), a tool developed for identifying Pseudouridine sites presented in Nanopore RNA sequence.
Penguin_classifiers will extract a set of features from the output of nanopolish eventalign module that is used for extracting raw signals (events) from Oxford Nanopore RNA Sequence, and use those features to predict whether the signal is modified by the presence of pseudouridine sites or not. 
Features extracted include:
- onehot encoding of reference_kmer
- event/signal  mean
- event/signal length
- event/signal standard deviation.
Penguin_classifiers have been trained and tested upon a set of 'unmodified' and 'modified' sequences containing Pseudouridine at known sites. 
Penguin_classifiers can be adopted to detect other RNA modifications which has not yet been tested.

# Considerations when using this predictor:
Current trained machine learning models of Penguin_classifiers are Support Vector Machine (SVM), Neural Network (NN), and Random Forest (RF). 
Those models has been trained and tested on cell lines 's data that has been base-called with Albacore 2.1.0.
Training new models for cell lines's data that has been base-called using other basecallers has not been yet tested. 

# What's included?
- SVM.py, NN.py, and RF.py  are python scripts for SVM, NN, and RF predictors respectively. Each script extracts three features from the output of eventalign module (event_mean, event_stdv, aand event_length) and use those features to train the classifier and test it to predict RNA Pseudouridine modifications.
- SVM_onehot.py, NN_onehot.py, and RF_onehot.py  are python scripts for SVM, NN, and RF predictors respectively. Each script use the afromentioned three features in addition to onehot encoding of reference_kmer in output of nanopolish eventalign module to train the classifier and test it to predict RNA Pseudouridine modifications.
- plot_learning_curves.py is a python script for plotting the learning curve for each predictor.

# Note:
Ech classifier script should take the following two inputs 
- The coordinate file. This file is needed to initially label the dataset feeded to train the classifier by informing it intitially about the location of Pseudouridine sites.
- The eventalign output file: This file is output from running nanopolish eventaliign module and is needed for feature extraction.

# Getting Started and pre-requisites

The following softwares and modules were used by PsNano   

- python				      3.6.10

- numpy				        1.18.1

- pandas				      1.0.1

- sklearn				      0.22.2.post1

- tensorflow			    2.0.0

- keras		            2.3.1 (using Tensorflow backend)


# Running the predictor
- To train SVM and perform predictions:
This step includes SVM training, prediction and performance assessment using the features that lead to best performance.

$ python SVM.py 

- Similarly, to train NN and perform predictions:

$ python NN.py 

- Finally to train RF and perform predictions:

$ python RF.py

# Authors:
Doaa Hassan


