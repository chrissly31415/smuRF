smuRF
=====

simple multi-threaded Random Forest (smuRF)

Mainly tree based machine learning code. Current features:
- classification and regression tree (CART)
- bagging of classification and regression trees (Random Forest)
- simple dataset statistics & manipulations
- crossvalidation
- shared memory parallelization
- runs under windows/linux

Dataset, parameters & workflow definition are read from a textfile "setup.txt" which may look like: 

dataset=data/mp_train.csv  
testset=data/mp_test.csv  
njobs = 4  
nrtrees = 200    
mtry = 20    
job = train_predict  

Save "setup.txt" to location of the binary.  

Then simply call the program from the command line:  

./smurf 

Binaries for windows and linux (64bit) are in folder bin/    




