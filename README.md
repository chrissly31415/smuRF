smuRF
=====

simple multi-threaded Random Forest (smuRF)

Mainly tree based machine learning code. Current features:
- prediction with classification and regression tree (CART)
- classification and regression with Random Forest
- simple dataset statistics & manipulations
- crossvalidation
- shared memory parallelization

Dataset, parameters & workflow definition are read from a textfile "setup.txt" which may look like: 

dataset=data/mp_train.csv  
testset=data/mp_test.csv  
njobs = 4 
nrtrees = 200  
mtry = 20  
job = train_predict  

Then simply call the program from the command line:  

./smurf 




