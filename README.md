smuRF
=====

simple & mostly userfriendly Random Forest (smuRF)

Mainly tree based machine learning code. Current features:
- prediction with classification and regression tree (CART)
- classification and regression with Random Forest
- simple dataset statistics & manipulations
- crossvalidation

Input: needs .csv file (comma separated file) as input

Parameters & workflow: read from setup.txt which contains something like:

dataset=data/mp_train.csv
testset=data/mp_test.csv
nrtrees = 200 
mtry = 20 
job=train_predict





