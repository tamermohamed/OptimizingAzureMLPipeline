# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains information about marketing campaign of financial institution, the classification model predicts if a client will subscribe to a fixed term deposit, where the y column indicates if a customer subscribed to a fixed term deposit. for a full description you can find the data set on [Kaggle](https://www.kaggle.com/henriqueyamahata/bank-marketing).

The best classification model that get the hightest accuracy is VotingEnsemble algorithm, which we get from the Automl experiment 

## Scikit-learn Pipeline

In this expiremnt we are using HyperDrive which helping us to cover a range of hyperparameters to find the best combination of parameteres to acheive the goal which in our case is Maximizing the Accuracy

to detirmine the hyperparametrs what we need to pass to the model and the range of values to cover we are using RandomParameterSampling, which takes the max number of iteration(--max_iter) as a chice of enumeration and the Regularization Strength (--c) as a value between .1 and 1

Another argument that we pass to the hyperdriveconfig is the stopping policy, we are using BanditPolicy, in our case each run which is less than 95% of the best performing run will be terminted, this will eliminate runs that get rsults we don't need.

There is the main argument which is the estimator which is your algorithm that you will apply, we are using SKLearn, this estimator takes the train.py which is the script file that contains your custome code.

The custome code in the train.py using the sklearn LogisticRegression and a method for cleaning the the data, splitting the data to training and testing set. 

The best run we acheive accuracy of 91.45 with max iterations of 75 and Regularization Strength of .93

## AutoML
Azure AutomML automating the process of finding the best model that gets the best metric for you instead of writing algorithms, trying different models and different hyperparameters and comapre all results to get the best model which is a time consuming process.

In this experiment we are using AutoML, where it tries many different models and the best performing model was VotingEnsemble which making a prediction that is the average of multiple other regression models, this model acheives accuracy of 91.8%, it uses k-fold cross validation, and the Number of cross validations is 3

## Pipeline comparison

VotingEnsemble and LogisticRegressions one that we are getting from AutomML and the other that we are chhoing to work with. the VotingEnsemble acheives a slightly better accuracy than our custome pipline, but if we take the time that we consume to choose the model, right the code and cofigure the hyperparameters, the AutoML will be better espically if you don't know what is the best algorithm you should use, this will be a good choice.

## Future work

We have noticed that the the data set is unbalanced, so dataset need more preparation and cleaning inorder to get better result, for our custome code we can extend the hyperparamter search space and examine if we can get better results.

## Proof of cluster clean up

Finally after we end our Expirement we will remove the compute cluster, this will reduction the cost of our resources, we can do that directly from the code as follow:

compute_target.delete()

or we can use GUI to do the same task