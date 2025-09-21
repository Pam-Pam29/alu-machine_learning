# hyperparameter_tuning
This directory contains work with tuning hyperparameters using Gaussian process and Bayesian optimization:

## Mandatory Tasks:

| Task # | Task Name | Description | File |
|--------|-----------|-------------|------|
| 0 | Initialize Gaussian Process | Create a Python class that represents a noiseless 1D Gaussian process with class constructor and public instance method to calculate the covariance kernel matrix between two matrices. | [0-gp.py](/unsupervised_learning/hyperparameter_tuning/0-gp.py) |
| 1 | Gaussian Process Prediction | Based on previous, update the class to include public instance method that predicts the mean and standard deviation of points in a Gaussian process. | [1-gp.py](/unsupervised_learning/hyperparameter_tuning/1-gp.py) |
| 2 | Update Gaussian Process | Based on previous, update the class to include public instance method that updates a Gaussian Process. | [2-gp.py](/unsupervised_learning/hyperparameter_tuning/2-gp.py) |
| 3 | Initialize Bayesian Optimization | Create a Python class that performs Bayesian optimization on a noiseless 1D Gaussian process. | [3-bayes_opt.py](/unsupervised_learning/hyperparameter_tuning/3-bayes_opt.py) |
| 4 | Bayesian Optimization - Acquisition | Based on previous, update the class to include public instance method that calculates the next best sample location. | [4-bayes_opt.py](/unsupervised_learning/hyperparameter_tuning/4-bayes_opt.py) |
| 5 | Bayesian Optimization | Based on previous, update the class to include public instance method that optimizes the black-box function. | [5-bayes_opt.py](/unsupervised_learning/hyperparameter_tuning/5-bayes_opt.py) |
| 6 | Bayesian Optimization with GPyOpt | Write a Python script that optimizes a machine learning model of choice using GPyOpt. Then, write a blog post describing the background and approach to this task. | [6-bayes_opt.py](/unsupervised_learning/hyperparameter_tuning/6-bayes_opt.py) |

### test_files directory
The test_files directory contains all files used to test output locally
