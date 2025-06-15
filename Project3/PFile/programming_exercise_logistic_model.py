#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
np.random.seed(42)

##################################################################
# Starter code for exercise 5: Logistic Model for Argument Quality
##################################################################
GROUP = "05"  # TODO: write in your group number


def load_feature_vectors(fl: str) -> np.array:
    datareader_var = pd.read_csv(fl, delimiter="\t", na_values=[''], keep_default_na=False)
    datanumeric_var = datareader_var.apply(pd.to_numeric, errors='coerce')
    datanumeric_var = datanumeric_var.fillna(datanumeric_var.mean())
    return datanumeric_var.to_numpy()

result = load_feature_vectors("features-train-cleaned.tsv")
print(f"Shape of the feature vectors: {result.shape}")

def load_class_values(fl: str) -> np.array:

    quality_dataset_values = pd.read_csv('quality-scores-train-cleaned.tsv',delimiter="\t") #to_numpy()
    quality_values_0 = np.where(quality_dataset_values["overall quality"] == 1.0, 0 ,1 )
    
    print(f"Number of examples in class 0: {np.count_nonzero(quality_values_0 == 0)}")
    print(f"Number of examples in class 1: {np.count_nonzero(quality_values_0== 1)}")
    return quality_values_0

result = load_class_values("quality-scores-train-cleaned.tsv")
print(f"Size of the feature vectors: {result.size}")




def misclassification_rate(cs: np.array, ys: np.array) -> float:
    
    if  len(cs) == 0:
        return float('nan')
    else:
        misclassified_values=np.sum(cs!=ys)
        total_quality = np.size(cs)
        misclass_rate=misclassified_values/total_quality
        return misclass_rate
    


def logistic_function(w: np.array, x: np.array) -> float:
    x =  np.array(x,dtype=float)
    logrithm_function =np.dot(x,w)
    return 1 / (1 + np.exp(np.clip(-logrithm_function, a_min=-500, a_max=500)))
 
    
      

def logistic_prediction(w: np.array, x: np.array) -> float:
    
    threshold_value= 0.5
    input_arary=logistic_function(w,x)
    if input_arary >=threshold_value:
        return 1
    else:
        return 0


def initialize_random_weights(p: int) -> np.array:
    return np.random.rand(p)  



def logistic_loss(w: np.array, x: np.array, c: int) -> float:
    x = np.array(x, dtype=float)
    delta = np.dot(w, x)
    loss_value = 1/(1+np.exp(np.clip(-delta,a_min=-500,a_max=500)))
    epsilon_value = 1e-15
    loss_value = np.clip(loss_value, epsilon_value, 1 - epsilon_value)
    training_loss = -(c * np.log(loss_value) + (1 - c) * np.log(1 - loss_value))
    return training_loss



def train_logistic_regression_with_bgd(xs: np.array, cs: np.array, eta: float=1e-8, iterations: int=1000, validation_fraction: float=0) -> Tuple[np.array, float, float]:
    numeric,prediction = xs.shape
    numeric_values=int(validation_fraction*numeric)
    traindata = numeric-numeric_values
    xs_train_dataset=xs[:traindata]
    cs_train_dataset=cs[:traindata]
    xs_val_classdataset=xs[traindata:]
    cs_val_classdataset=cs[traindata:]

    w=initialize_random_weights(prediction)

    trainingset_misclassification_rates =[]
    valid_set_misclassification_rates=[]
    training_loss_function = []
    
    for i in range(iterations):
        delta_weight=0
        loss_function =0
        for x,c in zip(xs_train_dataset,cs_train_dataset):
            x = np.array(x)
            if np.isnan(x).any():
                continue
            y = logistic_function(w,x)
            delta=c-y
            delta_weight += eta * np.asarray(delta, dtype=float) * np.asarray(x, dtype=float)
            loss_function += logistic_loss(w,x,c)
        w+=delta_weight
        


        training_dataset_prediction=logistic_function(w,xs_train_dataset)
        training_misclassify_rate=misclassification_rate(cs_train_dataset,training_dataset_prediction)
        trainingset_misclassification_rates.append(training_misclassify_rate)

        validation_dataset_prediction=logistic_function(w,xs_val_classdataset)
        valid_misclassify_rate=misclassification_rate(cs_val_classdataset,validation_dataset_prediction)
        valid_set_misclassification_rates.append(valid_misclassify_rate)

        avgrage_loss_function = loss_function/ len(xs_train_dataset)
        training_loss_function.append(avgrage_loss_function)
        
    return w,training_loss_function, trainingset_misclassification_rates, valid_set_misclassification_rates
    
    
  
  
def plot_loss_and_misclassification_rates(losss: List[float], train_misclassification_rates: List[float], validation_misclassification_rates: List[float]):
    """
    Plots the normalized loss (divided by max(losss)) and both misclassification rates
    for each iteration.
    """
    # TODO: Your code here
    
    normalized_loss_function = [i/max(losss) for i in losss]

    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    
    plt.plot(normalized_loss_function, label='Training Loss', color='blue')
    plt.plot( train_misclassification_rates, label='Train Misclassification Rate', color='green')
    plt.plot(validation_misclassification_rates, label='Validation Misclassification Rate', color='red')
    plt.title('Misclassification Rates over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Rate')
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)     
    # plt.close()    
    

def predict_test_set(weights, test_file='quality-scores-test-predicted.tsv'):
   
    predictions_dataset = [logistic_prediction(weights, i) for i in xs_test]

    with open(test_file, 'w') as file:
        file.write("overall quality\n") 
        for prd in predictions_dataset:
            file.write(f"{prd}\n")
    np.savetxt(test_file, predictions_dataset, fmt='%d')
    pass
########################################################################
# Tests
import os
from pytest import approx


def test_logistic_function():
    x = np.array([1, 1, 2])
    assert logistic_function(np.array([0, 0, 0]), x) == approx(0.5)
    assert logistic_function(np.array([1e2, 1e2, 1e2]), x) == approx(1)
    assert logistic_function(np.array([-1e2, -1e2, -1e2]), x) == approx(0)
    assert logistic_function(np.array([1e2, -1e2, 0]), x) == approx(0.5)


def test_bgd():
    xs = np.array([
        [1, -1],
        [1, 2],
        [1, -2],
    ])
    cs = np.array([0, 1, 0])
    
    w, _, _, _ = train_logistic_regression_with_bgd(xs, cs, 0.1, 100)
    assert w @ [1, -1] < 0 and w @ [1, 2] > 0
    w, _, _, _ = train_logistic_regression_with_bgd(-xs, cs, 0.1, 100)
    assert w @ [1, -1] > 0 and w @ [1, 2] < 0



########################################################################
# Main program for running against the training dataset

if __name__ == "__main__":
    import pandas as pd
    import pytest
    import sys

    train_features_file_name = sys.argv[1]
    train_classes_file_name = sys.argv[2]
    test_features_file_name = sys.argv[3]
    test_predictions_file_name = sys.argv[4]

    print("(a)")
    xs = load_feature_vectors(train_features_file_name)
    xs_test = load_feature_vectors(test_features_file_name)
    cs = load_class_values(train_classes_file_name)
    # TODO print number of examples with each class

    result = load_feature_vectors(train_features_file_name)
    print(f"Size of the feature vectors: {result}")

    print("(b)")
    # TODO print misclassification rate of random classifier

    print("(c)")
    test_c_result = pytest.main(['-k', 'test_logistic_function', '--tb=short', __file__])
    if test_c_result != 0:
        sys.exit(test_c_result)
    print("Test logistic function successful")

    print("(d)")
    test_d_result = pytest.main(['-k', 'test_bgd', '--tb=short', __file__])
    if test_d_result != 0:
        sys.exit(test_d_result)
    print("Test bgd successful")
    w, losss, train_misclassification_rates, validation_misclassification_rates = train_logistic_regression_with_bgd(xs, cs, validation_fraction = 0.2)

    print("(e)")
    plot_loss_and_misclassification_rates(losss, train_misclassification_rates, validation_misclassification_rates)

    print("(f)")
    predict_test_set(w)
    # TODO predict on test set and write to test_predictions_file_name

