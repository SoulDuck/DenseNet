# DenseNet

## abstract 
#### 1.DenseNet connects each layer to every other layer in a feed-forward fashion

#### 2.DenseNet has L(L+1)/2 direct connections

#### 3.For each layer, the feature-maps of all preceding layers are used as inputs, 
#### and its own feature-maps are used as inputs into all subsequent layers.

<img src="readme_pic/figure__1.png" alt="Drawing" style="width: 100px; height: 200px"/>



## advantages:

#### 1.alleviate the vanishing-gradient problem

#### 2.strengthen feature propagation

#### 3.encourage feature reuse

#### 4.substantially reduce the number of parameters

#### 5.requiring less computation to achieve high performance

# Usage
#### train --train
#### test --test 

#### example : python run_this_code.py --train --not_renew_logs 


# Result
<img src="readme_pic/figure_2.png" alt="Drawing" style="width: 200px; height: 200px"/>

<img src="readme_pic/figure_3.png" alt="Drawing" style="width: 200px; height: 200px"/>

<img src="readme_pic/figure_4.png" alt="Drawing" style="width: 200px; height: 200px"/>

# update list 

#### restore model and continue Training 

#### Class Relation Flow 

#### BottleNect | Depth 40 | Block 3 

#### Ensemble



