<p align="center">
<img src="img/anomaly_detection.png" width="600" height="250" />
</p>

# Data Challenge - Machine Learning for anomaly detection
Author : Delarue Simon  
February 2021

**Objective**  

> _The main goal of this project is to detect anomalies on one of Valeo's production line (Valeo is a French Original Equipment Manufacturer). To perform this, it is required to build a machine learning model that can provides anomaly score for each produced item on the line - based on measured parameters of the item. The greater the score, the greater item's probability of being defectious (i.e anomaly)._

**Abstract**  

Learning from an imbalanced dataset is a classic problem in machine learning tasks. A few approaches to deal with it can be considered on different levels. Balancing the training dataset by oversampling the minority class, undersampling the majority class or enlarging the feature space with new information based on data. On the algorithm level, one can act on cost function by adjusting class weights or by tuning models to take into account data's specificities.  
The approach presented here can be summarized by the following steps. For each one of them, we are trying to get the best **AUC** :  
* I quickly checked the datasets and noticed the strongly unbalanced classes (Anomalies representing 2.51% of the training dataset).  
First, I decided to focus on testing several models on the original data (yet scaled), by trying techniques that usually show good results in Kaggle challenges (random forest, SVM and especially boosting techniques like XGBoost, LightGBM or CatBoost). I've also tried stacking methods (detailed further) on models that showed great performances. During these steps, I mainly focused on the **cost sensitive** aspect, by tuning hyperparameters to fit Veolia's data specificities (especially, acting on cost function by weighting data with the `scale_pos_weight` parameter. These methods led to a maximum **public test score of 80.2869 (XGBoost)**. Interestingly, even thought stacking methods were showing better results on validation set, it didn't improve public test score.  
* The predictions evaluated by the classifiers are generated with respect to the starting kit conditions ; they can be ordered by their probability of being an anomaly. However, the range of these predictions was often really small and difficult to be considered by human. I've decided to implement a **calibration technique**, in order to reshape the prediction's range and get more intuition about the probability of an item being an anomaly (using **Brier Score**). I've also tried to use these calibrations to improve predicted AUC, which was successfull for LightGBM classifier.  
* Then, taking into account that noise was generated because of strongly unbalanced classes, and also because of the overlapping classes pattern, I've followed the litterature's results on these problems, and tried **sampling methods** (oversampling and undersampling), that act directly on data, like SMOTE and SMOTE's variants (Borderline SMOTE and SVM Smote). However, these techniques led to bigger overfitting than previous tries, thus **lowered public test score**.
* Finally, I've tried some **feature engineering** techniques. Because the dataset is not massive, I've tried to enlarge the original feature space by computing either **pairwise operations** on the existing features, or **quadratic forms**. Also, considering that some data were **duplicated** in the training and test sets, I've created an additional feature, `dup` that equals $1$ fi sample is duplicated and $0$ otherwise. This field extracts information about Veolia measuring again some samples, and we can hypothetize that this is done because these samples were suspicious. If it is the case, it gives great information about the probability of an item being defectious. This field allowed me to improve significantly my public test score, which jumped to **89.4506 (XGBoost)**. 

**Results**  

Best Public Test Score : __89.4506 (AUC)__  

| Model|Feature engineering|Stacking|Valid AUC|Public Test AUC|
|------|-------------------|--------|---------|---------------|
|SGDClassifier (baseline)|Initial features|No|77.167%|74.859%|
|XGBoost|Initial features|No|84.003%|80.287%|
|Random Forest|Enlarged features|No|88.717%|87.675%|
|XGB + LGBM + RandF|Enlarged features|Yes|91.171%|87.984%|
|CatBoost|Enlarged features|No|89.968%|89.261%|
|LightGBM|Enlarged features|No|90.560%|89.381%|
|**XGBoost**|Enlarged features|No|90.714%|**89.450%**|

**Table of contents**  

1. [Exploratory Data Analysis](#ExploratoryDataAnalysis)  
    1.1. [Feature correlations](#FeatCorr)  
    1.2. [Principal Components Analysis (PCA)](#PCA)    
2. [Feature Engineering](#FeatureEngineering)  
    2.1. [Sampling Methods](#SamplingMethods)  
    2.2. [Enlarging Feature space](#EnlargeFeat)  
    2.3. [Train Test Split](#TrainTestSplit)   
    2.4. [Probability Calibration](#ProbCalib)
3. [Models](#Models)  
    3.1. [Support Vector Machines](#SVM)  
    3.2. [Xtreme Gradient Boosting (XGBoost)](#XGBoost)  
    3.3. [Light Gradient Boosting](#LGBM)  
    3.4. [CatBoost](#CatBoost)  
    3.5. [Random Forest](#RandF)  
4. [Stacking](#Stacking)  

## 1. Exploratory Data Analysis <a class="anchor" id="ExploratoryDataAnalysis"></a>

### 1.1 Features correlations  <a class="anchor" id="FeatCorr"></a>  
It can be interesting to get an intuition about the correlations between features, and the correlations between each feature and the target. However, in the next parts of this document, we will focus on **regularization** parameters of our algorithms, instead of trying to reduce the feature space. Indeed, the public test score after computing **PCA** on data didn't show any improvement.

**Feature correlation with target**
    
![png](img/output_21_0.png)

We notice that several features show great correlations, but none of them is highly correlated with the target.

### 1.2 Principal components analysis (PCA)  <a class="anchor" id="PCA"></a>  
As we mentionned, applying PCA before fitting models didn't seem to improve public test score, nonetheless, PCA is a great tool to reduce data dimension and be able to get a good intuition of the data with a visualization. Let's try this here.

**Variance explained by principal components**  
We first take a look at the Inertia explained by the principal components of the data. We notice that there is no clear **elbow** in the plot, which leads us to think that none of the features explain a significant part of the total variance.  
By selecting 11 features above 27, we can explain approwximately **90% of the total variance** though.
    
![png](img/output_25_0.png)

**Visualization**  

Interestingly, we can note there are some outliers that remains in the 'Not Anomaly' class (see below). We can hypothetize that these points are failures of sensors. To get a cleaner view of data, en also to ensure that we do not add noise in the already strongly unbalanced classes, I remove the two extreme points. Since the 'Not anomaly' class is huge, it should not interfere negatively with our model's ability to learn.

![png](img/output_29_0.png)
    
**2D plot**

![png](img/output_32_0.png)
    


The problem seems to be that classes are highly **overlapping**, which will create strong noise in our predictions.   
Training and test sets seem to have similar data distribution, which is a good start.  
These plots are just for **visualization**, since two principal components explain only a little part of the total variance.

## 2. Feature Engineering <a class="anchor" id="FeatureEngineering"></a>

### 2.1 Sampling Methods (optional) <a class="anchor" id="SamplingMethods"></a>  
Dealing with highly unbalanced classes can be challenging. The **Synthetic Minority Oversampling Technique** can be used to oversample the minority class, by creating new datapoints from the original dataset. This is a type of data augmentation. However in the present case, there is a **strong overlapping classes** schema on our data, which results in ambiguous examples after SMOTE and thus, a lower public test score than without data augmentation. We can note that this technique leads to great **overfitting** !    

Below, an example of classifier using SMOTE in its classic form. Other SMOTE variants have been tested, without better results (SVM SMOTE, Borderline SMOTE).

**Classic SMOTE**

    New distribution of target :
    ----------------------------
    Anomaly
    0          8963
    1          2689
    dtype: int64


As **XGBoost** is the model that allowed me to get the best score on public test set in the exploratory step, I stick with it when trying to improve the current performance with feature engineering.


    Best AUC score (training) : 0.9265217543433015
    Best params : {'eta': 0.01, 'eval_metric': 'auc', 'gamma': 1, 'learning_rate': 0.1, 'max_depth': 3, 'objective': 'binary:logistic', 'scale_pos_weight': 9, 'subsample': 0.9}
    
![png](img/output_39_2.png)

### 2.2 Enlarging feature space  <a class="anchor" id="EnlargeFeat"></a>    
Since the dataset is not massive, we can try to create features in order to ease the learning process. Here are the choices I made :  
* By looking at the dataset, I notice that several columns are dealing with only two values. A good start would be to **One Hot Encode** these features to prevent the model to learn hierarchy pattern between these values 
* I also notice that some features are dealing with **angles** or **torques**, and that these features are coming by pairs ; I decide to compute the **difference between the corresponding angles and torques**, in order to create new features that would reflect a disproportion in these values from a different view  
* Finally, because some features range are really small, I create a bunch of new features that are just the **squares** of original ones. The intuition behind this is to strengthen - if they exists - the weight of the anomalies  
* As mentionned in the data analysis part, some samples are **duplicated** in the datasets. We can hypothetize that these are not errors but rather that Veolia **specifically decided to measure again these items**. Following this reasonning, this could gives great information about the probability of these items to be anomalies. Indeed, when looking at the ratio of anomalies in the duplicated values, we find that the proportions are far much greather than the one observed in the entire dataset (around 50% vs less than 3% initially). Thus, I create a new feature `dup` that equals $1$ if the samples is duplicated, $0$ otherwise. This field allows the best model to gain 10 points in the public test score.  

```
Anomaly ratio in duplicated training samples : 49.39%
```

    Number of features post engineering : 49


### 2.3 Train test split  <a class="anchor" id="TrainTestSplit"></a>    
In the rest of the document, we will use the enlarged feature matrix, without SMOTE techniques.  
To **prevent further overfitting** when testing the public data, I decide to split the training dataset into :   
* A smaller training dataset : _(X_train_cut, y_train_cut)_  
* A validation set _(X_valid, y_valid)_, which will be used for hyperparameters tunning  

I make sure to use the `stratify` parameter to split the data, in order to keep the inital proportion of each classes in the splitted datasets.

### 2.4 Probability calibration  <a class="anchor" id="ProbCalib"></a>    

In our case, it could be interesting to get - not only the values of the predictions, ordered by their probability of being defectious - but also **well-calibrated predictions**, for which the predicted value of **0.8** (for example), would mean that the item has a probability of being defectious of **80%**. Indeed, this kind of technique helps human-interpretation of results.  
This can be done using the `CalibratedClassifierCV` function (invoked in our own _calibrate_predictions_ method), wihch fits a model on the predictions, with respect to the initial weights of the target, and return values smoothed in range $[0, 1]$.  

Diffenrent kinds of calibration exist. We implement two of them :  
* **Isotonic** calibration 
* **Sigmoid** calibration  

A good measure for calibration is the **Brier Score**, which I will display for each kind of calibration (the smaller the score, the better).  
As these calibrations can results in slightly different **AUC**, I also focus on trying them on the public test set, in order to check if they allow better predictions and thus higher rank in the challenge.  
To use probability calibration, it is important to get the **weights of target** in our original dataset. I create another split of the data, with respect to these weights, for the calibration step.

**Scaling Data**  
Because many features show range of values completely different, it is a wise step to **scale** the datasets. This will help the learning step, and will allow us to use **regularization techniques**.

## 3. Models <a class="anchor" id="Models"></a>  

In this section, I try to fit different models to the splitted and scaled data. First, I train the algorithm on the training dataset, then I use **GridSearchCV** to fine-tune hyperparameters on the validation set, which leads to **greater performance** - especially on tree-based methods.  

Since applying **PCA** before training models did not improve my position in public leaderbord, I've kept all the features from enlarged feature matrix, but tried `regularization` parameters.  

Below the best algorithms are listed. I've tried other methods that did not show great performance or were too slow ; Logistic Regression, C-Support vector classification, etc.

### 3.1 Support Vector Machine (SVM)  <a class="anchor" id="SVM"></a>   

The **SGDClassifier** fits a Support Vector Machine model, using **stochastic gradient descent**, which allows a much faster learning process. However, the results obtained with this classifier show **poor performance**, on the validation set, as well as the public test set.

**Stochastic Gradient Descent**

    Best AUC score (training) : 0.8691630670061313
    Best params : {'penalty': 'l2', 'loss': 'modified_huber', 'learning_rate': 'invscaling', 'eta0': 0.01, 'epsilon': 1e-05, 'early_stopping': False, 'class_weight': 'balanced', 'alpha': 0.1}
    CPU times: user 683 ms, sys: 98.2 ms, total: 781 ms
    Wall time: 498 ms
    
![png](img/output_57_0.png)
    
### 3.2 Xtreme Gradient Boosting (XGB)  <a class="anchor" id="XGB"></a>   

**XGBoost** is extensively used by Machine Learning practionnier, thank to its great amount of participation in winning solutions during Kaggle challenges. Let's try and implement a version that fits our problem, using the dedicated API : XGBClassifier.

**Hyperparameters**  
Larger ranges of hyperparameters have been tested iteratively. The one left below are the ones creating the best performance.  
* `max_depth` : The depth of the tree. Higher values lead to complex trees, and thus overfitting. 5 allows the model to learn quite quickly and perform well  
* `Learning rate`: Boosting learning rate. This rate must not be too big (too much bias) or too small (not enough learning)  
* `gamma`: Minimum loss reduction to make further partition on a leaf node  
* `subsample`: Subsample ratio of training instance  
* `eval_metric`: We want to maximize **AUC** for our problem  
* `scale_pos_weight` : Balancing of positive and negative weights. This parameter is essential regarding our dataset and its highly unbalanced classes. 9 seems the best value after multiple testing, even though it is not quite the ratio we observe in our _training data_
```
    Best AUC score (training) : 0.8814854181963238
    Best params : {'eval_metric': 'auc', 'gamma': 0.3, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 60, 'objective': 'binary:logistic', 'scale_pos_weight': 9, 'subsample': 0.9}
    CPU times: user 4min 31s, sys: 2.59 s, total: 4min 34s
    Wall time: 1min 13s
```

**Probability Calibration**  
As mentionned earlier, we fit a calibration on our predictions, in order to smoothen them in the range $[0, 1]$.  
For the XGBoostClassifier, the calibrated predictions don't seem to outperform the original results on the **public test score**, thus, we stick with the orginal model (still, it is interesting to notice that Isotonic calibration outperforms original predictions on Validation set).

![png](img/output_64_0.png)
    

XGBClassifier gives predictions in the range $[0, 1]$, but it is still interesting to try calibration techniques. In this case, it does not improve the results for the AUC metric.

### 3.3 LightGBM <a class="anchor" id="LGB"></a>

**LightGBM** is a gradient boosting framework that uses tree-based learning algorithm.  
By creating asymetric trees when selecting only promising leaves, LightGBM can achieve **faster training** step than XGBoost.  
The focus on speed of training is quite useful, especially when trying large set of parameters for **GridSearch**.  

Similar to XGBoost, I've tried different set of parameters and the one below were leading to best performance.


```python
params ={'learning_rate':[5e-2],
        'max_depth': [4],
        'reg_alpha':[1e-1],
        'reg_lambda':[1e-3],
        'gamma':[0.3],
        'subsample':[0.9],
        'objective':['binary'],
        'eval_metric':['auc'],
        'scale_pos_weight':[1]}
```
    Best AUC score (training) : 0.8870172044702539
    Best params : {'eval_metric': 'auc', 'gamma': 0.3, 'learning_rate': 0.05, 'max_depth': 4, 'objective': 'binary', 'reg_alpha': 0.1, 'reg_lambda': 0.001, 'scale_pos_weight': 1, 'subsample': 0.9}
    CPU times: user 33 s, sys: 1.83 s, total: 34.8 s
    Wall time: 10.6 s

**Calibration predictions**

![png](img/output_72_0.png)
    


For this particular problem, LightGBM AUC score is greater than the one obtained with XGBoost on the validation set. However, it led to slightly bigger **overfitting**, and the performance on public test set for LightGBM did not outperformed the on with XGBoost.  
Predictions made by **LightGBM** are not well-calibrated (see above). Sigmoid calibration achieve the same performance, but smoothen the predictions in the range $[0, 1]$ (i.e lower **Brier Score**).

### 3.4 CatBoost <a class="anchor" id="CatBoost"></a>

CatBoost is again a library for **Gradient Boosting** on decision tree, developped by Yandex.  
It is supposed to provide better generalization by reducing tree correlation and also allow to use categorical feature without pre-processing.  
The method remains the same as previously, I use `GridSearchCV` and cross-validation to find best hyper-parameters.


```python
params ={'learning_rate':[5e-2],
        'max_depth': [4],
        'reg_lambda':[1],
        'bootstrap_type': ['Bernoulli'],
        'subsample':[0.95],
        'eval_metric':['AUC'],
        'scale_pos_weight':[15],
        'logging_level':['Silent']}
```

    Best AUC score (training) : 0.8834076905595126
    Best params : {'bootstrap_type': 'Bernoulli', 'eval_metric': 'AUC', 'learning_rate': 0.05, 'logging_level': 'Silent', 'max_depth': 4, 'reg_lambda': 1, 'scale_pos_weight': 15, 'subsample': 0.95}
    CPU times: user 1min 30s, sys: 9.59 s, total: 1min 40s
    Wall time: 36.6 s

**Calibration predictions**

![png](img/output_79_0.png)

Performance is similar to the one observed with LightGBM algorithm on the validation set. However, this technique led again to bigger **overfitting** and the public test score was lower than the one with XGBoost.

### 3.5 Random forest <a class="anchor" id="RandF"></a>

A **Random Forest** is a meta-estimator that fits decision tree classifiers. It does not use gradient boosting techniques, which is slower than the techniques we used before, but is still frequently used in competitions since it uses averaging to improve performance and control overfitting.  

To allow the algorithm to automatically adjust **class weights**, I set the `class_weight` to _'balanced'_. 


```python
params ={'max_depth': [5],
        'criterion': ['gini'],
        'max_features': ['auto'],
        'class_weight':['balanced'],
        'bootstrap': [True]}
```

    Best AUC score (training) : 0.8650642316909983
    Best params : {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 5, 'max_features': 'auto'}
    CPU times: user 39.3 s, sys: 398 ms, total: 39.7 s
    Wall time: 40 s


![png](img/output_86_0.png)

Without Gradient boosting, the performance is quite lower on validation set. 

## 4. Stacking <a class="anchor" id="Stacking"></a>

**Stacking outputs** from individual classifiers, and use another classifier to compute final prediction is commonly used  to improve performance and also to reduce **overfitting**.    

In this section I use the `StackingClassifier` method from Scikit-learn, to build a meta-estimator based on classifiers listed previously (i.e classifiers that showed greatest performance on public test score).  

For each classifier, I use the best parameters found with `GridSearchCV` method during my different tries, then I build a final classifier using **XGBoost**, since it is the one that showed best individual performance.  

Finally, I plot the contributions from each classifier, and the final AUC on validation set for each individual classifiers, as for the meta-classifier.  

The **AUC** on validation set is indeed greater with **stacking method**, however, when dealing with public test dataset, the performance of this meta-classifier did not improve the leaderboard position I obtained with XGBoost only.

_Source : https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

    AUC Scores 
    ----------------------------
    XGB : 0.88393  (0.01188)
    AUC Scores 
    ----------------------------
    LGBM : 0.88231  (0.00947)
    AUC Scores 
    ----------------------------
    RandomF : 0.88535  (0.00768)
    AUC Scores 
    ----------------------------
    stacking : 0.88131  (0.00824)
    CPU times: user 7min 23s, sys: 6.17 s, total: 7min 29s
    Wall time: 2min 56s

**Plots**
    
![png](img/output_92_0.png)
    
![png](img/output_93_1.png)