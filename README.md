
# Dibetes Prediction Models

The project uses external dataset to operate AutoML model classfication and save the best run model from this experiment. Then use the same dataset to operate Hyperdrive model and save best run model from it also. 
The second step is comparing between two final models’ performance and choose the best performance to deploy it as a web service.
The last step is to interact with the deployed  model endpoint and post sample data to test it’s work and receives results.
<img src="imges/2.jpg" >



## Dataset

### Overview
The dibetes dataset from Kaggle. It is consist of 768 records and 9 features whish are Pregnancies,	Glucose, BloodPressure, SkinThickness,	Insulin,	BMI	DiabetesPedigreeFunction,	Age, Outcome
https://www.kaggle.com/saurabh00007/diabetescsv
  
### Task
The objective is to predict based on diagnostic measurements whether a patient has diabetes or not. 
The Outcome binary result 0 or 1 express if the person has diabetes or not so we will use the other features to predict the Outcome column. 

### Access
The python code will loud data saved in the project folder in my Github account by URL.
Then register the dataset in Azure workspace by python code.  

## Automated ML
The overview of the `automl` settings configuration I used for this experiment
AutoMl experiment to classify the dataset records 
##
1. Create cluster compute "capcpu"
2. creating the expermint in the folder project and load data to the workspace.
3. Set the automl config sitting to "experiment_timeout_minutes": 30 the expermint will stop after 30 minutes
                         "max_concurrent_iterations": 3 used this number of itration to be less than compute nodes
                         "primary_metric" : 'accuracy'chose the same algorithem the automl and hyperdrive mpdels
                         task = "classification",
                         label_column_name="Outcome", it has binary result 0 or 1 express if the person has diabetes or not so we will use the other features to predict the Outcome column.
                         
## Results
### The Accuracy of voting enasamble is 0.783
### The autoML model paremters are : 'estimators': ['32', '9', '1', '28', '31', '30', '21'],
 'weights': [0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285,
             0.14285714285714285]}
  
  <a/>
## The picture show the accuracy of best run model
 <img src="imges/Automl model accuracy.png">
  
 <a/>_____________________________________________________________________________________________________________________________
  
## The Rundetailes widget of AutoML
<img src="imges/Automl rundetails.png">
<img src="imges/Automl rundetails2.png">

<a/>________________________________________________________________________________________________________________________________

## The best model of AutoML

<img src="imges/Automl best run id.png">

<img src="imges/Automl model accuracy.png">

________________________________________________________________________________________________________________________________

## Hyperparameter Tuning
Run Hyperdrive model with Logistic Regression using Sklearn with hyperparmeter c: Inverse of regularization strength (its range choicing from (0.01,0.03,0.05,0.1))
and max_iter which is the number of iterations through all classes the range between 100-200.

# parameter sampler
Chose the random sampling because it more efficient to move randomly over the space than use grid sampling which it has to go through all possible value. Random way allow adding range of parameters and it is better than put specific numbers may they aren’t best choice.

# Early stopping policy
Use BanditPolicy to defined after how much certain number of failures the experiment will stop looking for answers 


# Results
## The hyperdrive model Accuracy: 0.7662337662337663
## the best run hyperparmeter value are  ['--C', '0.1', '--max_iter', '154']

## Hyperdrive rundetails

<img src="imges/Hyperdrive rundetails.png">

<img src="imges/Hyperdrive comleted.png">


## Hyperdrive best run id

<img src="imges/Hyperdrive best run id.png">

## Model Deployment
After comparing the results of AutoML and Hyperdrive models.
<a/>________________________________________________________________________________________________________________________________

Deployed AutoML model as a webservice with name 'automl-sample-diabetes'
and make the enable insights and auth_enabled set to True for retrieve keys needed to post data to the model endpoint and receive results
### used the primary key and score url and two different data sample to test the deployed model 

### The Two sets of data to score, so we get two results back

            "Pregnancies": 5,
            "Glucose": 151,
            "BloodPressure": 60,
            "SkinThickness": 45,
            "Insulin": 0,
            "BMI":33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age":55,
                       
         
            "Pregnancies": 17,
            "Glucose": 85,
            "BloodPressure": 63,
            "SkinThickness": 29,
            "Insulin": 94,
            "BMI": 28.9,
            "DiabetesPedigreeFunction": 0.352,
            "Age": 33,
    
<a/>
### the results getting from this post showing in below picture

<img src="imges/model responce.png">

## the web service status is healthy as in picutre 
<img src="imges/endpoint model status.png">



## Screen Recording

https://drive.google.com/file/d/1-l5Or5Ut_bFdNT3VmFY5PrgL2DnD93jj/view?usp=sharing


## future work 
### Try Using the project codes with different algorithem to find better results

### Try Using the project codes with different dataset to solve anothe problem 


