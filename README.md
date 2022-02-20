# 🏡 House Price Prediction: Project Overview
* End to end project predicting the price of a house using housing attributes like rooms and bathrooms etc.
* Used XGBoost regressor model for predictions 

## Table of Contents 
[Resources](#resources)<br>
[Data Collection](#DataCollection)<br>
[Data Pre-processing](#DataPre-processing)<br>
[Data Warehousing](#DataWarehousing)<br>
[Exploratory data analysis](#EDA)<br>
[Feature Engineering](#FeatEng)<br>
[ML/DL Model Building](#ModelBuild)<br>
[Model Evaluation](#ModelEval)<br>
[Project Management (Agile | Scrum)](#Prjmanage)<br>
[Project Evaluation](#PrjEval)<br>
[Looking Ahead](#Lookahead)<br>
[Questions | Contact me ](#Lookahead)<br>

<a name="Resources"></a>  

## Resources Used
**Python 3, PostgreSQL** 

[**Anaconda Packages:**](requirements.txt) **pandas, numpy, pandas_profiling, ipywidgets, sklearn, xgboost, matplotlib, seaborn, sqlalchemy, kaggle** <br>
Powershell command for installing anaconda packages used for this project  
```powershell
pip install pandas numpy pandas_profiling ipywidgets sklearn xgboost matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel
```
<a name="DataCollection"></a>  

## [Data Collection](Code/P5_Code.ipynb)
Powershell command for data import using kaggle API <br>
```powershell
!kaggle datasets download -d shree1992/housedata -p ..\Data --unzip 
```
[Data source link](https://www.kaggle.com/shree1992/housedata)
[Data](Data/data.csv)
*  Rows: 4600 | Columns: 18
    *  date           
    *  price          
    *  bedrooms       
    *  bathrooms      
    *  sqft_living    
    *  sqft_lot       
    *  floors         
    *  waterfront     
    *  view           
    *  condition      
    *  sqft_above     
    *  sqft_basement  
    *  yr_built       
    *  yr_renovated   
    *  street         
    *  city           
    *  statezip       
    *  country        
          
                     

<a name="DataPre-processing"></a>  

## [Data Pre-processing](Code/P5_Code.ipynb)
After I had all the data I needed, I needed to check it was ready for exploration and later modelling.   
*   Standard NULL and data validity checks in to ensure data is reliable and compatible for the model.  

<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P5_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted column headers to SQL compatibility.  

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P7_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.
* We can see that most houses in the data have between 2 - 5 rooms. 
<img src="images/categoricalfeatures_countdistrib.png" />
<img src="images/categoricalfeatures_distrib.png" />

*   I looked at the correlation the features have
<img src="images/correlation.png" />

<!-- <a name="Dataviz"></a>  

## [Data Visualisation & Analytics](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
*   I created an interactive dashboard to deploy the machine learning model to benefit the business.
*   I visualised various key features and highlighted their overall correlation to a customer’s churn. 

<a name="Busintelli"></a>  

## Business Intelligence
On Page 2 of the interactive dashboard, I have provided the stake holders with the new customer names and the customers that are likely to churn due to their characteristics.

*   These customers can be offered subsidised deals and incentives to keep them on
*   Greater engagement with customers could keep some customers on board 
*   Providing quality customer service can also provide customers with long term value and appreciation for the business
*   The complaints team should pay particular attention to complaints from customers who are predicted to churn.
- 96% of unhappy customers don’t complain and 91% of those will simply leave and never come back? -->

<a name="FeatEng"></a>  

## [Feature Engineering](Code/P2_Code.ipynb) 
Here I used OneHotEncoding to turn all features into numeric features as the XGBRegressor can only take numeric inputs. 
```python
# One Hot encoding for remaining categorical field 
data = pd.get_dummies(data, drop_first = False)
```
I also split the data into train and tests sets with a test size of 20%.

<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P11_Code.ipynb)

I applied the XGBRegressor model to achieve the predictions. 
<!-- I tried eight different models:
*   **KN Neighbors Classifier** 
*   **Linear SVC** 
*   **Decision Tree Classifier** 
*   **Random Forest Classifier**
*   **XGB Regressor** 
*   **AdaBoost Classifier**  
*   **Gaussian NB** 
*   **Quadratic Discriminant Analysis** 

<img src="images/Crossvalidation.png" /> -->

<!-- <a name="ModelPerf"></a> 

## [Model performance](Code/P11_Code.ipynb)
The Quadratic Discriminant Analysis model outperformed the other approaches on the test and validation sets. 
*   **Quadratic Discriminant Analysis** : Accuracy = 96% 

<a name="ModelOpt"></a> 

## [Model Optimisation](Code/P11_Code.ipynb)
In this step, I used GridsearchCV to find the best parameters to optimise the performance of the model.
Using the best parameters, I improved the model accuracy by **1%**

*   **Quadratic Discriminant Analysis** : Accuracy = 97% | MSE = 0.03 | RMSE = 0.17 (2dp) -->

<a name="ModelEval"></a> 

## [Model Evaluation](Code/P4_Code.ipynb)
*   I used the r2_score to see the error associated with the model. But because it is a regression use case, I can’t give an accuracy score. 
An R-Squared value above 0.7 would generally be seen as showing a high level of correlation. The model achieved a R2 value of 0.688.
A value of 0.5 means that half of the variance in the outcome variable is explained by the model.

*   Plotting the actual and predicted values for botht the training and test sets shows how accracy and linear correlation decreases in the test data. 
<img src="images/trainevaluation.png" />
<img src="images/testevaluation.png" />
<!-- <img src="images/Confusionmatrix.png" /> -->

<!-- <a name="ModelProd"></a> 

## [Model Productionisation](Code/P11_Code.ipynb)
*   A confusion matrix showing the accuracy score of 97.25% achieved by the model. 
<img src="images/Confusionmatrix.png" />

<a name="ModelDeploy"></a> 

## [Deployment](https://app.powerbi.com/view?r=eyJrIjoiNDExYjQ0OTUtNWI5MC00OTQ5LWFlYmUtYjNkMzE1YzE2NmE0IiwidCI6IjYyZWE3MDM0LWI2ZGUtNDllZS1iZTE1LWNhZThlOWFiYzdjNiJ9&pageName=ReportSection)
I built a flask REST API endpoint that was hosted on a local webserver before AWS EC2 deployment. The API endpoint takes in a request value; height and weight and returns predicted BMI index. I also optimised and formatted the frontend using HTML and CSS.  -->

<a name="Prjmanage"></a> 

## [Project Management (Agile | Scrum)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation](Presentation/P11Presentation.pptx) 
*   WWW
    *   The end-to-end process
    *   The review and process of a regression use case 
*   EBI 
    *   Better project management and planning would have made this project faster
    *   Deployment of project like this could be very beneficial to many 

<a name="Lookahead"></a> 

## Looking Ahead
*   How can I predict the price of any place of residence, caravan, apartment, flat, room and house - With data on all these places can I create one quantifiable model to accurately predict the prices of all potential residences? 

<a name="Questions"></a> 

## Questions | Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:theanalyticsolutions@gmail.com) 
* ### [See more projects here](https://github.com/MattithyahuData?tab=repositories)

 


