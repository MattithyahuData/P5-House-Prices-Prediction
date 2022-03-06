# 🏡 House Price Prediction: Project Overview
* End to end project predicting the price of a house using housing attributes like rooms and bathrooms etc.
* Used XGBoost regressor model for predictions 

## Table of Contents 
*   [Resources](#resources)<br>
*   [Data Collection](#DataCollection)<br>
*   [Data Pre-processing](#DataPre-processing)<br>
*   [Data Warehousing](#DataWarehousing)<br>
*   [Exploratory data analysis](#EDA)<br>
*   [Feature Engineering](#FeatEng)<br>
*   [ML/DL Model Building](#ModelBuild)<br>
*   [Model Evaluation](#ModelEval)<br>
*   [Project Management (Agile/Scrum/Kanban)](#Prjmanage)<br>
*   [Project Evaluation](#PrjEval)<br>
*   [Looking Ahead](#Lookahead)<br>
*   [Questions & Contact me](#Lookahead)<br>

<a name="resources"></a>  

## Resources Used
**Python 3, PostgreSQL** 

[**Anaconda Packages:**](requirements.txt) **pandas, numpy, pandas_profiling, ipywidgets, sklearn, xgboost, matplotlib, seaborn, sqlalchemy, kaggle** <br><br>
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
*  Rows: 4600 / Columns: 18
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

The data contained no null values and all datatypes lined up with their field description. <br>

```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```

<br>

*   Some programming languages can be case sensitive like python and C++ for example, so using lower case letters for variable names allows for straightforward use of data in different programming languages.<br>

```python
# (SQL standard) Formatting column headers by removing potential capital letters and spaces in column headers 
data.columns = data.columns.str.lower()
data.columns = data.columns.str.replace(' ','_')
```

<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P5_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted column headers to SQL compatibility. 


```python 
# Function to warehouse data in a Postgre database 
def store_data(data,tablename):
    """
    :param data: variable, enter name of dataset you'd like to warehouse
    :param tablename: str, enter name of table for data 
    """

    # SQL table header format
    tablename = tablename.lower()
    tablename = tablename.replace(' ','_')

    # Saving cleaned data as csv
    data.to_csv(f'../Data/{tablename}_clean.csv', index=False)

    # Engine to access postgre
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/projectsdb')

    # Loads dataframe into PostgreSQL and replaces table if it exists
    data.to_sql(f'{tablename}', engine, if_exists='replace',index=False)

    # Confirmation of ETL 
    return("ETL successful, {num} rows loaded into table: {tb}.".format(num=len(data.iloc[:,0]), tb=tablename))
 
# Calling store_data function to warehouse cleaned data
store_data(data,"P2 Bank Churn Prediction")
```

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P7_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.

* We can see that most houses in the data have between 2 - 5 rooms. 
<img src="images/categoricalfeatures_countdistrib.png" />
<img src="images/categoricalfeatures_distrib.png" />

*   The data is somewhat correlated.
<img src="images/correlation.png" />


<a name="FeatEng"></a>  

## [Feature Engineering](Code/P2_Code.ipynb) 
Here I used OneHotEncoding to turn all features into numeric features as the XGBRegressor can only take numeric inputs. 

```python
# One Hot encoding for remaining categorical field 
data = pd.get_dummies(data, drop_first = False)

# Splitting dependent and independent features to apply scaling
X = data.drop(['price'], axis=1)

# Dependent feature | Target variable 
y = data['price']
```

I also split the data into train and tests sets with a test size of 20%.

```python
# Using train test split to split train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23, shuffle=True)

# Viewing shape of train / test data
print(X_train.shape)
print(X_test.shape)
```


<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P11_Code.ipynb)

I applied the XGBRegressor model to achieve the predictions. 


<a name="ModelEval"></a> 

## [Model Evaluation](Code/P4_Code.ipynb)
*   I used the r2_score to see the error associated with the model. But because it is a regression use case, I can’t give an accuracy score. 
An R-Squared value above 0.7 would generally be seen as showing a high level of correlation. The model achieved a R2 value of 0.688.
A value of 0.5 means that half of the variance in the outcome variable is explained by the model.

*   Plotting the actual and predicted values for botht the training and test sets shows how accracy and linear correlation decreases in the test data. 
<img src="images/trainevaluation.png" />
<img src="images/testevaluation.png" />


<a name="Prjmanage"></a> 

## [Project Management (Agile/Scrum/Kanban)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation]() 
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

## Questions & Contact me
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)

 


