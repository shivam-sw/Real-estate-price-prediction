import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib_inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
import seaborn as sns
import streamlit as st



df1 = pd.read_csv("Bengaluru_House_Data.csv")

df1.groupby('area_type')['area_type'].agg('count')  #to see how many variables are there in perticular column


df2 = df1.drop(['area_type','availability','society','balcony'], axis='columns')   #drop the useless columns
df3=df2.dropna()  

#df3.groupby('size')['size'].agg('count')

df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))



#so in the above line run we will see there is one home with 43 bedroom in 2400 sqft which is kind of an error, which we need to clean


#we will see some value which is in range

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


#taking the avg of range
def convert_sqft_to_num(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None

#convert_sqft_to_num('2166 - 2345')

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
#lets check


#getting the price per sqft
df5 = df4.copy()
df5['price_per_sqft'] = (df5['price']*100000)/df5['total_sqft']


#lets handle categorical data - location
#first we will handle extra spaces
df5.location = df5.location.apply(lambda x:x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending= False)
#this will give you different location name with number of rows


location_stats_less_than_10 = location_stats[location_stats<=10]

df5.location= df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x )

#outlier detection

df6 = df5[~(df5.total_sqft/df5.BHK<300)]


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)


def pot_scatter_chart(df,location):
    BHK2 = df[(df.location==location) & (df.BHK==2)]
    BHK3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(BHK2.total_sqft,BHK2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(BHK3.total_sqft,BHK3.price,marker='+',color='green',label='3 BHK', s=50)
    plt.xlabel("Total square feet area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
pot_scatter_chart(df7,"Hebbal")



def remove_BHK_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
       BHK_stats = {}
       for BHK, BHK_df in location_df.groupby('BHK'):
            BHK_stats[BHK] = {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
    for BHK, BHK_df in location_df.groupby('BHK'):
            stats = BHK_stats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')


df8 = remove_BHK_outliers(df7)
#pot_scatter_chart(df8,"Hebbal")               
        


# plt.hist(df8.bath,rwidth=0.8)
# plt.xlabel("Number of bathrooms")
# plt.ylabel("count")


df9 = df8[df8.bath<df8.BHK+2]

df10 = df9.drop(['size','price_per_sqft'],axis='columns')

#model building
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')

df12 = df11.drop('location', axis='columns')


X = df12.drop('price',axis='columns')
y = df12.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10) 

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test, y_test)


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
       gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
       gs.fit(X,y)
       scores.append({
           'model': algo_name,
           'best_score': gs.best_score_,
           'best_params': gs.best_params_
       })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


find_best_model_using_gridsearchcv(X,y)


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

predict_price('1st Phase JP Nagar',0, 0, 0)

#################################################################################################
#Streamlit Code

feature_names = ['Locality', 'Area in square feet', 'Number of Bathrooms', 'BHK']
location = df12.columns


st.title("Real Estate Price Prediction")


menu = ['Home', 'Prediction']

choice = st.sidebar.selectbox("Menu", menu)

if choice=='Prediction':
    st.subheader('Get Comprehensive Information on Real Estate Prices in Bangalore')
    sqft = st.number_input('Area in square feet')
    bath = st.number_input('Number of Bathrooms')
    bhk = st.number_input('BHK')
    location = st.selectbox("Locality", location )
    
    pp = predict_price(location,sqft,bath,bhk)
    
    if st.button("Predict the Price"):
              st.write('Predicted Price is: ', pp)
    
    



