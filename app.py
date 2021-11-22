from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
dataset1 = pd.read_csv('data/odi.csv')
dataset2 = pd.read_csv('data/t20.csv')
dataset3 = pd.read_csv('data/ipl.csv')
dataset4 = pd.read_csv('data/spcl_t20.csv')


def score(dataset,cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker):
    # Importing the dataset
    
    X = dataset.iloc[:,[7,8,9,12,13]].values

    y = dataset.iloc[:, 14].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the dataset
    lin = LinearRegression()
    lin.fit(X_train,y_train)

    ran = RandomForestRegressor(n_estimators=100,max_features=None)
    ran.fit(X_train,y_train)

    knn = KNeighborsClassifier(n_neighbors = 11)
    knn.fit(X_train,y_train)

    las = Lasso(alpha=0.01, max_iter=10e5)
    las.fit(X_train,y_train)

    # Testing with a custom input
    if cur_score<cur_striker+cur_non_striker or cur_overs<0 or cur_overs>50:
        print('Error in Input')
    else:
    
        lin_prediction = lin.predict(sc.transform(np.array([[cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker]])))
        #print("Linear Regression - Prediction score:" , new_prediction)
        ran_prediction = ran.predict(sc.transform(np.array([[cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker]])))
        #print("Random Forest Regression - Prediction score:" , new_prediction)
        knn_prediction = knn.predict(sc.transform(np.array([[cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker]])))
        #print("K Nearest Neighbours - Prediction score:" , new_prediction)
        los_prediction = las.predict(sc.transform(np.array([[cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker]])))
        #print("Lasso Regression - Prediction score:" , new_prediction)
        return [lin_prediction,ran_prediction,knn_prediction,los_prediction]



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    arr = list()
    
    if request.method == 'POST':
        type_csv = request.form['format_type']
        if type_csv == 'odi':
            dataset =dataset1
           
        elif type_csv == 't20':
            dataset =dataset2
            
        elif type_csv == 'ipl':
            dataset =dataset3

        elif type_csv == 'all_stars':
            dataset =dataset4
        
        cur_score = int(request.form['runs'])
        cur_wickets = int(request.form['wickets'])
        cur_overs = float(request.form['overs'])
        cur_striker = int(request.form['runs_in_prev_5'])
        cur_non_striker = int(request.form['wickets_in_prev_5'])

        
        l = score(dataset,cur_score,cur_wickets,cur_overs,cur_striker,cur_non_striker)        
                     
        return render_template('result.html', lin=l[0][0],ran=l[1][0],knn=l[2][0],las=l[3][0])



if __name__ == '__main__':
	app.run(debug=True)