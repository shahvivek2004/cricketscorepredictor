import pandas as pd
import pickle
import matplotlib.pyplot as plt
df = pd.read_csv('ipl.csv')

columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']



df = df[df['overs']>=5.0]

from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))
print(df['bat_team'].unique())
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

print(encoded_df.head(10))

X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor
regressor2=RandomForestRegressor(n_estimators=100,random_state=0)

regressor2.fit(X_train,y_train)

y2_pred=regressor2.predict(X_test)
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))
y_pred = regressor.predict(X_test)

print(y_pred)

a=list()
a2=list()

range_min = -10
range_max = 10

count1 = 0
for i in range(len(y_test)):
    if y_pred[i] >= y_test[i] + range_min and y_pred[i] <= y_test[i] + range_max:
        count1 += 1
percentage1 = (count1 / len(y_test)) * 100
print(f"Percentage of predictions within the range {range_min} to {range_max} using random forest: {percentage1}%")
count = 0
for i in range(len(y2_pred)):
    if y_pred[i] >= y_test[i] + range_min and y_pred[i] <= y_test[i] + range_max:
        count += 1

percentage = (count / len(y_test)) * 100
print(f"Percentage of predictions within the range {range_min} to {range_max} using Linear Regression: {percentage}%")
plt.hist(y2_pred-y_test, bins=30, color='blue', edgecolor='black', alpha=0.65)
plt.title('Histogram of Predicted Scores(Random forest)')
plt.xlabel('Predicted Score - Actual Score')
plt.ylabel('Hit')
plt.show()
percentage_within_range = (count / len(y_test)) * 100
print(f"Percentage of predictions within the range {range_min} to {range_max}: {percentage_within_range}%")
plt.hist(y_pred-y_test, bins=30, color='blue', edgecolor='black', alpha=0.65)
plt.title('Histogram of Predicted Scores(Linear Regression)')
plt.xlabel('Predicted Score - Actual Score')
plt.ylabel('Hit')
plt.savefig('graph.png')