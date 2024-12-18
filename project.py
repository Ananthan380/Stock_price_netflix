import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('c:\\Users\\user\\Desktop\\stockprice.csv')
print(df.head())
print(df.shape)
print(df.describe())
print(df.info())
plt.figure(figsize=(15,5))
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
plt.plot(df['date'],df['close'])
plt.title('Netflix close price',fontsize=15)
plt.ylabel('price in dollars')
plt.xlabel('date')
plt.show()
print(df.isnull().sum())
features = ['open', 'high', 'low', 'close', 'volume']

plt.subplots(figsize=(20,10))
plt.axis('off')
i=0
for col in features:
  plt.subplot(2,3,i+1)
  sb.histplot(df[col],kde=True)
  i=i+1
plt.show()

plt.subplots(figsize=(20,10))
plt.axis('off')
i=0
for col in features:
  plt.subplot(2,3,i+1)
  sb.boxplot(x=df[col],orient='h')
  i=i+1
plt.show()

df.columns=df.columns.str.strip()
df['date']=pd.to_datetime(df['date'],format='%d-%m-%Y')
df['day']=df['date'].dt.day
df['month']=df['date'].dt.month
df['year']=df['date'].dt.year
df['is_quarter_end']=[1 if month % 3 == 0 else 0 for month in df['month']]
print(df.head())

yearly_average_prices = df.groupby('year')['close'].mean()
print(yearly_average_prices)

data_grouped = df.drop('date', axis=1).groupby('year').mean()
plt.subplots(figsize=(10,10))
plt.axis('off')
i=0
for col in ['open','high','low','close']:
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
  plt.title(col)
  i=i+1
  plt.tight_layout()
plt.show()

print(df.drop('date', axis=1).groupby('is_quarter_end').mean())

df['open-close']=df['open']-df['close']
df['low-high']=df['low']-df['high']
df['target']=np.where(df['close'].shift(-1)>df['close'],1,0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10)) 
sb.heatmap(df.drop('date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()
features=df[['open-close','low-high','is_quarter_end']]
target=df['target']
scaler=StandardScaler()
features=scaler.fit_transform(features)
X_train,X_valid,Y_train,Y_valid=train_test_split(
    features,target,test_size=0.1,random_state=2024)
print(X_train.shape,X_valid.shape)
models =[LogisticRegression(),SVC(
  kernel='poly',probability=True),XGBClassifier()]
for i in range(3):
  models[i].fit(X_train,Y_train)

  print(f'{models[i]} :')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()
  from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.title('LogisticRegression')
plt.show()

# ### Simulating Predictions for 2024 ###

# # Generate future dates for 2024 (business days only)
# future_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')
# future_df = pd.DataFrame(future_dates, columns=['date'])

# # Create placeholders for future features based on historical patterns
# future_df['day'] = future_df['date'].dt.day
# future_df['month'] = future_df['date'].dt.month
# future_df['year'] = future_df['date'].dt.year
# future_df['is_quarter_end'] = [1 if month % 3 == 0 else 0 for month in future_df['month']]

# # Start with the last known closing price
# last_known_price = df['close'].iloc[-1]
# future_df['close'] = last_known_price

# # Simulate 'open-close' and 'low-high' using random noise
# future_df['open-close'] = np.random.normal(loc=0, scale=10, size=len(future_df))  # Random noise
# future_df['low-high'] = np.random.normal(loc=0, scale=5, size=len(future_df))  # Random noise

# # Scale future features just like the training data
# future_features = future_df[['open-close', 'low-high', 'is_quarter_end']]
# future_features = scaler.transform(future_features)

# # Predict movement (up/down) for 2024 using the Logistic Regression model
# future_df['predicted_movement'] = models[0].predict(future_features)

# # Simulate the stock price changes based on predicted movement
# for i in range(1, len(future_df)):
#     if future_df.loc[i, 'predicted_movement'] == 1:  # If up
#         future_df.loc[i, 'close'] = future_df.loc[i - 1, 'close'] * (1 + np.random.uniform(0.001, 0.01))  # Positive change
#     else:  # If down
#         future_df.loc[i, 'close'] = future_df.loc[i - 1, 'close'] * (1 - np.random.uniform(0.001, 0.01))  # Negative change

# # Plot the predicted price trend for 2024
# plt.figure(figsize=(10, 5))
# plt.plot(future_df['date'], future_df['close'], label='Predicted Stock Price')
# plt.title('Simulated Netflix Stock Price Trend for 2024')
# plt.xlabel('Date')
# plt.ylabel('Stock Price (in USD)')
# plt.legend()
# # plt.show()
# ### Combining Historical and Predicted Prices into a Bar Plot ###

# # Calculate the average predicted closing price for 2024
# avg_close_2024 = future_df['close'].mean()

# # Append the predicted 2024 price to the historical yearly averages
# combined_yearly_avg_prices = pd.concat([yearly_average_prices, pd.Series({'2024': avg_close_2024})])

# # Plot the bar graph from the first known year to the predicted 2024 price
# plt.figure(figsize=(12, 6))
# combined_yearly_avg_prices.plot(kind='bar', color='skyblue')
# plt.title('Netflix Stock Price Trend (Historical & Predicted for 2024)')
# plt.ylabel('Average Closing Price (USD)')
# plt.xlabel('Year')
# plt.xticks(rotation=45)
# plt.show()
