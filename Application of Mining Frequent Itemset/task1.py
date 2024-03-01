import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

data = pd.read_excel("Online_Retail.xlsx")
data.head()
data.tail()
data.info()
data=data[pd.notnull(data['CustomerID'])]

filtered_data=data[['Country', 'CustomerID']].drop_duplicates()

filtered_data.Country.value_counts()[:10].plot(kind='bar')
uk_data=data[data.Country=='United Kingdom']
uk_data.info()
uk_data.describe()

#calculate Recency
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#calculate last purchase date
last_purchase_date = data['InvoiceDate'].max()

#recency for each customer
data['Recency'] = last_purchase_date - data.groupby('CustomerID')['InvoiceDate'].transform('max')
data['Recency'] = data['Recency'].dt.days

#calculate frequency
frequency_data = data.groupby('CustomerID')['InvoiceNo'].count().reset_index()
frequency_data.columns = ['CustomerID', 'Frequency']

#calculate monetary
monetary_data = data.groupby('CustomerID')['UnitPrice'].sum().reset_index()
monetary_data.columns = ['CustomerID', 'Monetary']

rfm_data = frequency_data.merge(monetary_data, on='CustomerID')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data['Recency'], kde=True)
plt.title('Recency Distribution')

plt.subplot(1, 3, 2)
sns.histplot(rfm_data['Frequency'], kde=True)
plt.title('Frequency Distribution')

plt.subplot(1, 3, 3)
sns.histplot(rfm_data['Monetary'], kde=True)
plt.title('Monetary Distribution')

plt.tight_layout()
plt.show()
