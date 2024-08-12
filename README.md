# TASK
import pandas as pd


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# Check for missing values
print(train_df.isnull().sum())


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)


train_df.drop(['Cabin'], axis=1, inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of Age
plt.hist(train_df['Age'], bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Count plot of 'Survived'
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.show()
# Correlation matrix
corr = train_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
