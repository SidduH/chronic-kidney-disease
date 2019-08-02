# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# function to classify and display the results 
def runClassifer(classifier, X_train, y_train, y_test):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print("Accuracy:",accuracy_score(y_test, y_pred))
        print("Confusion matrix: \n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))   


# Set the preferences
pd.set_option('display.max_columns',100) 
pd.set_option('display.max_colwidth', 100)

# Read the data from the file
Data = pd.read_csv('kidneyChronic.csv')

# Data Pre Processing
# Steps followed
# 1. Replace all ? with np.nan
# 2. Use SimpleImputer to update missing values with mean for numeric attributes
#    and most frequent value for nominal variables
# 3. Encode the labels for the nominal attribute values
# 4. Pick one of the attributes among the highly correlated attributes
# 4. Use StandardScaler to scale and transform the data  

# Replace missing values with NaN
Data.replace(['?', '?\t', '\t?'], np.nan, inplace=True)

# Columns for the numeric and nominal attributes
numerical_columns = ['age', 'bp',  'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc'] # 11 columns
nominal_columns = ['sg','al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad','appet', 'pe', 'ane', 'class']  # 14 columns

# Fill the missing values with mean and most frequent value for the numeric and nominal
# attributes
imp_nominal = SimpleImputer(strategy="most_frequent")
imp_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
Data[nominal_columns] = imp_nominal.fit_transform(Data[nominal_columns])
Data[numerical_columns] = imp_numeric.fit_transform(Data[numerical_columns])
# Encode the labels for the nominal attributes
labelencoder = LabelEncoder()
Data[nominal_columns] = Data[nominal_columns].apply(labelencoder.fit_transform)
Data = Data.astype(float)

# Check if any of the attributes are corelated using the the heatmap
corr = Data.iloc[:, 0:-1].astype(float).corr(method='spearman').abs()
print("Heatmap for correlation of the attributes:")
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
plt.show()
# Select upper triangle of correlation matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
Data = Data.drop(columns=to_drop)

# Divide X and y based on the features and class column/attribute
X = Data.iloc[:, 0:-1]
y = Data.iloc[:, -1]

# Split the training and test data set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=2)

# Scale the values using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Plot the histogram for the attribute values
print("Histogram plots for the Nominal attributes:")
hist = X.hist(figsize=[20,20])
plt.show()

print("Statistics and box plot for the Numerical attributes:")
Data[numerical_columns] = scaler.fit_transform(Data[numerical_columns])
Data.describe()
boxplot = Data[numerical_columns].astype(float).boxplot(figsize=[10,10])
plt.show()

# Create Decision Tree classifer object
dtc = DecisionTreeClassifier(random_state=2)
print("Classifier used - Decision Tree Classifier :")
runClassifer(dtc,X_train,y_train,y_test)

# Create Gaussian Naive Bayes classifer object
gnb = GaussianNB()
print("Classifier used - Gaussian Naive Bayes :")
runClassifer(gnb,X_train,y_train,y_test)

# Create Gradient Boosting classifer object
gbc = GradientBoostingClassifier()
print("Classifier used - Gradient Boosting :")
runClassifer(gbc,X_train,y_train,y_test)

# Create SVM classifer object
svm = SVC(kernel='rbf',random_state=2)
print("Classifier used - SVM :")
runClassifer(svm,X_train,y_train,y_test)
