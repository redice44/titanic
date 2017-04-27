import pandas as pd
import numpy as np
import random as rnd
import math

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Utility to create common charts with sum, size, and mean values.
def chartFeatures(dataFrame, features, group, showLog=False):
  chart = dataFrame[features].groupby(group).agg([np.sum, np.size, np.mean])
  if showLog:
    print(chart)
    print('\n\n')

  return chart

# Formats common Json output
def formatJson(out):
  out = out.replace('\"[\"Survived\",', '')
  out = out.replace(']\"', '')
  out = out.replace('\"sum\"', '\"Survived\"')
  out = out.replace('\"size\"', '\"Total\"')
  return out

# Saves DataFrame as .csv and .json
def saveFiles(df, fileName):
  path = './data/'
  df.to_csv(path + fileName + '.csv')
  out = df.to_json(orient='index')
  out = formatJson(out)
  with open(path + fileName + '.json', 'w') as f:
    f.write(out)

  return 

def preprocessSex(df, saveFile=False, printLog=False):
  df['Sex'] = df['Sex'].map( { 'female': 0, 'male': 1 }).astype(int)

  if printLog:
    print(chartFeatures(df, ['Sex', 'Survived'], ['Sex']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['Sex', 'Survived'], ['Sex']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-sex')

  return df

def preprocessClass(df, saveFile=False, printLog=False):
  if printLog:
    print(chartFeatures(df, ['Pclass', 'Survived'], ['Pclass']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['Pclass', 'Survived'], ['Pclass']).sort_values(by=[('Survived', 'mean')], ascending=False),'feature-class')

  return df

def preprocessSexClass(df, saveFile=False, printLog=False):
  df['SexClass'] = (df['Sex'] * 3) + df['Pclass']
  df['SexClass'] = df['SexClass'].astype(int)

  if printLog:
    print(chartFeatures(df, ['SexClass', 'Survived'], ['SexClass']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['SexClass', 'Survived'], ['SexClass']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-sexclass')

  return df

def fillAge(df):
  sexClassMean = np.zeros((6))
  for i in range(0,6):
    temp_df = df[df['SexClass'] == i+1]['Age'].dropna()
    sexClassMean[i] = int(temp_df.mean())

  for i in range(0, 6):
    df.loc[ ((df.Age.isnull()) & (df.SexClass == (i+1))), 'Age'] = sexClassMean[i]
  df['Age'] = df['Age'].astype(int)

  return df

def preprocessAge(df, saveFile=False, printLog=False):
  if saveFile:
    saveFiles(chartFeatures(df, ['Age', 'Survived'], ['Age']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-age')

  df = fillAge(df)
  freq = df['Age'].mode()[0]
  size = df[df['Age'] == freq]['Age'].count()
  cutSize = int(math.floor(df['Age'].count() / size) + 1)

  df['AgeRange'] = pd.cut(df['Age'], cutSize)
  df['AgeFreq'] = pd.qcut(df['Age'], cutSize)

  if printLog:
    print(chartFeatures(df, ['AgeRange', 'Survived'], ['AgeRange']).sort_values(by=[('Survived', 'mean')], ascending=False))
    print(chartFeatures(df, ['AgeFreq', 'Survived'], ['AgeFreq']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['AgeRange', 'Survived'], ['AgeRange']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-agerange-pre')
    saveFiles(chartFeatures(df, ['AgeFreq', 'Survived'], ['AgeFreq']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-agefreq-pre')

  catlen = len(df['AgeRange'].cat.categories)
  df['AgeRange'].cat.categories = [str(i) for i in range(0, catlen)]
  df['AgeRange'] = df['AgeRange'].astype(int)

  catlen = len(df['AgeFreq'].cat.categories)
  df['AgeFreq'].cat.categories = [str(i) for i in range(0, catlen)]
  df['AgeFreq'] = df['AgeFreq'].astype(int)

  if printLog:
    print(chartFeatures(df, ['AgeRange', 'Survived'], ['AgeRange']).sort_values(by=[('Survived', 'mean')], ascending=False))
    print(chartFeatures(df, ['AgeFreq', 'Survived'], ['AgeFreq']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['AgeRange', 'Survived'], ['AgeRange']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-agerange-post')
    saveFiles(chartFeatures(df, ['AgeFreq', 'Survived'], ['AgeFreq']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-agefreq-post')

  return df

def fillFare(df):
  sexClassMean = np.zeros((6))
  for i in range(0,6):
    temp_df = df[df['SexClass'] == i+1]['Fare'].dropna()
    sexClassMean[i] = int(temp_df.mean())

  for i in range(0, 6):
    df.loc[ ((df.Fare.isnull()) & (df.SexClass == (i+1))), 'Fare'] = sexClassMean[i]
  df['Fare'] = df['Fare'].astype(int)

  return df

def preprocessFare(df, saveFile=False, printLog=False):
  df = fillFare(df)
  freq = df['Fare'].mode()[0]
  size = df[df['Fare'] == freq]['Fare'].count()
  cutSize = int(math.floor(df['Fare'].count() / size) + 1)

  df['FareRange'] = pd.cut(df['Fare'], cutSize)
  df['FareFreq'] = pd.qcut(df['Fare'], cutSize)

  if printLog:
    print(chartFeatures(df, ['FareRange', 'Survived'], ['FareRange']).sort_values(by=[('Survived', 'mean')], ascending=False))
    print(chartFeatures(df, ['FareFreq', 'Survived'], ['FareFreq']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['FareRange', 'Survived'], ['FareRange']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-farerange-pre')
    saveFiles(chartFeatures(df, ['FareFreq', 'Survived'], ['FareFreq']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-farefreq-pre')

  catlen = len(df['FareRange'].cat.categories)
  df['FareRange'].cat.categories = [str(i) for i in range(0, catlen)]
  df['FareRange'] = df['FareRange'].astype(int)

  catlen = len(df['FareFreq'].cat.categories)
  df['FareFreq'].cat.categories = [str(i) for i in range(0, catlen)]
  df['FareFreq'] = df['FareFreq'].astype(int)

  if printLog:
    print(chartFeatures(df, ['FareRange', 'Survived'], ['FareRange']).sort_values(by=[('Survived', 'mean')], ascending=False))
    print(chartFeatures(df, ['FareFreq', 'Survived'], ['FareFreq']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['FareRange', 'Survived'], ['FareRange']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-farerange-post')
    saveFiles(chartFeatures(df, ['FareFreq', 'Survived'], ['FareFreq']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-farefreq-post')

  return df

def preprocessSibSp(df, saveFile=False, printLog=False):
  if printLog:
    print(chartFeatures(df, ['SibSp', 'Survived'], ['SibSp']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['SibSp', 'Survived'], ['SibSp']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-sibsp')

  return df

def preprocessParch(df, saveFile=False, printLog=False):
  if printLog:
    print(chartFeatures(df, ['Parch', 'Survived'], ['Parch']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['Parch', 'Survived'], ['Parch']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-parch')
    
  return df

def preprocessFamily(df, saveFile=False, printLog=False):
  df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
  df['FamilySize'] = df['FamilySize'].astype(int)

  if printLog:
    print(chartFeatures(df, ['FamilySize', 'Survived'], ['FamilySize']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['FamilySize', 'Survived'], ['FamilySize']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-family')

  return df

def preprocessEmbark(df, saveFile=False, printLog=False):
  df['Embarked'] = df['Embarked'].fillna(df.Embarked.dropna().mode()[0])
  df['Embarked'] = df['Embarked'].map( { 'C': 0, 'Q': 1, 'S': 2 }).astype(int)

  if printLog:
    print(chartFeatures(df, ['Embarked', 'Survived'], ['Embarked']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['Embarked', 'Survived'], ['Embarked']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-embarked')

  return df

def preprocessTitle(df, saveFile=False, printLog=False):
  df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
  df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')

  df['Title'] = df['Title'].replace(to_replace='^[^M].*|Maj.*', regex=True, value='Other')
  df['Title'] = df['Title'].map({ 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Other': 4, 'Mr': 5 })
  df['Title'] = df['Title'].astype(int)

  if printLog:
    print(chartFeatures(df, ['Title', 'Survived'], ['Title']).sort_values(by=[('Survived', 'mean')], ascending=False))

  if saveFile:
    saveFiles(chartFeatures(df, ['Title', 'Survived'], ['Title']).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-title')

  return df

def preprocessDrops(df, printLog=False):
  df = df.drop(['Name', 'Ticket', 'Cabin', 'AgeRange', 'FareRange', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Pclass'], axis=1)

  if printLog:
    print(df.columns.values)
    
  return df

def trainWithModel(X_train, Y_train, X_test, model):
  model.fit(X_train, Y_train)
  return [round(model.score(X_train, Y_train) * 100, 2), model.predict(X_test)]

def testModels(dfs, saveFile=False, printLog=False):
  training_df = dfs[0]
  testing_df = dfs[1]
  X_train = training_df.drop(['Survived', 'PassengerId'], axis=1)
  Y_train = training_df['Survived']

  X_test = testing_df.drop('PassengerId', axis=1)

  models = list(map(lambda model: trainWithModel(X_train, Y_train, X_test, model), [SVC(), KNeighborsClassifier(n_neighbors=3), KNeighborsClassifier(n_neighbors=4), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=7), GaussianNB(), Perceptron(), LinearSVC(), SGDClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5,5), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,2,2), random_state=1), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8,8), random_state=1)]))

  modelsAcc = pd.DataFrame({
    'Model': ['Support Vector Machine', 'KNN-3', 'KNN-4', 'KNN-5', 'KNN-7', 'Naive Bayes', 'Perceptron', 'Linear SVC', 'Stochastic Gradient Descent', 'Decision Tree', 'Random Forest', 'Neural Network (12,2)', 'Neural Network (5,5,5)', 'Neural Network (15,2)', 'Neural Network (8,2)', 'Neural Network (8,5)', 'Neural Network (12,2,2)'],
    'Score': list(map(lambda results: results[0], models)),
    })

  bestModelIndex = modelsAcc.sort_values(by='Score', ascending=False).head(1).index.values[0]
  
  if printLog:
    print(modelsAcc.sort_values(by='Score', ascending=False))

  if saveFile:
    saveFiles(modelsAcc.sort_values(by='Score', ascending=False), 'models-score')

  return [models, modelsAcc]

def makeCrossSection(df, crossSize):
  minRange = df['PassengerId'].min()
  maxRange = df['PassengerId'].max() + 1
  selection = rnd.sample(range(minRange, maxRange), crossSize)

  return (df[~df['PassengerId'].isin(selection)], df[df['PassengerId'].isin(selection)].drop(['Survived'], axis=1))


# Initialize Datasets
training_features = pd.read_csv('./data/train.csv')
testing_features = pd.read_csv('./data/test.csv')

# Get some basic info about the dataset
training_features.head(5).to_csv('./data/training_features-5.csv')
print(training_features.info())
training_features.describe(include='all').to_csv('./data/training_features-describe.csv')

# Create a list of all features' survival rates
featureCharts = [
  chartFeatures(training_features, features=['Sex', 'Survived'], group=['Sex']),
  chartFeatures(training_features, features=['Pclass', 'Survived'], group=['Pclass']),
  chartFeatures(training_features, features=['SibSp', 'Survived'], group=['SibSp']),
  chartFeatures(training_features, features=['Parch', 'Survived'], group=['Parch']),
  chartFeatures(training_features, features=['Embarked', 'Survived'], group=['Embarked'])
]

# Renaming Pclass Feature
featureCharts[1] = featureCharts[1].rename(index={1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

# Renaming SibSp Feature
featureCharts[2] = featureCharts[2].rename(index={0: '0-SibSp', 1: '1-SibSp', 2: '2-SibSp', 3: '3-SibSp', 4: '4-SibSp', 5: '5-SibSp', 8: '8-SibSp'})

# Renaming Parch Feature
featureCharts[3] = featureCharts[3].rename(index={0: '0-Parch', 1: '1-Parch', 2: '2-Parch', 3: '3-Parch', 4: '4-Parch', 5: '5-Parch', 6: '6-Parch'})

# Renaming Embarked Feature
featureCharts[4] = featureCharts[4].rename(index={'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})

saveFiles(pd.concat(featureCharts).sort_values(by=[('Survived', 'mean')], ascending=False), 'feature-all')

# Preprocess the data
training_features = preprocessSex(training_features, saveFile=True)
testing_features = preprocessSex(testing_features)
training_features = preprocessClass(training_features, saveFile=True)
testing_features = preprocessClass(testing_features)
training_features = preprocessSexClass(training_features, saveFile=True)
testing_features = preprocessSexClass(testing_features)
training_features = preprocessAge(training_features, saveFile=True)
testing_features = preprocessAge(testing_features)
training_features = preprocessFare(training_features, saveFile=True)
testing_features = preprocessFare(testing_features)
training_features = preprocessSibSp(training_features, saveFile=True)
testing_features = preprocessSibSp(testing_features)
training_features = preprocessParch(training_features, saveFile=True)
testing_features = preprocessParch(testing_features)
training_features = preprocessFamily(training_features, saveFile=True)
testing_features = preprocessFamily(testing_features)
training_features = preprocessEmbark(training_features, saveFile=True)
testing_features = preprocessEmbark(testing_features)
training_features = preprocessTitle(training_features, saveFile=True)
testing_features = preprocessTitle(testing_features)
training_features = preprocessDrops(training_features, printLog=True)
testing_features = preprocessDrops(testing_features)

# Train and test classifiers on full dataset
fullValidation = testModels([training_features, testing_features], saveFile=True, printLog=True)

# Cross Validation 
crossValidations = [
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300)),
  testModels(makeCrossSection(training_features, 300))
]

# Get the mean of all of the cross validations
print(pd.concat(list(map(lambda s: s[1], crossValidations))).groupby(['Model']).agg([np.mean]).sort_values(by=[('Score', 'mean')], ascending=False))
saveFiles(pd.concat(list(map(lambda s: s[1], crossValidations))).groupby(['Model']).agg([np.mean]).sort_values(by=[('Score', 'mean')], ascending=False), 'cross-validation')

# Create Submission File
# Index of the classifer to submit
classifier = 13
submission = pd.DataFrame({
  "PassengerId": testing_features["PassengerId"],
  "Survived": fullValidation[0][classifier][1]
})
submission.to_csv('./data/submission.csv', index=False)