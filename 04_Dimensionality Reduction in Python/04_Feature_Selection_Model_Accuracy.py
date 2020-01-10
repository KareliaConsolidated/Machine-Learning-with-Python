# Building a diabetes classifier
# You'll be using the Pima Indians diabetes dataset to predict whether a person has diabetes using logistic regression. There are 8 features and one target in this dataset. The data has been split into a training and test set and pre-loaded for you as X_train, y_train, X_test, and y_test.

# A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr.
# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train)

# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print("{0:.1%} accuracy on test set.".format(accuracy_score(y_test, y_pred))) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))
# 79.6% accuracy on test set.
# {'age': 0.34, 'diastolic': 0.03, 'bmi': 0.38, 'glucose': 1.23, 'family': 0.34, 'pregnant': 0.04, 'triceps': 0.24, 'insulin': 0.19}

# Manual Recursive Feature Elimination
# Now that we've created a diabetes classifier, let's see if we can reduce the number of features without hurting the model accuracy too much.
# On the second line of code the features are selected from the original dataframe. Adjust this selection.
# A StandardScaler() instance has been predefined as scaler and a LogisticRegression() one as lr.
# All necessary functions and packages have been pre-loaded too.
# Remove the feature with the lowest model coefficient
X = diabetes_df[['pregnant', 'glucose', 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Remove the 2 features with the lowest model coefficients
X = diabetes_df[['glucose', 'triceps', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print("{0:.1%} accuracy on test set.".format(acc)) 
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))
# 76.5% accuracy on test set.
# {'glucose': 1.27}

# Removing all but one feature only reduced the accuracy by a few percent.

# Automatic Recursive Feature Elimination
# Now let's automate this recursive process. Wrap a Recursive Feature Eliminator (RFE) around our logistic regression estimator and pass it the desired number of features.

# All the necessary functions and packages have been pre-loaded and the features have been scaled for you.
# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
# acc = accuracy_score(y_test, rfe.predict(X_test))
# print("{0:.1%} accuracy on test set.".format(acc)) 
#  Fitting estimator with 8 features.
#     Fitting estimator with 7 features.
#     Fitting estimator with 6 features.
#     Fitting estimator with 5 features.
#     Fitting estimator with 4 features.
#     {'age': 1, 'diastolic': 6, 'bmi': 1, 'glucose': 1, 'family': 2, 'pregnant': 5, 'triceps': 3, 'insulin': 4}
#     Index(['glucose', 'bmi', 'age'], dtype='object')
#     80.6% accuracy on test set.

# Building a random forest model
# You'll again work on the Pima Indians dataset to predict whether an individual has diabetes. This time using a random forest classifier. You'll fit the model on the training data after performing the train-test split and consult the feature importance values.
# The feature and target datasets have been pre-loaded for you as X and y. Same goes for the necessary packages and functions.
# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print("{0:.1%} accuracy on test set.".format(acc))
# The random forest model gets 78% accuracy on the test set and 'glucose' is the most important feature (0.21).

# Random forest for feature selection
# Now lets use the fitted random model to select the most important features from our input dataset X.

# The trained model from the previous exercise has been pre-loaded for you as rf.
# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:,mask]

# prints out the selected column names
print(reduced_X.columns)

# <script.py> output:
#     Index(['glucose', 'age'], dtype='object')

# Recursive Feature Elimination with random forests
# You'll wrap a Recursive Feature Eliminator around a random forest model to remove features step by step. This method is more conservative compared to selecting features after applying a single importance threshold. Since dropping one feature can influence the relative importances of the others.

# You'll need these pre-loaded datasets: X, X_train, y_train.

# Functions and classes that have been pre-loaded for you are: RandomForestClassifier(), RFE(), train_test_split().
# Wrap the feature eliminator around the random forest model
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask using an attribute of rfe
mask = rfe.support_ > 0.15

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:,mask]
print(reduced_X.columns)


# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)