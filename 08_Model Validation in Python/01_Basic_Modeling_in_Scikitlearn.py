# Seen vs. unseen data
# Model's tend to have higher accuracy on observations they have seen before. In the candy dataset, predicting the popularity of Skittles will likely have higher accuracy than predicting the popularity of Andes Mints; Skittles is in the dataset, and Andes Mints is not.
# You've built a model based on 50 candies using the dataset X_train and need to report how accurate the model is at predicting the popularity of the 50 candies the model was built on, and the 35 candies (X_test) it has never seen. You will use the mean absolute error, mae(), as the accuracy metric.
# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))

# Set parameters and fit a model
# Predictive tasks fall into one of two categories: regression or classification. In the candy dataset, the outcome is a continuous variable describing how often the candy was chosen over another candy in a series of 1-on-1 match-ups. To predict this value (the win-percentage), you will use a regression model.

# In this exercise, you will specify a few parameters using a random forest regression model rfr.

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)

# Feature importances
# Although some candy attributes, such as chocolate, may be extremely popular, it doesn't mean they will be important to model prediction. After a random forest model has been fit, you can review the model's attribute, .feature_importances_, to see which variables had the biggest impact. You can check how important each variable was in the model by looping over the feature importance array using enumerate().
# If you are unfamiliar with Python's enumerate() function, it can loop over a list while also creating an automatic counter.
# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))

#<script.py> output:
# chocolate: 0.44
# fruity: 0.03
# caramel: 0.02
# peanutyalmondy: 0.05
# nougat: 0.01
# crispedricewafer: 0.03
# hard: 0.01
# bar: 0.02
# pluribus: 0.02
# sugarpercent: 0.17
# pricepercent: 0.19

# Classification predictions
# In model validation, it is often important to know more about the predictions than just the final classification. When predicting who will win a game, most people are also interested in how likely it is a team will win.

# Probability	Prediction	Meaning
# 0 < .50	0	Team Loses
# .50 +	1	Team Wins
# In this exercise, you look at the methods, .predict() and .predict_proba() using the tic_tac_toe dataset. The first method will give a prediction of whether Player One will win the game, and the second method will provide the probability of Player One winning. Use rfc as the random forest classification model.
# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))

# Reusing model parameters
# Replicating model performance is vital in model validation. Replication is also important when sharing models with co-workers, reusing models on new data or asking questions on a website such as Stack Overflow. You might use such a site to ask other coders about model errors, output, or performance. The best way to do this is to replicate your work by reusing model parameters.

# In this exercise, you use various methods to recall which parameters were used in a model.
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))

# Random forest classifier
# This exercise reviews the four modeling steps discussed throughout this chapter using a random forest classification model. You will:

# Create a random forest classification model.
# Fit the model using the tic_tac_toe dataset.
# Make predictions on whether Player One will win (1) or lose (0) the current game.
# Finally, you will evaluate the overall accuracy of the model.
# Let's get started!
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])

# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))
# That's all the steps! Notice the first five predictions were all 1, indicating that Player One is predicted to win all five of those games. You also see the model accuracy was only 82%.