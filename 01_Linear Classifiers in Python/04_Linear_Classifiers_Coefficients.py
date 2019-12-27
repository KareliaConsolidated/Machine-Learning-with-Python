# Changing the model coefficients
# When you call fit with scikit-learn, the logistic regression coefficients are automatically learned from your dataset. In this exercise you will explore how the decision boundary is represented by the coefficients. To do so, you will change the coefficients manually (instead of with fit), and visualize the resulting classifiers.

# A 2D dataset is already loaded into the environment as X and y, along with a linear classifier object model.
# Set the coefficients
model.coef_ = np.array([[0,1]])
model.intercept_ = np.array([0])
a=np.array([[1.78,0.43]])
b=np.array([0.43])
# Plot the data and decision boundary
plot_classifier(a,b,model)

# Print the number of errors
num_err = np.sum(y != model.predict(X))
print("Number of errors:", num_err)