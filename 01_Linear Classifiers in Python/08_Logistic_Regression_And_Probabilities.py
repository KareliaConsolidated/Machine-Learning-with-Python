# Regularization and probabilities
# In this exercise, you will observe the effects of changing the regularization strength on the predicted probabilities.
# A 2D binary classification dataset is already loaded into the environment as X and y.
# Set the regularization strength
X = np.array([[ 1.78862847,  0.43650985], [ 0.09649747, -1.8634927 ], [-0.2773882 , -0.35475898], [-3.08274148,  2.37299932], [-3.04381817,  2.52278197], [-1.31386475,  0.88462238], [-2.11868196,  4.70957306], [-2.94996636,  2.59532259], [-3.54535995,  1.45352268], [ 0.98236743, -1.10106763], [-1.18504653, -0.2056499 ], [-1.51385164,  3.23671627], [-4.02378514,  2.2870068 ], [ 0.62524497, -0.16051336], [-3.76883635,  2.76996928], [ 0.74505627,  1.97611078], [-1.24412333, -0.62641691], [-0.80376609, -2.41908317], [-0.92379202, -1.02387576], [ 1.12397796, -0.13191423]]) model = LogisticRegression(C=1)
y = np.array([-1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1])
# Set the regularization strength
model = LogisticRegression(C=1)

# Fit and plot
model.fit(X,y)
plot_classifier(X,y,model,proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))

# Set the regularization strength
model = LogisticRegression(C=0.1)

# Fit and plot
model.fit(X,y)
plot_classifier(X,y,model,proba=True)

# Predict probabilities on training points
prob = model.predict_proba(X)
print("Maximum predicted probability", np.max(prob))
# <script.py> output:
#     Maximum predicted probability 0.9761229966765974

# <script.py> output:
#     Maximum predicted probability 0.8990965659596716
# smaller values of C lead to less confident predictions. That's because smaller C means more regularization, which in turn means smaller coefficients, which means raw model outputs closer to zero and, thus, probabilities closer to 0.5 after the raw model output is squashed through the sigmoid function. That's quite a chain of events!

# Visualizing easy and difficult examples
# In this exercise, you'll visualize the examples that the logistic regression model is most and least confident about by looking at the largest and smallest predicted probabilities.
# The handwritten digits dataset is already loaded into the variables X and y. The show_digit function takes in an integer index and plots the corresponding image, with some extra information displayed above the image.
lr = LogisticRegression()
lr.fit(X,y)

# Get predicted probabilities
proba = lr.predict_proba(X)

# Sort the example indices by their maximum probability
proba_inds = np.argsort(np.max(proba,axis=1))

# Show the most confident (least ambiguous) digit
show_digit(proba_inds[-1], lr)

# Show the least confident (most ambiguous) digit
show_digit(proba_inds[0], lr)
# the least confident example looks like a weird 4, and the most confident example looks like a very typical 0.