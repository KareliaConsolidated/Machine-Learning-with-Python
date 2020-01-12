# Missing data - columns
# We have a dataset comprised of volunteer information from New York City. The dataset has a number of features, but we want to get rid of features that have at least 3 missing values.

# How many features are in the original dataset, and how many features are in the set after columns with at least 3 missing values are removed?

# The dataset volunteer has been provided.
# Use the dropna() function to remove columns.
# You'll have to set both the axis= and thresh= parameters.
volunteer.info()
volunteer.dropna(axis=1, thresh=3)

# Missing data - rows
# Taking a look at the volunteer dataset again, we want to drop rows where the category_desc column values are missing. We're going to do this using boolean indexing, by checking to see if we have any null values, and then filtering the dataset so that we only have rows with those values.
# Check how many values are missing in the category_desc column
print(volunteer['category_desc'].isna().sum())

# Subset the volunteer dataset
volunteer_subset = volunteer[volunteer['category_desc'].notnull()]

# Print out the shape of the subset
print(volunteer_subset.shape)

# Working with data types
# Converting a column type
# If you take a look at the volunteer dataset types, you'll see that the column hits is type object. But, if you actually look at the column, you'll see that it consists of integers. Let's convert that column to type int.
# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype('int')

# Look at the dtypes of the dataset
print(volunteer.dtypes)

# Class distribution
# Class imbalance
# In the volunteer dataset, we're thinking about trying to predict the category_desc variable using the other features in the dataset. First, though, we need to know what the class distribution (and imbalance) is for that label.

# Which descriptions occur less than 50 times in the volunteer dataset?

# The dataset volunteer has been provided.
# The colum you want to check is category_desc.
# Use the value_counts() method to check variable counts.
volunteer.category_desc.value_counts()
# Out[1]: 
# Strengthening Communities    307
# Helping Neighbors in Need    119
# Education                     92
# Health                        52
# Environment                   32
# Emergency Preparedness        15
# Name: category_desc, dtype: int64

# Stratified sampling
# We know that the distribution of variables in the category_desc column in the volunteer dataset is uneven. If we wanted to train a model to try to predict category_desc, we would want to train the model on a sample of data that is representative of the entire dataset. Stratified sampling is a way to achieve this.
# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())