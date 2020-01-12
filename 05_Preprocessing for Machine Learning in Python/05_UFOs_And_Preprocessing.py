# Checking column types
# Take a look at the UFO dataset's column types using the dtypes attribute. Two columns jump out for transformation: the seconds column, which is a numeric column but is being read in as object, and the date column, which can be transformed into the datetime type. That will make our feature engineering efforts easier later on.
# Check the column types
print(ufo.dtypes)

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype(float)

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo[["seconds", "date"]].dtypes)

# Dropping missing data
# Let's remove some of the rows where certain columns have missing values. We're going to look at the length_of_time column, the state column, and the type column. If any of the values in these columns are missing, we're going to drop the rows.
# Check how many values are missing in the length_of_time, state, and type columns
print(ufo[['length_of_time', 'state', 'type']].isnull().sum())

# Keep only rows where length_of_time, state, and type are not null
ufo_no_missing = ufo[ufo['length_of_time'].notnull() & 
          ufo['state'].notnull() & 
          ufo['type'].notnull()]

# Print out the shape of the new dataset
print(ufo_no_missing.shape)

# Extracting numbers from strings
# The length_of_time field in the UFO dataset is a text field that has the number of minutes within the string. Here, you'll extract that number from that text field using regular expressions.
def return_minutes(time_string):

    # Use \d+ to grab digits
    pattern = re.compile(r"\d+")
    
    # Use match on the pattern and column
    num = re.match(pattern, time_string)
    if num is not None:
        return int(num.group(0))
        
# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(lambda x:return_minutes(x))

# Take a look at the head of both of the columns
print(ufo[['length_of_time','minutes']].head())

# Identifying features for standardization
# In this section, you'll investigate the variance of columns in the UFO dataset to determine which features should be standardized. After taking a look at the variances of the seconds and minutes column, you'll see that the variance of the seconds column is extremely high. Because seconds and minutes are related to each other (an issue we'll deal with when we select features for modeling), let's log normlize the seconds column.
# Check the variance of the seconds and minutes columns
print(ufo[['seconds','minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo["seconds_log"].var())

# Encoding categorical variables
# There are couple of columns in the UFO dataset that need to be encoded before they can be modeled through scikit-learn. You'll do that transformation here, using both binary and one-hot encoding methods.
# Use Pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda x: 1 if x == 'us' else 0)

# Print the number of unique type values
print(len(ufo['type'].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)

# Features from dates
# Another feature engineering task to perform is month and year extraction. Perform this task on the date column of the ufo dataset.
# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].apply(lambda x: x.month)

# Extract the year from the date column
ufo["year"] = ufo["date"].apply(lambda x: x.year)

# Take a look at the head of all three columns
print(ufo[['date','month','year']].head())

# Text vectorization
# Let's transform the desc column in the UFO dataset into tf/idf vectors, since there's likely something we can learn from this field.
# Take a look at the head of the desc field
print(ufo['desc'].head())

# Create the tfidf vectorizer object
vec = TfidfVectorizer()

# Use vec's fit_transform method on the desc field
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns this creates
print(desc_tfidf.shape)
# <script.py> output:
#     0    It was a large&#44 triangular shaped flying ob...
#     1    Dancing lights that would fly around and then ...
#     2    Brilliant orange light or chinese lantern at o...
#     3    Bright red light moving north to north west fr...
#     4    North-east moving south-west. First 7 or so li...
#     Name: desc, dtype: object
#     (1866, 3422)

# Feature selection and modeling

# Selecting the ideal dataset
# Let's get rid of some of the unnecessary features. Because we have an encoded country column, country_enc, keep it and drop other columns related to location: city, country, lat, long, state.
# We have columns related to month and year, so we don't need the date or recorded columns.
# We vectorized desc, so we don't need it anymore. For now we'll keep type.
# We'll keep seconds_log and drop seconds and minutes.
# Let's also get rid of the length_of_time column, which is unnecessary after extracting minutes.
# Check the correlation between the seconds, seconds_log, and minutes columns
print(ufo[['seconds','seconds_log','minutes']].corr())

# Make a list of features to drop
to_drop = ['city','country','lat','long','state','date','recorded','desc','seconds','minutes','length_of_time']

# Drop those features
ufo_dropped = ufo.drop(to_drop,axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)

# Modeling the UFO dataset, part 1
# In this exercise, we're going to build a k-nearest neighbor model to predict which country the UFO sighting took place in. Our X dataset has the log-normalized seconds column, the one-hot encoded type columns, as well as the month and year when the sighting took place. The y labels are the encoded country column, where 1 is us and 0 is ca.
# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets using train_test_split, setting stratify=y
train_X, test_X, train_y, test_y = train_test_split(X,y,stratify=y)

# Fit knn to the training sets
knn.fit(train_X, train_y)

# Print the score of knn on the test sets
print(knn.score(test_X,test_y))

# Modeling the UFO dataset, part 2
# Finally, let's build a model using the text vector we created, desc_tfidf, using the filtered_words list to create a filtered text vector. Let's see if we can predict the type of the sighting based on the text. We'll use a Naive Bayes model for this.
# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y 
train_X, test_X, train_y, test_y = train_test_split(filtered_text.toarray(), y, stratify=y)

# Fit nb to the training sets
nb.fit(train_X,train_y)

# Print the score of nb on the test sets
nb.score(test_X, test_y)