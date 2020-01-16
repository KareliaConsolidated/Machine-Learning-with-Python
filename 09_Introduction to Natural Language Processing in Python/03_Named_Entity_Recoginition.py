# NER with NLTK
# You're now going to have some fun with named-entity recognition! A scraped news article has been pre-loaded into your workspace. Your task is to use nltk to find the named entities in this article.
# What might the article be about, given the names you found?
# Along with nltk, sent_tokenize and word_tokenize from nltk.tokenize have been pre-imported.
# Tokenize the article into sentences: sentences
sentences = sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == 'NE':
            print(chunk)

# Charting practice
# In this exercise, you'll use some extracted named entities and their groupings from a series of newspaper articles to chart the diversity of named entity types in the articles.

# You'll use a defaultdict called ner_categories, with keys representing every named entity group type, and values to count the number of each different named entity type. You have a chunked sentence list called chunked_sentences similar to the last exercise, but this time with non-binary category names.

# You can use hasattr() to determine if each chunk has a 'label' and then simply use the chunk's .label() method as the dictionary key.            
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()

# Introduction to SpaCy
# Comparing NLTK with spaCy NER
# Using the same text you used in the first exercise of this chapter, you'll now see the results using spaCy's NER annotator. How will they compare?

# The article has been pre-loaded as article. To minimize execution times, you'll be asked to specify the keyword arguments tagger=False, parser=False, matcher=False when loading the spaCy model, because you only care about the entity in this exercise.
# Import spacy
import spacy

# Instantiate the English model: nlp
nlp = spacy.load('en',tagger=False, parser=False, matcher=False)

# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ent in doc.ents:
    print(ent.label_, ent.text)

# Multilingual NER with polyglot
# French NER with polyglot I
# In this exercise and the next, you'll use the polyglot library to identify French entities. The library functions slightly differently than spacy, so you'll use a few of the new things you learned in the last video to display the named entity text and category.

# You have access to the full article string in article. Additionally, the Text class of polyglot has been imported from polyglot.text.
# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))

# French NER with polyglot II
# Here, you'll complete the work you began in the previous exercise.

# Your task is to use a list comprehension to create a list of tuples, in which the first element is the entity tag, and the second element is the full string of the entity text.
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)

# Spanish NER with polyglot
# You'll continue your exploration of polyglot now with some Spanish annotation. This article is not written by a newspaper, so it is your first example of a more blog-like text. How do you think that might compare when finding entities?

# The Text object has been created as txt, and each entity has been printed, as you can see in the IPython Shell.

# Your specific task is to determine how many of the entities contain the words "Márquez" or "Gabo" - these refer to the same person in different ways!

# txt.entities is available.
# Calculate the proportion of txt.entities that
# contains 'Márquez' or 'Gabo': prop_ggm

# Initialize the count
count = 0

# Iterate over all the entities
for entity in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in entity or "Gabo" in entity:
        # Increment count
        count += 1

# Calculate the proportion of entities counted: prop_ggm
prop_ggm = count / len(txt.entities)
print(prop_ggm)