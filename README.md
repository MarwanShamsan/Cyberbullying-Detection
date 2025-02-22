# Cyberbullying Detection
## Project Overview
This project focuses on detecting cyberbullying in textual data using Natural Language Processing (NLP), machine learning. The system processes social media text, detects harmful or profane content, and classifies it into relevant categories.

## Packages Used
### NLP & Text Processing:
nltk – Tokenization, stopword removal, POS tagging, and lemmatization
textblob – Sentiment analysis for cyberbullying detection
gensim.models – Word2Vec embeddings for textual representation
better_profanity – Detecting and filtering profane language
hatesonar – Hate speech detection

### Spelling & Correction:
spellchecker – Standard spell-checking algorithm
symspellpy – Fast spelling correction with SymSpell
autocorrect – Context-based spelling correction
fuzzywuzzy, strsimpy – String similarity and fuzzy matching

### Machine Learning Models:
sklearn.linear_model.LogisticRegression – Logistic Regression
sklearn.svm.SVC – Support Vector Machine
sklearn.naive_bayes.MultinomialNB – Naïve Bayes for text classification
sklearn.ensemble.RandomForestClassifier – Random Forest Classifier
xgboost.XGBClassifier – XGBoost
sklearn.tree.DecisionTreeClassifier – Decision Trees

### Dataset Handling & Visualization:
pandas – Data management and preprocessing
seaborn, matplotlib.pyplot – Data visualization and analysis

### Web Application:
gradio – Web-based GUI for interactive cyberbullying detection


## Approaches Used
### 1. Preprocessing
The preprocessing step ensures that the textual data is clean, structured, and ready for analysis. The following transformations were applied:

#### Text Cleaning:
Removal of special characters, punctuation, URLs, and unnecessary whitespace.
Contractions Expansion: Expanding words like "don't" to "do not" using contractions.

#### Tokenization: 
Splitting sentences into individual words using nltk.word_tokenize().

#### Stopword Removal: 
Eliminating common words (e.g., "the", "is") using nltk.corpus.stopwords.

#### Lemmatization: 
Converting words to their base form (e.g., "running" → "run") using WordNetLemmatizer().

#### Spelling Correction:
Applied correction techniques for the dataset using SpellChecker, SymSpell, and Autocorrect.

#### Profanity Filtering: 
Identified and removed offensive words using better_profanity and hatesonar.

#### Feature Extraction:
TF-IDF Vectorization: Converts text into numerical vectors using TfidfVectorizer().

#### Word2Vec Embeddings:
Creates word relationships using gensim.models.Word2Vec().

### 2. Model Training
After preprocessing, the dataset was split into training and testing sets using train_test_split(). The following machine learning models were trained:

#### Logistic Regression:A simple linear model to classify text.

#### Support Vector Machine (SVM): Used for high-dimensional text classification.

#### Naïve Bayes (MultinomialNB): Well-suited for text classification tasks.

#### Random Forest: A tree-based model for better generalization.

#### XGBoost: Boosted decision trees for improved accuracy.

#### Decision Tree: A simpler tree-based classification model.

To improve performance, RandomizedSearchCV and GridSearchCV were used for hyperparameter tuning.

### 3. Model Evaluation
The trained models were evaluated using:

#### Accuracy Score: Measures the proportion of correctly classified texts.

#### Classification Report: Includes precision, recall, and F1-score for each class.

Confusion Matrix: Displays how well the model distinguishes between cyberbullying and non-cyberbullying text.
Additionally, visualizations such as word clouds and bar plots were generated using matplotlib and seaborn to analyze the dataset further.

### 4. Web Application
A Gradio-based web UI was developed to allow users to input text and receive real-time cyberbullying detection results. This provides a user-friendly interface for easy interaction.

### Future Improvements
Deep learning models (BERT, LSTMs) for more robust classification
Multi-language support for broader applicability
Real-time social media monitoring and reporting tools






