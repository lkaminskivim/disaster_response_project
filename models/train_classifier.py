import sys
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('MessageTable', engine)
    x = df['message']
    y = df[df.columns[4:]]
    return x, y, y.columns


url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # build pipeline
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])


def evaluate_model(model, x_test, y_test, category_names):
    y_pred = pd.DataFrame(model.predict(x_test), columns=category_names)
    for col in y_test.columns:
        print(classification_report(y_true=y_test[col].values, y_pred=y_pred[col].values))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def prepare_classifier(database_filepath, model_filepath):
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    x, y, category_names = load_data(database_filepath)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(x_train, y_train)

    print('Evaluating model...')
    evaluate_model(model, x_test, y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        prepare_classifier(database_filepath, model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
