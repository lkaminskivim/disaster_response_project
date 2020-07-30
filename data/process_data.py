import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(left=messages, right=categories, left_on='id', right_on='id', how='outer')

    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[1]
    category_colnames = row.replace(regex=r'(-\d)', value='')
    categories.columns = category_colnames
    categories = categories.replace(regex='([a-z_]+-)', value='').astype(int)
    df.drop('categories', 1, inplace=True)
    return pd.concat([df, categories], axis=1, join='outer')


def clean_data(df):
    return df.drop_duplicates()


def save_data(df, database_filename):
    engine = create_engine("sqlite:///{0}".format(database_filename))
    df.to_sql('MessageTable', engine, index=False)


def prepare_database(messages_filepath, categories_filepath, database_filepath):
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print(df.shape)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        prepare_database(messages_filepath, categories_filepath, database_filepath)

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
