from sqlalchemy import create_engine
import pandas as pd
import yaml


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


def main():
    with open("config.yml", "r") as ymlfile:
        conf = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    messages_filepath = conf["input"]["messages_csv_filepath"]
    categories_filepath = conf["input"]["categories_csv_filepath"]
    database_filepath = conf["output"]["database_filepath"]

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print(df.shape)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
