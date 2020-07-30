import yaml
from data.process_data import prepare_database


def main():
    try:
        with open("config.yml", "r") as ymlfile:
            conf = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        messages_filepath = conf["input"]["messages_csv_filepath"]
        categories_filepath = conf["input"]["categories_csv_filepath"]
        database_filepath = conf["output"]["database_filepath"]

        prepare_database(messages_filepath, categories_filepath, database_filepath)
    except FileNotFoundError:
        print("Could not proceede because config.yml is missing.")


if __name__ == '__main__':
    main()
