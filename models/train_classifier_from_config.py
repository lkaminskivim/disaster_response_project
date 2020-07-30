import yaml
from models.train_classifier import prepare_classifier


def main():
    with open("config.yml", "r") as ymlfile:
        conf = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    database_filepath = conf["output"]["database_filepath"]
    model_filepath = conf["output"]["model_filepath"]
    prepare_classifier(database_filepath, model_filepath)


if __name__ == '__main__':
    main()
