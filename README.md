# Disaster Response Pipeline Project

This is project developed for Data Scientist training from Udacity

### Dependencies

* sqlalchemy
* scikitlearn
* nltk
* plotly
* yaml

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database `data/process_data.py`
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py data/DisasterResponse.db models/classifier.pkl`

3. Go to http://0.0.0.0:3001/

