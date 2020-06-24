# Disaster Response Pipeline Project

I created a disaster response pipeline to learn from existing data and categorize new messages to appropriate categories.

### Folder and Files
Folder - APP:
    Folder - templates:
        go.html - result page.
        master.html - home page.
    run.py - python script that loads database from data folder and model from models folder. Then it generate to webpage on a local host.

Folder - data:
    disaster_categories.csv - file containing categorized data.
    disaster_messages.csv - file containing raw messages.
    DisasterResponse.db - database containing processed data for machine learning.
    process_data.py - script that cleans and tokenizes data.

Folder - models:
    classifier.pkl - pickle file generated after machine learning process. It contains the model.
    train_classifier.py - script that loads database from data folder and performs machine learning.

README.md - this file
screenshot1.png - a screenshot of the local web page

### Required libraries
nltk 3.3.0+
numpy 1.15.2+
pandas 0.23.4+
scikit-learn 0.20.0+
sqlalchemy 1.2.12+

### Data Source

This is a project guided by Udacity. The data came from Figure Eight.