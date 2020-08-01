# Disaster Response Pipeline:
## Motivation:
The disaster response pipeline project is done as a part of Udemy’s Data Scientist Nano Degree.  In this project, a code is written to initiate a web app which an emergency operators could exploit during a disaster, to classify a disaster text message into many categories (36) which then can be sent to the body incharge
## Introduction:
This project is a part of my Udacity Data Scientist Nano Degree. The project has three components:
1)ETL Pipeline: 
->Extracting, loading, and transforming data set using Pandas
->Storing in the SQL Lite database. 
2) ML Pipeline: 
->Extracting the data set
->Dividing the data set to train and test data sets
-> Building a text processing and machine learning pipeline
->Training and tuning a model using GridSearchCV
->Outputting the results on the test data set
->Exporting the data set to a pickle file
3) Flask Webapp
->File paths are changed according to the requirements
-> Adding data visualizations using Plotly in the web app.
## Technologies and libraries used in the project:
•	Python 3.7
•	Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
•	Natural Language Process Libraries: NLTK
•	SQLlite Database Libraqries: SQLalchemy
•	Web App and Data Visualization: Flask, Plotly
•	Model Loading and Saving Library: Pickle
### File Description:
•	train_classifier.py: This code trains the ML model with the SQL data base
•	ETL Pipeline Preparation.ipynb: process_data.py development procces
•	data: This folder contains sample messages and categories datasets in csv format.
•	app: cointains the run.py to iniate the web app.
•	ML Pipeline Preparation.ipynb: train_classifier.py. development procces
•	process_data.py: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# Results
The text messages when entered in the dialogue box of the trained classifier can be seen classified by running this application.

