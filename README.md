# Disaster Response Pipeline Project
A model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. A machine learning pipeline to categorize these events were created so that we can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

### Table of Contents

1. [Installation](#installation)
2. [File Descriptions](#files)
3. [Instructions](#Instructions)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

No necessary libraries to run the code except the ones available as part of the Anaconda distribution of Python.

## File Descriptions <a name="files"></a>

There are 3 main folders as follows:
    1. app:
        - run.py: Flask app 
        - templates:
            -master.html: index web page.
            -go.html: classification result page of web app
    2. data:
        - disaster_categories.csv:  data to process 
        - disaster_messages.csv:  data to process
        - process_data.py: python file to process and clean the data

    3. models:
        - train_classifier.py: python file to build and test the model
 
### Instructions<a name="Instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app..
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

feel free to use the code here as you would like! 



