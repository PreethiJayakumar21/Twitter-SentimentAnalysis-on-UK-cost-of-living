# Sentiment Analysis of Public Tweets on the UK's Cost of Living
## Project Overview
This project performs sentiment analysis on a dataset of public tweets related to the UK's cost of living. The dataset is sourced from Kaggle, and the analysis uses the VADER sentiment analysis model to classify the sentiment behind these tweets. The project aims to provide insights into public opinion and trends by categorizing tweets into positive, negative, or neutral sentiment.

## Key Features
Sentiment Classification: Uses VADER sentiment analysis to categorize tweets into positive, negative, or neutral sentiment.

## Data Collection: 
Fetch dataset of tweets from Kaggle.
[Tweets datasets] https://www.kaggle.com/datasets/tleonel/cost-of-living?select=costofliving-query-tweets.csv

Text Preprocessing: Cleans and processes raw tweet text for accurate sentiment analysis.
Visualization: Provides visual representations of sentiment distribution across different topics.

## Requirements
Programming Language: Python 3.x
Libraries:
kaggle (for dataset downloading)
nltk (for text preprocessing and sentiment analysis)
sklearn (for preprocessing and encoding)
matplotlib (for visualizations)
pandas (for data manipulation)
numpy (for numerical operations)

```bash
pip install kaggle nltk scikit-learn matplotlib pandas numpy
```

## Data Collection
The dataset for sentiment analysis is sourced from Kaggle. You will need to configure your Kaggle API credentials to download the dataset.

### Steps to Download the Dataset:
Set up your Kaggle API credentials by following this [guide].

Add your kaggle.json file (containing your API credentials) to the project directory or the appropriate folder on your system.

Download the dataset using the Kaggle API with the following command:

```bash
kaggle datasets download -d datasets/tleonel/cost-of-living?select=costofliving-query-tweets.csv
```
or copy the API command
Here’s how you can find the guide:

Go to the Kaggle website: https://www.kaggle.com.
Navigate to the "API" section in your account settings.
Follow the instructions to generate and download the kaggle.json file, which will contain your API credentials.
Use the credentials to authenticate your access via the Kaggle API.
For more detailed steps, you can look for the section on "Getting Started Installation & Authentication" in the Kaggle API documentation.

![image](https://github.com/user-attachments/assets/ff47cf4b-adf2-45c6-9c98-4841547c4849)

or load the dataset given in the git 

Extract the dataset to the data/ folder:

```bash
unzip <dataset-name>.zip -d cost-of-living?select=costofliving-query-tweets.csv
```

##Text Preprocessing and Sentiment Analysis
The raw tweets undergo the following preprocessing steps before sentiment analysis:

Cleaning: Removes URLs, special characters, and non-text elements.
Tokenization: Splits the text into tokens (words).
Stop-word Removal: Filters out common stop-words.
Sentiment Analysis: Sentiment is analyzed using the VADER model from the nltk library, which assigns a sentiment score to each tweet.
The sentiment scores are then classified as:

### Positive, Negative and Neutral


## Models Used
## 1. Lexicon-Based Approach
VADER (Valence Aware Dictionary and sEntiment Reasoner) is used to score sentiment based on a predefined lexicon of words.
## 2. Machine Learning Models
Logistic Regression: A binary classifier that predicts tweet sentiment.
Support Vector Machines (SVM): A robust classifier that separates classes with a hyperplane.
Decision Trees: A model that splits data into branches to predict sentiment.
## 3. Deep Learning Models
LSTM (Long Short-Term Memory): A recurrent neural network (RNN) model used to predict sentiment from sequential tweet data.
CNN (Convolutional Neural Network): A deep learning model that captures sentiment by analyzing local text features.

## Model Evaluation
Each model’s performance is evaluated using metrics such as:
1.Accuracy
2. Precision
3. Recall
4. F1 Score
Comparative performance between models is visualized to showcase which approach works best for this specific dataset.

## How to Run
Clone the repository:

```bash
git clone https://github.com/PreethiJayakumar21/Twitter-SentimentAnalysis-on-UK-cost-of-living.git
```

Install the requirements

Download the dataset from Kaggle and place it in the data/ folder.

Run the Jupyter notebook:

```bash
jupyter notebook Sentiment\ Analysis.ipynb
```
The analysis will output the results for each model along with visualizations showing sentiment distribution and model performance.

Project Structure
bash
Copy code
├── data/                          # Folder for the dataset
├── Sentiment Analysis.ipynb        # Jupyter notebook for the analysis
├── README.md                       # This file
Example Output
After running the analysis, you will see:

Sentiment distribution across topics (positive, negative, neutral).
Model comparison charts showing accuracy, precision, recall, and F1 scores.
Visualizations highlighting the effectiveness of different approaches on various topics (e.g., government response, inflation).

