# Movie Review Sentiment Analysis

This project implements a machine learning model to classify movie reviews as positive or negative based on sentiment analysis.

## Project Overview

The system uses Natural Language Processing (NLP) techniques to analyze the sentiment of movie reviews from the IMDb dataset. It includes:

- Text preprocessing (lowercasing, removing stopwords, tokenization)
- Model training using machine learning algorithms (Logistic Regression, Naïve Bayes, or SVM)
- Model evaluation using accuracy and F1-score
- A simple web interface for users to enter reviews and get sentiment predictions

## Directory Structure

```
sentiment_analysis/
├── data/                   # Store the IMDb dataset here
├── models/                 # Saved trained models
├── src/                    # Source code
│   ├── preprocessing.py    # Text preprocessing functions
│   ├── model.py            # Model training and evaluation
│   └── app.py              # Flask web application
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup and Installation

1. Clone the repository or download the project files.

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place the IMDb dataset in the `data/` directory.
   The dataset should be a CSV file named "IMDB Dataset.csv" with at least two columns:
   - `review`: The text of the movie review
   - `sentiment`: The sentiment label ("positive" or "negative")

## Usage

### Training the Model

Run the training script:

```
python src/train.py --data data/IMDB\ Dataset.csv --model logistic
```

Options:
- `--data`: Path to the dataset CSV file (default: '../data/IMDB Dataset.csv')
- `--model`: Type of model to train (choices: 'logistic', 'naive_bayes', 'svm', default: 'logistic')
- `--test-size`: Proportion of data to use for testing (default: 0.2)

### Running the Web Application

After training the model, start the Flask web application:

```
cd src
python app.py
```

Then open your web browser and navigate to `http://127.0.0.1:5000/` to interact with the sentiment analysis system.

## Model Evaluation

After training, the model's performance metrics (accuracy and F1-score) will be displayed in the console. A confusion matrix image will also be saved in the `models/` directory.

## Extending the Project

- Add more preprocessing techniques (e.g., lemmatization)
- Implement different machine learning algorithms
- Add support for deep learning models (e.g., LSTM, BERT)
- Improve the web interface with additional features
- Add cross-validation for more robust model evaluation