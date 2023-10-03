# Sentiment Analysis Model

## Table of Contents
- [About](#about)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Text Preprocessing](#text-preprocessing)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Results Visualization](#results-visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About
This project is a sentiment analysis model designed to classify news headlines into three categories: positive, negative, or neutral sentiment. Sentiment analysis is a crucial component of natural language processing that has practical applications in understanding public opinion, monitoring brand perception, and making data-driven decisions.

## Project Overview
In this project, we utilize a dataset consisting of approximately 210,000 news headlines spanning from 2012 to 2022, sourced from HuffPost. Our goal is to build a sentiment analysis model to automatically categorize these headlines based on their sentiment.

## Dataset
The dataset used in this project is sourced from HuffPost and comprises news headlines. It includes various attributes such as category, headline, date, and more. You can access the dataset [here](https://www.kaggle.com/datasets/rmisra/news-category-dataset/code).

### Colab Link
To access the Colab Notebook for this project, click [here](https://colab.research.google.com/drive/1ji9wBHSBNe14V3SRAcdleGM0BIh_ENlK?usp=sharing).

## Getting Started
1. To begin, mount Google Drive for dataset access:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Load the dataset using pandas:
   ```python
   import pandas as pd
   df = pd.read_json("/content/drive/My Drive/News_Category_Dataset_v3.json", lines=True)
   ```

## Text Preprocessing
We perform text preprocessing to clean the headlines before model training. The following code snippet demonstrates text cleaning:

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    # Remove punctuation and convert to lowercase
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    cleaned_text = " ".join(tokens)
    return cleaned_text

df["cleaned_headline"] = df["headline"].apply(clean_text)
```

## Handling Class Imbalance
In cases where class imbalance is observed, we apply techniques to balance the distribution of sentiment classes, ensuring a fair representation of each sentiment.

## Model Building and Evaluation
We utilize a logistic regression model for sentiment analysis. The model is trained and evaluated using the following code:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(df["cleaned_headline"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df["sentiment"], test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Results Visualization
We visualize the sentiment distribution and precision of the model. The following code generates bar plots and a confusion matrix:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

results_df = pd.DataFrame({'sentiment': y_pred})

# Generating a classification report to get precision for each class
class_report = classification_report(y_test, y_pred, output_dict=True)

# Extracting precision values for each sentiment category
sentiment_precisions = {
    'positive': class_report['positive']['precision'],
    'negative': class_report['negative']['precision'],
    'neutral': class_report['neutral']['precision']
}

sentiment_counts = results_df['sentiment'].value_counts()

plt.figure(figsize=(8, 6))
ax = sentiment_counts.plot(kind='bar', color=['blue', 'green', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')

for sentiment, precision in sentiment_precisions.items():
    ax.text(0.5, 0.75 - list(sentiment_counts.index).index(sentiment) * 0.1,
            f'{sentiment.capitalize()} Precision: {precision:.2f}', transform=ax.transAxes, fontsize=12, ha='center')

# Rotate the x-axis labels to be horizontal
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Usage
Explain how others can use your project, including any scripts or commands they need to run.

## Contributing
If you'd like to contribute to this project, feel free to submit bug reports, feature requests, or code contributions.

## License
This project is licensed under the [MIT License](LICENSE).

---
