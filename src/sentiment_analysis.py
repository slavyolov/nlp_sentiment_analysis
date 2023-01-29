import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
import seaborn as sns
from setup_logger import create_logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


class DataPreparation:
    def __init__(self, config):
        self.config = config
        self.logger = create_logger(file_path='src/output/data_preparation.log')
        self.data = self.read_data()

    def run(self):
        nlp_df = self.data

        # take only the sources with translated information from Bulgarian to English:
        nlp_df = nlp_df[~nlp_df['translated_body'].isnull()].reset_index()
        self.data_size(nlp_df, message="translated_body == EN")

        # Check the 'sentiment' field
        self.logger.log(logging.INFO,
                        f'\nValues count sentiment column :\n{nlp_df["sentiment"].value_counts()}')

        # Check the 'source_type_name' field
        self.logger.log(logging.INFO,
                        f'\nValues count source_type_name column :\n{nlp_df["source_type_name"].value_counts()}')

        # Remove rows that are rarely expressing sentiments
        source_types_to_remove = ["Government Website", "Company Website"]
        nlp_df = nlp_df[~nlp_df['source_type_name'].isin(source_types_to_remove)]

        # Drop duplicates
        nlp_df = nlp_df.drop_duplicates(subset=["translated_body"])

        self.logger.log(logging.INFO,
                        f"""\nValues count source_type_name column - after 
                            filtering :\n{nlp_df['source_type_name'].value_counts()}""")

        # count_by source_name
        self.logger.log(logging.INFO,
                        f"""\nValues count source_name column - after 
                            filtering :\n{nlp_df['source_name'].value_counts()}""")

        # plot the source_names
        source_names_count = pd.DataFrame(nlp_df['source_name'].value_counts().reset_index().values,
                                          columns=["source_name", "count"])

        fig = plt.figure(figsize=(10, 4))
        plt.xticks(rotation=45)
        sns.barplot(data=source_names_count[:5], x="source_name", y="count")
        fig.savefig('src/output/plots/eda/bar_plot_source_names.png', bbox_inches="tight")
        plt.close()

        # take the labels from the Annotated field / RNN model :
        existing_annotations_df = nlp_df[nlp_df["sentiment"].isnull() == False]
        existing_annotations_df.reset_index(drop=True, inplace=True)
        get_rnn_scores = existing_annotations_df["sentiment"].to_list()
        get_rnn_scores = [int(x["annotation_class"]) for x in get_rnn_scores]
        score_labels = ['positive' if score == 1
                        else 'negative' if score == -1
                        else 'neutral'
                        for score in get_rnn_scores]

        existing_annotations_df["existing_label_from_RNN_model"] = score_labels
        existing_annotations_df = existing_annotations_df[["key", "existing_label_from_RNN_model"]]

        # Extract the text body and get word_frequency
        text = " ".join(text_body for text_body in nlp_df['translated_body'])
        self.word_frequency(text=text)

        nlp_df.reset_index(drop=True, inplace=True)

        return nlp_df, existing_annotations_df

    def read_data(self):
        file_name = Path(self.config.data_path)
        return pd.read_pickle(file_name.resolve())

    @staticmethod
    def data_size(df, message):
        print(f"row count after filtering | {message} | ", len(df))

    @staticmethod
    def word_frequency(text) -> None:
        """
        Display word frequency to better understand the data

        source : https://www.milindsoorya.com/blog/introduction-to-word-frequencies-in-nlp

        :param text: Combined text bodies
        :return:
        """
        nltk.download("stopwords")
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.append("also")

        # tokenization
        tokenized_input = text.split()

        # lowercase the text
        tokenized_input = list(map(lambda x: x.lower(), tokenized_input))

        # remove digits and special characters
        tokenized_clean = []
        for word in tokenized_input:
            if word.isalpha() == True:
                tokenized_clean.append(word)

        # get the list without stop words
        tokenized_final = []
        for word in tokenized_clean:
            if word not in stop_words:
                tokenized_final.append(word)

        sns.set_style('darkgrid')
        fig = plt.figure(figsize=(10, 4))
        nlp_words = nltk.FreqDist(tokenized_final)
        nlp_words.plot(20)
        fig.savefig('src/output/plots/eda/word_frequency_distribution.png', bbox_inches="tight")
        plt.close()

    @staticmethod
    def sentiment_labels(row):
        if row['compound'] >= 0.05:
            return 'Positive'
        elif row['compound'] <= - 0.05:
            return 'Negative'
        else:
            return 'Neutral'


class SentimentAnalysis:
    def __init__(self, data):
        self.data = data

    def run(self):
        """
        Return sentiment analysis using different approaches

        :return:
        """
        return self.baseline(), self.vader(), self.text_blob()

    def baseline(self):
        random_sentiments = self.data[["key"]]
        random_sentiments["random_labels"] = pd.DataFrame(
            np.random.choice(["positive", "negative", "neutral"], size=len(self.data))
        )
        return random_sentiments

    def vader(self) -> pd.DataFrame:
        """
        Provide sentiments using Vader (lexicon) polarity_scores method. The method gives a sentiment dictionary
        which contains pos, neg, neu, and compound scores.

        :return:
        """
        vader_ = SentimentIntensityAnalyzer()

        texts = self.data["translated_body"].to_list()

        sentiments_dict = []
        for text in texts:
            sentiments_dict.append(vader_.polarity_scores(text))

        sentiments_df = pd.DataFrame(sentiments_dict)
        sentiments_df["key"] = self.data["key"]

        # provide sentiment labels (https://github.com/cjhutto/vaderSentiment#about-the-scoring)
        conditions = [
            (sentiments_df['compound'] >= 0.05),
            (sentiments_df['compound'] <= -0.05),
            (sentiments_df['compound'] > -0.05) & (sentiments_df['compound'] < 0.05),
        ]
        values = ['positive', 'negative', 'neutral']

        # create a new column and use np.select to assign values to it using our lists as arguments
        sentiments_df['sentiment_vader'] = np.select(conditions, values)

        return sentiments_df

    def text_blob(self) -> pd.DataFrame:
        """
        The output of TextBlob sentiment is polarity and subjectivity.

        Polarity score lies between (-1 to 1) where -1 identifies the most negative words such as ‘disgusting’,
        ‘awful’, ‘pathetic’, and 1 identifies the most positive words like ‘excellent’, ‘best’.

        Subjectivity score lies between (0 and 1), It shows the amount of personal opinion, If a sentence has high
        subjectivity i.e. close to 1, It resembles that the text contains more personal opinion than factual
        information.

        For the purpose of the curent project we will employ only the polarity score method. For neutral we are going
        to take the same range as in Vader [-0.05, 0.05]

        :return:
        """
        texts = self.data["translated_body"].to_list()

        polarity_scores = []
        for text in texts:
            text_blob_ = TextBlob(text)
            polarity_scores.append(text_blob_.sentiment.polarity)

        polarity_labels = ['positive' if score >= 0.05
                           else 'negative' if score <= -0.05
                           else 'neutral'
                           for score in polarity_scores]

        sentiments_df = self.data[["key"]]
        sentiments_df['scores'] = polarity_scores
        sentiments_df['text_blob_sentiments'] = polarity_labels
        return sentiments_df
