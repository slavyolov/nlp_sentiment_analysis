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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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
        fig.savefig('src/output/bar_plot_source_names.png', bbox_inches="tight")
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
        # self.word_frequency(text=text)  #TODO: close the fig

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
        fig.savefig('src/output/word_frequency_distribution.png', bbox_inches="tight")
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


class ModelEvaluation:
    def __init__(self, data):
        self.data = data

    def run(self, y_true, y_pred):
        labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        categories = ['negative', 'positive', 'neutral']
        cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.make_confusion_matrix(cf_matrix,
                                   group_names=labels,
                                   categories=categories,
                                   cmap='binary')
        return self.evaluate(y_test=y_test, y_pred=y_pred)

    @staticmethod
    def make_confusion_matrix(cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None,
                              y_true=None,
                              y_pred=None):
        """
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

        :param cf:            confusion matrix to be passed in
        :param group_names:   List of strings that represent the labels row by row to be shown in each square.
        :param categories:    List of strings containing the categories to be displayed on the x,y axis. Default is
                              'auto'
        :param count:         If True, show the raw number in the confusion matrix. Default is True.
        :param normalize:     If True, show the proportions for each category. Default is True.
        :param cbar:          If True, show the color bar. The cbar values are based off the values in the confusion
                              matrix. Default is True.
        :param xyticks:       If True, show x and y ticks. Default is True.
        :param xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        :param sum_stats:     If True, display summary statistics below the figure. Default is True.
        :param figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        :param cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                              See http://matplotlib.org/examples/color/colormaps_reference.html
        :param title:         Title for the heatmap. Default is None.
        :param y_true         Ground truth label
        :param y_pred         Predicted label
        :return:
        """
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            # Accuracy is sum of diagonal divided by total observations
            accuracy = np.trace(cf) / float(np.sum(cf))

            # if it is a binary confusion matrix, show some more stats
            if len(cf) == 2:
                # Metrics for Binary Confusion Matrices
                precision = cf[1, 1] / sum(cf[:, 1])
                recall = cf[1, 1] / sum(cf[1, :])
                f1_score = 2 * precision * recall / (precision + recall)
                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
            else:
                precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
                recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
                f1_score = recall_score(y_true=y_true, y_pred=y_pred, average='macro')

                stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                    accuracy, precision, recall, f1_score)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        fig = plt.figure(figsize=figsize)
        # plt.figure(figsize=figsize)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)

        fig.savefig('src/output/confusion_matrix_summary.png', bbox_inches="tight")
        plt.close()


# class SentimentAnalysis2:
#     def __init__(self, data):
#         self.data = data
#
#     def run(self):
#
#         # import SentimentIntensityAnalyzer class
#         # from vaderSentiment.vaderSentiment module.
#
#         sid_obj = SentimentIntensityAnalyzer()
#         # txt = ["Rumen Draganov received an award for overall contribution to the development, promotion and creation of a historical memory for tourism in the country",
#         #        "Lyubozar Fratev received an award for his contribution to the development and promotion of regional tourism"
#         #        ]
#         # txt = ["study is going on as usual", "I am very sad today."]
#
#         txt = nlp_df["translated_body"].to_list()[:5]
#
#         for sentence in txt:
#             print(sid_obj.polarity_scores(sentence))
#




    # def sentiment_labels(sentence):
    #     if sentiment_dict['compound'] >= 0.05:
    #         print("Positive")
    #
    #     elif sentiment_dict['compound'] <= - 0.05:
    #         print("Negative")
    #
    #     else:
    #         print("Neutral")
    #
    #     # Create a SentimentIntensityAnalyzer object.
    #     sid_obj = SentimentIntensityAnalyzer()
    #
    #     # polarity_scores method of SentimentIntensityAnalyzer
    #     # object gives a sentiment dictionary.
    #     # which contains pos, neg, neu, and compound scores.
    #     sentiment_dict = sid_obj.polarity_scores(sentence)
    #
    #     print("Overall sentiment dictionary is : ", sentiment_dict)
    #     print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
    #     print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
    #     print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")
    #
    #     print("Sentence Overall Rated As", end=" ")
    #
    #     # decide sentiment as positive, negative and neutral
    #     if sentiment_dict['compound'] >= 0.05:
    #         print("Positive")
    #
    #     elif sentiment_dict['compound'] <= - 0.05:
    #         print("Negative")
    #
    #     else:
    #         print("Neutral")
    #
    # # TODO: do mermaid ?
    #
    # # Filter only the information coming from forums and "bg-mamma.com"
    # # sources_l = nlp_df['source_name'].to_list()
    # # forum_sources = [source for source in sources_l if "forum" in source]
    # # forum_sources.append("bg-mamma.com")
    # # nlp_df = nlp_df[nlp_df["source_name"].isin(forum_sources)]
    # # nlp_df.reset_index(drop=True, inplace=True)
    # # self.data_size(nlp_df, message="forums and bg-mamma")
    #
    # # Select only cases that have both 'electronic' and 'government'
    # nlp_df = nlp_df[(nlp_df["translated_body"].str.contains("electronic")) & (
    #     nlp_df["translated_body"].str.contains("government"))]
    # nlp_df.reset_index(drop=True, inplace=True)
    # self.data_size(nlp_df, message="electronic government")
    #
    # # Afinn sentiment analysis :
    # from afinn import Afinn
    # afinn = Afinn()
    # sentiments = []
    # for sentence in nlp_df["translated_body"].to_list():
    #     sentiments.append(afinn.score(sentence))
    #
    # sentiment = ['positive' if score > 0
    #              else 'negative' if score < 0
    # else 'neutral'
    #              for score in sentiments]
    #
    # df = pd.DataFrame()
    # df['topic'] = nlp_df["translated_body"]
    # df['scores'] = sentiments
    # df['sentiments'] = sentiment
    # print(df)
    #
    # # TODO: use source_type_name for filtering/grouping of results (also evaluation)
    #
    # # Sentiment analysis using Vader
    # sid_obj = SentimentIntensityAnalyzer()
    # sentiments = []
    # for sentence in nlp_df["translated_body"].to_list():
    #     sentiments.append(sid_obj.polarity_scores(sentence))
    #
    # sentiments_vader_df = pd.DataFrame.from_records(sentiments)
    # sentiments_vader_df['predictions'] = sentiments_vader_df.apply(self.sentiment_labels, axis=1)
    # sentiments_vader_df['translated_body'] = nlp_df[['translated_body']]
    # sentiments_vader_df['body_clean'] = nlp_df[['body_clean']]
    #
    # print("sentiments_vader_df row count\n", sentiments_vader_df["predictions"].value_counts())
    # print("")
    # print("sentiments_vader_df share\n", sentiments_vader_df["predictions"].value_counts(normalize=True).round(1))
    #
    # # pick random sample for labeling
    # df_sample = sentiments_vader_df.groupby("predictions", group_keys=False).apply(lambda x: x.sample(frac=0.1))
    # print("df_sample row count\n", df_sample["predictions"].value_counts())
    # print("")
    # print("df_sample share\n", df_sample["predictions"].value_counts(normalize=True).round(1))
    #
    # # TODO: filter only electeronic government
    #
    # # TODO: remove quotes when doing sentiment analysis. Vader ?
    # df_sample.to_excel("src/output/sample_to_label.xlsx")
    # with open('readme.txt', 'w') as f:
    #     f.write('readme')
    #
    # # nlp_df.to_csv("src/output/english_information.csv")
    # bmw = nlp_df[nlp_df['source_name'] == "bmwpower-bg.net/forums"]
    # print(bmw['body_clean'].to_markdown())
    #
    # bg_mamma = nlp_df[nlp_df['source_name'] == "bg-mamma.com"]
    # print(bg_mamma['body_clean'].to_markdown())
    #
    # # word cloud - #TODO : piut in a function
    # from wordcloud import WordCloud
    # # import matplotlib.pyplot as plt
    # text = " ".join(text_body for text_body in bmw['translated_body'])
    # word_cloud = WordCloud(collocations=False, background_color='white').generate(text=text)
    # # plt.imshow(word_cloud, interpolation='bilinear')
    # # plt.axis("off")
    # # plt.show()
    #
    # word_cloud = WordCloud(collocations=False, background_color='white') \
    #     .generate(text=text).to_file("src/output/site_bmw.png")
    #
    # text = " ".join(text_body for text_body in nlp_df['translated_body'])
    # word_cloud = WordCloud(collocations=False, background_color='white') \
    #     .generate(text=text).to_file("src/output/nlp_df.png")
    #
    # return 1 + 1