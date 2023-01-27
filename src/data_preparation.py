import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import nltk
import seaborn as sns
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class DataPreparation:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()

    def run(self):
        nlp_df = self.data

        # take only the sources with translated information from Bulgarian to English:
        nlp_df = nlp_df[~nlp_df['translated_body'].isnull()].reset_index()
        self.data_size(nlp_df, message="translated_body == EN")

        print("values count based on source type", nlp_df["source_type_name"].value_counts())
        #TODO

        #TODO: there are annotations ?
        print("Existing labels", nlp_df["sentiment"].value_counts())

        # take the labels from the RNN model :
        xx = nlp_df[nlp_df["sentiment"].isnull() == False]
        xx["sentiment"].to_list()[0].values()
        get_rnn_scores = xx["sentiment"].to_list()
        get_rnn_scores = [int(x["annotation_class"]) for x in get_rnn_scores]
        column_name = "label_" + "200311_1621_GS_BG_MM_teXnetModelWordvecsMultiClassRNN_R80"
        score_labels = ['positive' if score == 1
                     else 'negative' if score == -1
                     else 'neutral'
                     for score in get_rnn_scores]


        # # Extract the text body
        # text = " ".join(text_body for text_body in nlp_df['translated_body'])
        #
        # # get word frequency :
        # self.word_frequency(text=text)

        # count_by source_url
        # sources = nlp_df['source_name'].value_counts()

        # Filter only the information coming from forums and "bg-mamma.com"
        # sources_l = nlp_df['source_name'].to_list()
        # forum_sources = [source for source in sources_l if "forum" in source]
        # forum_sources.append("bg-mamma.com")
        # nlp_df = nlp_df[nlp_df["source_name"].isin(forum_sources)]
        # nlp_df.reset_index(drop=True, inplace=True)
        # self.data_size(nlp_df, message="forums and bg-mamma")

        # Select only cases that have both 'electronic' and 'government'
        nlp_df = nlp_df[(nlp_df["translated_body"].str.contains("electronic")) & (
            nlp_df["translated_body"].str.contains("government"))]
        nlp_df.reset_index(drop=True, inplace=True)
        self.data_size(nlp_df, message="electronic government")

        # Afinn sentiment analysis :
        from afinn import Afinn
        afinn = Afinn()
        sentiments = []
        for sentence in nlp_df["translated_body"].to_list():
            sentiments.append(afinn.score(sentence))

        sentiment = ['positive' if score > 0
                     else 'negative' if score < 0
        else 'neutral'
                     for score in sentiments]

        df = pd.DataFrame()
        df['topic'] = nlp_df["translated_body"]
        df['scores'] = sentiments
        df['sentiments'] = sentiment
        print(df)

        #TODO: use source_type_name for filtering/grouping of results (also evaluation)


        # Sentiment analysis using Vader
        sid_obj = SentimentIntensityAnalyzer()
        sentiments = []
        for sentence in nlp_df["translated_body"].to_list():
            sentiments.append(sid_obj.polarity_scores(sentence))

        sentiments_vader_df = pd.DataFrame.from_records(sentiments)
        sentiments_vader_df['predictions'] = sentiments_vader_df.apply(self.sentiment_labels, axis=1)
        sentiments_vader_df['translated_body'] = nlp_df[['translated_body']]
        sentiments_vader_df['body_clean'] = nlp_df[['body_clean']]

        print("sentiments_vader_df row count\n", sentiments_vader_df["predictions"].value_counts())
        print("")
        print("sentiments_vader_df share\n", sentiments_vader_df["predictions"].value_counts(normalize=True).round(1))

        # pick random sample for labeling
        df_sample = sentiments_vader_df.groupby("predictions", group_keys=False).apply(lambda x: x.sample(frac=0.1))
        print("df_sample row count\n", df_sample["predictions"].value_counts())
        print("")
        print("df_sample share\n", df_sample["predictions"].value_counts(normalize=True).round(1))

        # TODO: filter only electeronic government


        # TODO: remove quotes when doing sentiment analysis. Vader ?
        df_sample.to_excel("src/output/sample_to_label.xlsx")
        with open('readme.txt', 'w') as f:
            f.write('readme')

        # nlp_df.to_csv("src/output/english_information.csv")
        bmw = nlp_df[nlp_df['source_name'] == "bmwpower-bg.net/forums"]
        print(bmw['body_clean'].to_markdown())

        bg_mamma = nlp_df[nlp_df['source_name'] == "bg-mamma.com"]
        print(bg_mamma['body_clean'].to_markdown())

        # word cloud - #TODO : piut in a function
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        text = " ".join(text_body for text_body in bmw['translated_body'])
        word_cloud = WordCloud(collocations=False, background_color='white').generate(text=text)
        # plt.imshow(word_cloud, interpolation='bilinear')
        # plt.axis("off")
        # plt.show()

        word_cloud = WordCloud(collocations=False, background_color='white')\
            .generate(text=text).to_file("src/output/site_bmw.png")

        text = " ".join(text_body for text_body in nlp_df['translated_body'])
        word_cloud = WordCloud(collocations=False, background_color='white')\
            .generate(text=text).to_file("src/output/nlp_df.png")


        return 1+1

    def read_data(self):
        file_name = Path(self.config.data_path)
        return pd.read_pickle(file_name.resolve())

    @staticmethod
    def data_size(df, message):
        print(f"row count after filtering | {message} | ", len(df))

    def word_frequency(self, text):
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
        # plt.show()
        fig.savefig('src/output/word_frequency_distribution.png', bbox_inches="tight")
        return 1+1

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

        # import SentimentIntensityAnalyzer class
        # from vaderSentiment.vaderSentiment module.
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sid_obj = SentimentIntensityAnalyzer()
        # txt = ["Rumen Draganov received an award for overall contribution to the development, promotion and creation of a historical memory for tourism in the country",
        #        "Lyubozar Fratev received an award for his contribution to the development and promotion of regional tourism"
        #        ]
        # txt = ["study is going on as usual", "I am very sad today."]

        txt = nlp_df["translated_body"].to_list()[:5]

        for sentence in txt:
            print(sid_obj.polarity_scores(sentence))





    def sentiment_labels(sentence):
        if sentiment_dict['compound'] >= 0.05:
            print("Positive")

        elif sentiment_dict['compound'] <= - 0.05:
            print("Negative")

        else:
            print("Neutral")

        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(sentence)

        print("Overall sentiment dictionary is : ", sentiment_dict)
        print("sentence was rated as ", sentiment_dict['neg'] * 100, "% Negative")
        print("sentence was rated as ", sentiment_dict['neu'] * 100, "% Neutral")
        print("sentence was rated as ", sentiment_dict['pos'] * 100, "% Positive")

        print("Sentence Overall Rated As", end=" ")

        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            print("Positive")

        elif sentiment_dict['compound'] <= - 0.05:
            print("Negative")

        else:
            print("Neutral")