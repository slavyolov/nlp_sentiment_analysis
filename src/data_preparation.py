import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import nltk
import seaborn as sns


class DataPreparation:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()

    def run(self):
        nlp_df = self.data

        # take only the sources with translated information from Bulgarian to English:
        nlp_df = nlp_df[~nlp_df['translated_body'].isnull()].reset_index()

        # Extract the text body
        text = " ".join(text_body for text_body in nlp_df['translated_body'])

        # get word frequency :
        self.word_frequency(text=text)


        # count_by source_url
        sources = nlp_df['source_name'].value_counts()

        # nlp_df.to_csv("src/output/english_information.csv")
        bmw = nlp_df[nlp_df['source_name'] == "bmwpower-bg.net/forums"]
        print(bmw['body_clean'].to_markdown())

        bg_mamma = nlp_df[nlp_df['source_name'] == "bg-mamma.com"]
        print(bg_mamma['body_clean'].to_markdown())

        # word cloud
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
