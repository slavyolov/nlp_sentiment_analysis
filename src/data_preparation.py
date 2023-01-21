import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class DataPreparation:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()

    def run(self):
        nlp_df = self.data

        # take only the sources with translated information from Bulgarian to English:
        nlp_df = nlp_df[~nlp_df['translated_body'].isnull()].reset_index()

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


#
# all_texts = []
#
# from bs4 import BeautifulSoup
# import requests
#
# def parse_article(article_url):
#     print("Downloading {}".format(article_url))
#     r = requests.get(article_url)
#     soup = BeautifulSoup(r.text, "html.parser")
#     ps = soup.find_all('p')
#     text = "\n".join(p.get_text() for p in ps)
#     return text
#
# for article in xx['source_url']:
#     all_texts.append(parse_article(article))
#
# cloud = get_wordcloud(" ".join(all_texts))
# articles.append(Article(url=None, image=cloud))  # no URL for the "meta-article"
# return render_template('home.html', articles=articles)