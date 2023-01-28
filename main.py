from src.data_preparation import DataPreparation, SentimentAnalysis
from pyhocon import ConfigFactory


config = ConfigFactory.parse_file('config/COMMON.conf')


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    nlp_df, existing_annotations_df = data_preparation.run()

    sentiment_analysis = SentimentAnalysis(data=nlp_df)
    baseline_df, vader_df, text_blob_df = sentiment_analysis.run()

    # get stratified random sample for labeling



    print("ok")

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
