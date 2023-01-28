from src.data_preparation import DataPreparation, SentimentAnalysis
from pyhocon import ConfigFactory
import pandas as pd


config = ConfigFactory.parse_file('config/COMMON.conf')
relabel = False


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    nlp_df, existing_annotations_df = data_preparation.run()

    sentiment_analysis = SentimentAnalysis(data=nlp_df)
    baseline_df, vader_df, text_blob_df = sentiment_analysis.run()

    # Combine data
    combined_df = nlp_df[["key", "source_type_name", "source_name", "body_clean", "translated_body"]]
    combined_df = pd.merge(left=combined_df, right=existing_annotations_df, how="left", on="key")
    combined_df = pd.merge(left=combined_df, right=baseline_df, how="left", on="key")
    combined_df = pd.merge(left=combined_df, right=vader_df[["key", "sentiment_vader"]], how="left", on="key")
    combined_df = pd.merge(left=combined_df, right=text_blob_df[["key", "text_blob_sentiments"]], how="left", on="key")

    # get stratified random sample (10% out of each group neutral, negative, positive) based on the vader seintiments
    if relabel:
        sample_v_p = combined_df[combined_df["sentiment_vader"] == "positive"].sample(n=25)
        sample_v_neu = combined_df[combined_df["sentiment_vader"] == "neutral"].sample(n=25)
        sample_v_neg = combined_df[combined_df["sentiment_vader"] == "negative"].sample(n=25)

        sample_df = pd.concat([sample_v_p[["body_clean", "translated_body"]],
                               sample_v_neu[["body_clean", "translated_body"]],
                               sample_v_neg[["body_clean", "translated_body"]]
                               ])
        sample_df.to_excel("src/output/random_sample_to_label.xlsx")


        df_sample = vader_df.groupby("sentiment_vader", group_keys=False).apply(lambda x: x.sample(frac=0.1))
        print("df_sample row count\n", df_sample["predictions"].value_counts())
        print("")
        print("df_sample share\n", df_sample["predictions"].value_counts(normalize=True).round(1))

    len(vader_df[vader_df["sentiment_vader"] == "positive"].sample(frac=0.05))
    # 25 negatives
    # 25 neutral
    # 25 positive


    # Evaluate data

    print("ok")

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
