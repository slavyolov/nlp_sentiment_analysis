from src.data_preparation import DataPreparation, SentimentAnalysis, ModelEvaluation
from pyhocon import ConfigFactory
import pandas as pd
from collections import Counter
from evaluation import ModelEvaluation


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
        sample_df.to_excel("src/output/data/random_sample_to_label.xlsx")

    # Prepare data for model evaluation
    def f(x):
        a, b = Counter(x).most_common(1)[0]
        return pd.Series([a, b])

    annotations = pd.read_csv("src/input/Annotations_of_sample_data.csv")
    annotations[['ground_truth_label', 'freq_count']] = annotations.apply(f, axis=1)

    combined_df = pd.merge(left=combined_df, right=annotations[['index_orig_df', 'ground_truth_label']],
                           how="left",
                           left_on=combined_df.index, right_on="index_orig_df")

    combined_df_w_expert_gt = combined_df[combined_df["ground_truth_label"].isnull() == False]
    combined_df_w_existing_label = combined_df[combined_df["existing_label_from_RNN_model"].isnull() == False]

    combined_df_w_expert_gt.to_excel("src/output/data/combined_df_w_expert_gt.xlsx")
    combined_df_w_existing_label.to_excel("src/output/data/combined_df_w_existing_label.xlsx")
