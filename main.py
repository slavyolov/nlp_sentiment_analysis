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



    # evaluation_metrics = ModelEvaluation(data=combined_df)
    # cf_matrix = evaluation_metrics.run(y_true=combined_df["random_labels"], y_pred=combined_df["sentiment_vader"])
    #
    # from sklearn.metrics import confusion_matrix, precision_score, recall_score
    #
    # precisioon_ = precision_score(y_true=combined_df["random_labels"], y_pred=combined_df["sentiment_vader"], average='macro')
    # recall_ = recall_score(y_true=combined_df["random_labels"], y_pred=combined_df["sentiment_vader"],
    #                               average='macro')
    # f = recall_score(y_true=combined_df["random_labels"], y_pred=combined_df["sentiment_vader"],
    #                               average='macro')
    #
    # from sklearn import metrics
    # import seaborn as sns
    #
    # clf_report = metrics.classification_report(y_true=combined_df["random_labels"],
    #                                             y_pred=combined_df["sentiment_vader"],
    #                                             digits=3,
    #                                             labels=["negative", "positive", "neutral"],
    #                                             target_names=["negative", "positive", "neutral"],
    #                                             output_dict=True
    #                                             )
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 4))
    # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    # fig.savefig('src/output/classification_report.png', bbox_inches="tight")
    # plt.close()
    #
    #
    #
    #
    #
    #
    #
    # labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    # categories = ['Zero', 'One']
    # make_confusion_matrix(cf_matrix,
    #                       group_names=labels,
    #                       categories=categories,
    #                       cmap='binary')

    # pd.DataFrame(confusion_matrix(y_test="random_labels", y_pred="sentiment_vader", columns=labels, index=labels))

    #
    # TODO: majority voting
    # TODO: scoring (f1-score, precision, recall)
    # TODO: word cloud (pick 2-3)

        # # Select only cases that have both 'electronic' and 'government'
        # nlp_df = nlp_df[(nlp_df["translated_body"].str.contains("electronic")) & (
        #     nlp_df["translated_body"].str.contains("government"))]
        # nlp_df.reset_index(drop=True, inplace=True)
        # self.data_size(nlp_df, message="electronic government")


    # Evaluate data

    print("ok")

    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
