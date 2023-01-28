import pandas as pd
from evaluation import ModelEvaluation
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


if __name__ == '__main__':
    combined_df_w_expert_gt = pd.read_excel("src/output/data/combined_df_w_expert_gt.xlsx")
    combined_df_w_existing_label = pd.read_excel("src/output/data/combined_df_w_existing_label.xlsx")

    # Model evaluation
    evaluation_o = ModelEvaluation(data=combined_df_w_expert_gt)

    # evaluate against Vader
    evaluation_o.plot_classification_report(y_true_col="ground_truth_label", y_pred_col="random_labels",
                                            method="baseline")

    evaluation_o.plot_classification_report(y_true_col="ground_truth_label", y_pred_col="sentiment_vader",
                                            method="vader")

    evaluation_o.plot_classification_report(y_true_col="ground_truth_label", y_pred_col="text_blob_sentiments",
                                            method="text_blob")

    # plot confusion matrix
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['negative', 'positive', 'neutral']

    # for baseline
    cf_matrix = confusion_matrix(y_true=combined_df_w_expert_gt["random_labels"],
                                 y_pred=combined_df_w_expert_gt["ground_truth_label"])

    evaluation_o.plot_confusion_matrix(cf_matrix,
                                       group_names=labels,
                                       categories=categories,
                                       cmap='binary', method='baseline')

    # for vader
    cf_matrix = confusion_matrix(y_true=combined_df_w_expert_gt["ground_truth_label"],
                                 y_pred=combined_df_w_expert_gt["sentiment_vader"])
    evaluation_o.plot_confusion_matrix(cf_matrix,
                                       group_names=labels,
                                       categories=categories,
                                       cmap='binary', method='vader')

    # for text_blob
    cf_matrix = confusion_matrix(y_true=combined_df_w_expert_gt["ground_truth_label"],
                                 y_pred=combined_df_w_expert_gt["text_blob_sentiments"])
    evaluation_o.plot_confusion_matrix(cf_matrix,
                                       group_names=labels,
                                       categories=categories,
                                       cmap='binary', method='text_blob')

    # word clouds
    positive_text = " ".join(text_body for text_body in combined_df_w_expert_gt[
        combined_df_w_expert_gt['ground_truth_label'] == 'positive']['translated_body'])

    WordCloud(collocations=False, background_color='white') \
        .generate(text=positive_text).to_file("src/output/plots/word_clouds/word_cloud_positive.png")

    negative_text = " ".join(text_body for text_body in combined_df_w_expert_gt[
        combined_df_w_expert_gt['ground_truth_label'] == 'negative']['translated_body'])

    WordCloud(collocations=False, background_color='white') \
        .generate(text=negative_text).to_file("src/output/plots/word_clouds/word_cloud_negative.png")

    neutral_text = " ".join(text_body for text_body in combined_df_w_expert_gt[
        combined_df_w_expert_gt['ground_truth_label'] == 'neutral']['translated_body'])

    WordCloud(collocations=False, background_color='white') \
        .generate(text=neutral_text).to_file("src/output/plots/word_clouds/word_cloud_neutral.png")
