from transformers import pipeline
from time import time
# import entropy from scipy.stats
from scipy.stats import entropy

class ZeroShotClassification:
    " A class to perform various NLP tasks using Hugging Face's Transformers library "
    
    def __init__(self, task, model_zero_shot_classification='facebook/bart-large-mnli', seed=42):
        assert task == 'zero-shot-classification', "Task must be zero-shot-classification"
        self.task = task
        self.seed = seed
        self.nlp = pipeline(task)
        self.model_zero_shot_classification = model_zero_shot_classification

    def zero_shot_classification(self, text, candidate_labels):
        return self.nlp(text, model=self.model_zero_shot_classification, candidate_labels=candidate_labels)
    

if __name__ == "__main__":

    import pandas as pd

    csv_path_neg = 'classification_tweets_10k_neg.csv'
    csv_path_pos = 'classification_tweets_10k_pos.csv'

    # read csv file
    df_neg = pd.read_csv(csv_path_neg)  # negative tweets
    df_pos = pd.read_csv(csv_path_pos)  # positive tweets
    df = pd.concat([df_neg, df_pos])

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head(2))

    # sentiment analysis
    labels_budget = 300
    
    text = df['sentence'].tail(labels_budget).tolist()

    # facebook/bart-large-mnli
    zero_shot_classification = ZeroShotClassification('zero-shot-classification', model_zero_shot_classification='facebook/bart-large-mnli')

    # print for each sequence label and score
    print('--------------------------------')
    print('Zero Shot Classification')
    print(f'Model: {zero_shot_classification.model_zero_shot_classification}')
    print('--------------------------------')
    start_time = time()

    # list to store predicted and observed labels
    list_predicted_labels = []
    list_observed_labels = []
    list_entropy = []
    list_scores = []


    for i in range(len(text)):

        # get the predicted labels and scores
        predicted_labels = zero_shot_classification.zero_shot_classification(text[i], ['positive', 'negative'])

        # get the observed labels
        observed_label = df['label'].tail(labels_budget).tolist()[i]

        # append the predicted and observed labels
        list_predicted_labels.append(1 if predicted_labels['labels'][0] == 'positive' else 0)
        list_observed_labels.append(observed_label)
        list_entropy.append(entropy(predicted_labels['scores']))
        list_scores.append(predicted_labels['scores'])

    # calculate F1 score
    from sklearn.metrics import f1_score

    F1 = f1_score(list_observed_labels, list_predicted_labels)

    # get sentence with highest entropy
    max_entropy = max(list_entropy)
    min_entropy = min(list_entropy)

    index_max_entropy = list_entropy.index(max_entropy)
    sentence_max_entropy = text[index_max_entropy]

    index_min_entropy = list_entropy.index(min_entropy)
    sentence_min_entropy = text[index_min_entropy]

    print(f'Sentence with highest entropy: {sentence_max_entropy}')
    print(f'Entropy: {max_entropy:.2f}')
    print(f'Scores: {list_scores[index_max_entropy]}')

    print('--------------------------------')

    print(f'Sentence with lowest entropy: {sentence_min_entropy}')
    print(f'Entropy: {min_entropy:.2f}')
    print(f'Scores: {list_scores[index_min_entropy]}')
    print(f'Observed label: {list_observed_labels[index_min_entropy]}')
    print(f'Predicted label: {list_predicted_labels[index_min_entropy]}')

    print('--------------------------------')

    print(f'Accuracy M1: {F1:.2f}')
    print(f'Time taken: {time() - start_time:.2f} seconds')
    print('--------------------------------')
