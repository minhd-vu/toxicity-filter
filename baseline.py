#!/usr/local/bin/python3
import numpy as np
import pandas as pd
import torch
import re
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def clean_text(df, text):
    """
    Cleans text by replacing unwanted characters with blanks
    Replaces @ signs with word at
    Makes all text lowercase
    """

    # df[text] = re.sub('s+', ' ', df[text])
    # df[text] = re.sub('^A-Za-z0-9()!?@\s\'\`\*\"\_\n]', '', df[text])
    # df[text] = df[text].str.lower()

    df[text] = df[text].str.replace(r'[^A-Za-z0-9()!?@\s\'\`\*\"\_\n]', '', regex=True)
    df[text] = df[text].str.replace(r'@', 'at', regex=True)
    df[text] = df[text].str.lower()

    return df


df = pd.read_csv('data/jigsaw-toxic-comment-classification-challenge/train.csv')
df = df[['comment_text', 'toxic']]
df = clean_text(df, 'comment_text')

print(df.head())
# print(df['toxic'].value_counts())

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = df['comment_text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0] * (max_len-len(i)) for i in tokenized.values])

np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()
labels = df['toxic']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

score = lr_clf.score(test_features, test_labels)

print(score)
