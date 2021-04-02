#!/usr/local/bin/python3
import pdb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from functions import *


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Preparing train data
    train_df = pd.read_csv("data/train.csv")
    train_df = train_df[["comment_text", "target"]]
    train_df = clean_text(train_df, "comment_text")
    #train_df["comment_text"][21] = train_df["comment_text"][21].replace('\n', ' ')
    train_df["target"] = class_labels(train_df["target"])
    train_df.columns = ["text", "labels"]
    #eval_df = train_df[-2000:]
    eval_df = train_df[-2000:]
    train_df = train_df[0:2000]
    #train_df = train_df[:-1999]
    train_df.to_csv("data/train.tsv", sep=",", index=False)
    eval_df.to_csv("data/eval.tsv", sep=",", index=False)
    # Preparing eval data

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1, lazy_loading=True, lazy_labels_column=1, lazy_text_column=0, lazy_delimiter=',', labels_map={'0': 0, '1': 1})
    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
    )

    # Train the model
    #pdb.set_trace()
    try:
        model.train_model("data/train.tsv")
    except:
        model = ClassificationModel(
            "roberta",
            "outputs/",
            use_cuda=True,
            args=model_args
        )

    # Evaluate the model
    pdb.set_trace()
    result, model_outputs, wrong_predictions = model.eval_model("data/eval.tsv")

    # Make predictions with the model
    # predictions, raw_outputs = model.predict(["you are trash"])

if __name__ == "__main__":
    main()
