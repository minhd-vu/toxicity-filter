#!/usr/local/bin/python3
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from functions import *


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Preparing train data
    train_df = pd.read_csv("data/jigsaw-unintended-bias-in-toxicity-classification/train.csv")
    train_df = train_df[["comment_text", "target"]]
    train_df = clean_text(train_df, "comment_text")
    train_df["target"] = class_labels(train_df["target"])
    train_df.columns = ["text", "labels"]
    eval_df = train_df[-200:]
    train_df = train_df[:200]

    # Preparing eval data

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)

    # Create a ClassificationModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=False,
        args=model_args,
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    # Make predictions with the model
    # predictions, raw_outputs = model.predict(["you are trash"])

if __name__ == "__main__":
    main()
