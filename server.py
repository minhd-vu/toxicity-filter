
#!/usr/local/bin/python3

from simpletransformers.classification import ClassificationModel


def main():
    model = ClassificationModel(
        "roberta", "outputs/checkpoint-183643-epoch-1",
        use_cuda=False
    )

    while True:
        text = input("> ")
        predictions, raw_outputs = model.predict([text])
        print(predictions, raw_outputs)


if __name__ == "__main__":
    main()
