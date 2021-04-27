# toxicity-filter

## Dependencies
```sh
mkdir data
cd data
pip3 install kaggle simpletransformers
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
```

## Training
```sh
python3 baseline.py
```

## Running
```sh
python3 server.py
```