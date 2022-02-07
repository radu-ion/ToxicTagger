## Installation
Install all requirements with:

`pip3 install -r requirements.txt`

Download the `nltk` English tokenizer:

`python3 -c "import nltk; nltk.download('punkt')"`

Download the English fastText word embeddings file from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz and
save the file under folder `data/` of this repository. I will automate this, eventually.

## Running

Execute a test run as follows:

`python3 offensive.py -r models/<latest model> test_file.txt`

All output is printed to the `sys.stdout` file.

## About

A simple LSTM-based neural network, written in PyTorch 1.10, that detects contiguous spans of "toxic text".
By "toxic text" we understand insulting/threating/identity-based attacking/obscene phrases.
Training data comes from the ``Semeval-2021 Task 5: Toxic Spans Detection'' competition.
The link to the relevant paper is here: https://aclanthology.org/2021.semeval-1.6.pdf.
