import os
import sys
import re
import gzip
import pandas as pd
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import TensorDataset


_comma_split_rx = re.compile(',\\s*')
_whitespace_rx = re.compile('\\s+')


class WordEmbeddings(object):
    """Reads the pre-trained word embeddings `cc.en.300.vec` file into a dictionary and makes it available.
    Get the English file from https://fasttext.cc/docs/en/crawl-vectors.html.
    """

    def __init__(self) -> None:
        self._load_word_embeddings()

    def _load_word_embeddings(self) -> dict:
        self._embeddings = {}
        we_file = os.path.join('data', 'cc.en.300.vec.gz')

        with gzip.open(we_file, mode='rt', encoding='utf-8') as f:
            line = f.readline()
            line = line.strip()
            word_count, vec_dim = [int(x) for x in line.split()]
            self._dimension = vec_dim
            counter = 1

            for line in f:
                line = line.rstrip()
                parts = line.split()
                word = parts[0]
                wvec = ' '. join(parts[1:])
                self._embeddings[word] = wvec

                if counter % 100000 == 0:
                    print(
                        f'Read {counter}/{word_count} words from file {we_file}')
                # end if

                counter += 1
            # end for
        # end with

    def get_word_embedding_vector(self, word: str) -> list:
        if word in self._embeddings:
            return [float(x) for x in self._embeddings[word].split(' ')]
        elif word.lower() in self._embeddings:
            return [float(x) for x in self._embeddings[word.lower()].split(' ')]
        else:
            return [0.5] * self._dimension
        # end if

    def get_vector_size(self) -> int:
        return self._dimension


class TextProcessor(object):
    """Takes a piece of text and returns a tensor for processing with the NN.
    If ground truth labels are also provided, returns the corresponding tensor as well.
    Otherwise, returns a tensor fo 0s."""

    unknown_word = '_UNK'

    def __init__(self, wd_emb: WordEmbeddings, seq_len: int) -> None:
        self._wordembeddings = wd_emb
        self._sequencelen = seq_len

    def get_sequence_length(self) -> int:
        """Returns the configured sequence length of this object."""
        return self._sequencelen

    def process_text(self, text: str) -> tuple:
        """Takes a simple `text` and returns a tuple of PyTorch tensors."""

        tokens = word_tokenize(text)
        labels = [0] * len(tokens)
        text_inputs, _ = self.process_training_tokens(tokens, labels)

        return (tokens, text_inputs)

    def process_training_tokens(self, tokens: list, labels: list) -> tuple:
        while len(tokens) < self._sequencelen:
            tokens.append(TextProcessor.unknown_word)
            labels.append(0)
        # end if

        text_embeddings = []
        ground_truth_labels = []

        for i in range(0, len(tokens) - self._sequencelen + 1):
            sequence = tokens[i:i + self._sequencelen]
            seq_embeddings = list(
                map(self._wordembeddings.get_word_embedding_vector, sequence))
            seq_labels = labels[i:i + self._sequencelen]
            text_embeddings.append(seq_embeddings)
            ground_truth_labels.append(seq_labels)
        # end for

        text_embeddings_tensor = torch.tensor(
            text_embeddings, dtype=torch.float32)
        labels_tensor = torch.tensor(ground_truth_labels, dtype=torch.long)

        return (
            # Shape of this is (batch_size, seq_len, 300)
            # 300 is the size of the embedding vector
            text_embeddings_tensor,
            # Shape of this is (batch_size, seq_len, 2)
            # We only have 2 classes
            labels_tensor
        )


class ToxicData(object):
    """Reads in the train/dev/test datasets from the ``Toxic Spans Detection (SemEval 2021 Task 5)'' competition.
    More details here: https://github.com/ipavlopoulos/toxic_spans."""

    def __init__(self, csv_file: str, txt_proc: TextProcessor) -> None:
        self._data = self._read_file(csv_file)
        self._processor = txt_proc
        self.dataset = None

    def _process_spans(self, spans: str) -> list[int]:
        """Takes a [1,2,3,7,8,9] string and returns a list[int] of offsets."""

        spans = spans.strip()

        if spans.startswith('['):
            spans = spans[1:]
        # end if

        if spans.endswith(']'):
            spans = spans[:-1]
        # end if

        result = []

        if not spans:
            return result
        # end if

        offsets = _comma_split_rx.split(spans)
        offsets = [int(x) for x in offsets]

        return offsets

    def _read_file(self, file: str) -> list:
        """Reads a Toxic Spans Detection .csv file and creates training examples, with lists of
        words and their corresponding 0/1 labels. For instance:
        Sentence: `['This', 'stupid', 'so-called', '"', 'president', '"', ...]`
        Labels: `[0, 1, 0, 0, 0, 0, ...]`"""

        df = pd.read_csv(file)
        toxic_data = []
        
        for k in range(df.shape[0]):
            # Get 'spans' and 'text' fields from the dataframe
            spans = df.iloc[k, :]['spans']
            proc_spans = self._process_spans(spans)
            text = df.iloc[k, :]['text']
            # Tokenize input text so that we can use pretrained word embeddings
            tokens = word_tokenize(text)
            labels = [0] * len(tokens)
            # Transfer span annotation into 0/1 labels, for each word
            current_offset = 0

            for i in range(len(tokens)):
                if not proc_spans:
                    break
                # end if

                for _ in range(len(tokens[i])):
                    if current_offset >= proc_spans[0]:
                        labels[i] = 1
                        proc_spans.pop(0)

                        if not proc_spans:
                            break
                        # end if
                    # end if

                    current_offset += 1
                # end for

                # Skip whitespace, and drop whitespace annotations
                while current_offset < len(text) and \
                        _whitespace_rx.fullmatch(text[current_offset]):
                    current_offset += 1

                    if current_offset >= proc_spans[0]:
                        proc_spans.pop(0)

                        if not proc_spans:
                            break
                        # end if
                    # end if
                # end while
            # end for

            if proc_spans:
                print(f'Warning: remaining toxic text spans for frame {k} in file {file}', file=sys.stderr, flush=True)
            # end if

            toxic_data.append((tokens, labels))
        # end for

        return toxic_data

    def get_parsed_data(self) -> list:
        return self._data

    def get_tensor_dataset(self) -> TensorDataset:
        """Builds a TensorDataset object out of the annotations in this file."""

        if self.dataset is not None:
            return self.dataset
        # end if

        inputs = []
        labels = []

        for counter, (toks, labs) in enumerate(self._data):
            in_tens, lb_tens = self._processor.process_training_tokens(
                tokens=toks, labels=labs)
            inputs.append(in_tens)
            labels.append(lb_tens)

            if counter >= 1 and counter % 100 == 0:
                print(f'Built {counter}/{len(self._data)} tensors')
            # end if
        # end all dataset

        all_inputs_tensor = torch.cat(inputs)
        all_labels_tensor = torch.cat(labels)

        return TensorDataset(all_inputs_tensor, all_labels_tensor)
