import os
import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import dataset

class ToxicDetectorModule(nn.Module):
    """Implements a LSTM recurrent NN to mark contiguous spans of toxic text.
    We will not distinguish among obscene/racist/insult, we will just mark the spans
    as we see them in the sentence."""

    # The size of the dense layer between LSTM output and classification layer
    # It is configurable
    _conf_dense_size = 128

    def __init__(self, in_size: int, hid_size: int, seq_length: int):
        super().__init__()

        # Input size, the size of the word embedding vector
        self.input_size = in_size
        # Hidden state size
        self.hidden_size = hid_size
        # The sequence length
        self.seq_length = seq_length

        # One layer of LSTM cells
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        # Insert a fully connected layer between LSTM output and the classification layer
        self.dense_1 = nn.Linear(
            self.hidden_size, ToxicDetectorModule._conf_dense_size)
        # This is the final classification layer
        # We only have 2 classes, 'offensive' and 'not offensive' => output vector of 2 elements
        self.dense_2 = nn.Linear(ToxicDetectorModule._conf_dense_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Hidden state initialization
        h_0 = Variable(torch.zeros(
            1, x.size(0), self.hidden_size))
        # Internal state initialization
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size))

        # Propagate input through LSTM, get output and state information
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # Reshaping the data for dense layer next
        # Hout (== self.hidden_size) is the last dimension
        h_n = h_n.view(-1, self.hidden_size)
        out = self.relu(h_n)
        out = self.dense_1(out)
        out = self.relu(out)
        out = self.dense_2(out)
        toxic_probs = self.softmax(out)

        return toxic_probs


class ToxicTagger(object):
    """This is the toxic tagger, tagging each token in a given text with labels
    `1` for toxic and `0' for non-toxic."""
    
    def __init__(self, td_mod: ToxicDetectorModule, tx_proc: dataset.TextProcessor) -> None:
        self._model = td_mod
        self._processor = tx_proc
        self._optimizer = Adam(params=self._model.parameters(), lr=0.003)
        self._loss_fn = nn.CrossEntropyLoss()
        self._training_loader = None
        self._test_loader = None

    def load_dataset(self, data: TensorDataset, ml_type: str):
        if ml_type == 'train' or ml_type == 'training':
            self._training_loader = DataLoader(data, batch_size=8, shuffle=True)
        elif ml_type == 'test':
            self._test_loader = DataLoader(data, batch_size=8, shuffle=True)
        # end if

    def train(self, epochs: int = 5):
        """Main training method. Trains for a number of epochs equal to `epochs`."""

        now = datetime.now()
        timestamp = f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.min}'
        tensorboard_writer = SummaryWriter(os.path.join('runs', 'toxictagger-' + timestamp))
        best_tloss = 1_000_000.

        # Doing epochs, as instructed.
        for epoch in range(epochs):
            epoch_number = epoch + 1
            print(f'Epoch {epoch_number}', file=sys.stderr, flush=True)

            # Make sure gradient tracking is on, and do a pass over the data
            self._model.train(True)
            avg_loss = self._train_one_epoch(epoch_number, tensorboard_writer)
            # We don't need gradients on to do reporting
            self._model.train(False)

            # Compute test loss
            running_tloss = 0.0

            for i, tdata in enumerate(self._test_loader):
                tinputs, tlabels = tdata
                toutputs = self._model(tinputs)
                tloss = self._loss_fn(toutputs, tlabels)
                running_tloss += tloss
            # end for

            avg_tloss = running_tloss / (i + 1)
            print(f'Train loss = {avg_loss}, Test loss = {avg_tloss}', file=sys.stderr, flush=True)

            # Log the running loss averaged per batch
            # for both training and test
            tensorboard_writer.add_scalars('Training vs. Test loss', {
                                           'Training': avg_loss, 'Test': avg_tloss}, epoch_number)
            tensorboard_writer.flush()

            # Track best performance and save the model's state
            if avg_tloss < best_tloss:
                best_tloss = avg_tloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self._model.state_dict(), model_path)
            # end if
        # end for epochs

    def _train_one_epoch(self, epoch: int, tb_writer: SummaryWriter):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self._training_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._model(inputs)

            # Compute the loss and its gradients
            loss = self._loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if (i + 1) % 1000 == 0:
                # loss per batch
                last_loss = running_loss / 1000  
                print(f'  batch {i + 1} loss: {last_loss}', file=sys.stderr, flush=True)
                tb_x = epoch * len(self._training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.flush()
                running_loss = 0.
            # end of

        return last_loss


if __name__ == '__main__':
    # 10 words in a sequence for training/detection toxicity
    conf_seqence_length = 10
    
    # Load word embeddings to use as inputs
    # Change the path here to point to your cc.en.300.vec file!
    # TODO: automatically download the file from https://fasttext.cc/docs/en/crawl-vectors.html
    we = dataset.WordEmbeddings(we_file='D:\\Temp\\cc.en.300.vec')
    conf_input_size = we.get_vector_size()
    
    # This is the text processor, processing 10 words sequences
    txt_proc = dataset.TextProcessor(we, conf_seqence_length)
    # Train and test sets
    td_train = dataset.ToxicData(
        os.path.join('data', 'tsd_train.csv'), txt_proc)
    td_test = dataset.ToxicData(os.path.join('data', 'tsd_test.csv'), txt_proc)
    # Train and test TensorDataset(s)
    train_tds = td_train.get_tensor_dataset()
    test_tds = td_test.get_tensor_dataset()

    # The toxic detector NN PyTorch module
    toxic_detector_module = ToxicDetectorModule(conf_input_size, 64, conf_seqence_length)
    toxic_tagger = ToxicTagger(toxic_detector_module, txt_proc)
    # Load train/test datasets to train/test on
    toxic_tagger.load_dataset(train_tds, ml_type='train')
    toxic_tagger.load_dataset(test_tds, ml_type='test')

    # Do the training
    toxic_tagger.train()
