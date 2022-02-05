import os
import sys
from datetime import datetime
from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import dataset


_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# The size of the dense layer between LSTM output and classification layer
_conf_dense_size = 128
# How many words in a sequence for training/detection toxicity
_conf_seqence_length = 20
# Size of the hidden layer
_conf_hidden_size = 32

class ToxicDetectorModule(nn.Module):
    """Implements a LSTM recurrent NN to mark contiguous spans of toxic text.
    We will not distinguish among obscene/racist/insult, we will just mark the spans
    as we see them in the sentence."""

    def __init__(self, in_size: int, hid_size: int, seq_length: int):
        super().__init__()

        # Input size, the size of the word embedding vector
        self.input_size = in_size
        # Hidden state size
        self.hidden_size = hid_size
        # The sequence length
        self.seq_length = seq_length

        # One layer of LSTM cells
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        # Insert a fully connected layer between LSTM output and the classification layer
        self.dense_1 = nn.Linear(2 * self.hidden_size, _conf_dense_size)
        # This is the final classification layer
        # We only have 2 classes, 'offensive' and 'not offensive' => output vector of 2 elements
        self.dense_2 = nn.Linear(_conf_dense_size, 2)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p=0.33)
        # Place module on CUDA, if available
        self.to(device=_device)

    def forward(self, x):
        # Hidden state initialization
        h_0 = torch.zeros(
            2, x.size(0), self.hidden_size)
        h_0 = h_0.to(device=_device)
        # Internal state initialization
        c_0 = torch.zeros(2, x.size(0), self.hidden_size)
        c_0 = c_0.to(device=_device)
        
        # Propagate input through LSTM, get output and state information
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.tanh(out)
        out = self.drop(out)
        out = self.dense_1(out)
        out = self.tanh(out)
        out = self.dense_2(out)
        out = self.logsoftmax(out)

        return out


class ToxicTagger(object):
    """This is the toxic tagger, tagging each token in a given text with labels
    `1` for toxic and `0' for non-toxic."""
    
    def __init__(self, td_mod: ToxicDetectorModule, tx_proc: dataset.TextProcessor) -> None:
        self._model = td_mod
        self._processor = tx_proc
        self._optimizer = Adam(params=self._model.parameters(), lr=0.001)
        self._loss_fn = nn.NLLLoss()
        self._training_loader = None
        self._test_loader = None

    def load_dataset(self, data: TensorDataset, ml_type: str):
        if ml_type == 'train' or ml_type == 'training':
            self._training_loader = DataLoader(data, batch_size=4, shuffle=True)
        elif ml_type == 'test':
            self._test_loader = DataLoader(data, batch_size=4, shuffle=True)
        # end if

    def load_model(self, model_file: str):
        print(f'Loading pre-trained model {model_file}', file=sys.stderr, flush=True)
        self._model.load_state_dict(torch.load(model_file))
        self._model.train(False)

    def train(self, epochs: int = 5):
        """Main training method. Trains for a number of epochs equal to `epochs`."""

        now = datetime.now()
        timestamp = f'{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}'
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
            running_correct = 0
            running_total = 0

            for i, (tinputs, tlabels) in enumerate(self._test_loader):
                # Run model on the test inputs
                tinputs = tinputs.to(device=_device)
                tlabels = tlabels.to(device=_device)
                toutputs = self._model(tinputs)
                
                # Compute test loss
                toutputs = torch.swapaxes(toutputs, 1, 2)
                tloss = self._loss_fn(toutputs, tlabels)
                running_tloss += tloss.item()

                # Compute test accuracy
                _, tpredictions = torch.max(toutputs, dim=1)
                running_correct += (tpredictions == tlabels).sum().item()
                running_total += tlabels.shape[0] * tlabels.shape[1]
            # end for

            avg_tloss = running_tloss / (i + 1)
            print(
                f'Train loss = {avg_loss:.5f}, Test loss = {avg_tloss:.5f}', file=sys.stderr, flush=True)
            test_acc = running_correct / running_total
            print(f'Test accuracy = {test_acc:.5f}',
                  file=sys.stderr, flush=True)

            # Log the running loss averaged per batch
            # for both training and test
            tensorboard_writer.add_scalars('Training vs. Test loss', {
                                           'Training': avg_loss, 'Test': avg_tloss}, epoch_number)
            tensorboard_writer.add_scalar('Test accuracy', test_acc, epoch_number)
            tensorboard_writer.flush()

            # Track best performance and save the model's state
            if avg_tloss < best_tloss:
                best_tloss = avg_tloss
                model_path = os.path.join('models', 'model_{}_{}'.format(timestamp, epoch_number))
                torch.save(self._model.state_dict(), model_path)
            # end if
        # end for epochs

    def _train_one_epoch(self, epoch: int, tb_writer: SummaryWriter):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, (inputs, labels) in enumerate(self._training_loader):
            # Every data instance is an input + label pair
            # If CUDA is available, use it.
            inputs, labels = inputs.to(device=_device), labels.to(device=_device)
            
            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self._model(inputs)

            # Have to swap axes for NLLLoss function
            # Classes are on the second dimension, dim=1
            outputs = torch.swapaxes(outputs, 1, 2)

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
                print(f'  batch {i + 1}/{len(self._training_loader)} loss: {last_loss:.5f}',
                      file=sys.stderr, flush=True)
                tb_x = epoch * len(self._training_loader) + i + 1
                tb_writer.add_scalar('Train loss', last_loss, tb_x)
                tb_writer.flush()
                running_loss = 0.
            # end of

        return last_loss

    def run_toxic_tagger(self, text: str) -> tuple:
        """This is the main method of the tagger.
        Takes the text, processes it and returns a list of tokens
        with their labels."""

        text_toks, text_inputs = self._processor.process_text(text)
        text_inputs = text_inputs.to(device=_device)
        text_predictions = self._model(text_inputs)
        pred_dicts = []

        for _ in range(len(text_toks)):
            pred_dicts.append({'0': 0., '1': 0.})
        # end for

        # There are text_predictions.shape[0] sequences
        # of the given length (e.g. 10) in text
        # Loop over each one and accumulate probabilities for
        # classes 0 and 1
        for i in range(text_predictions.shape[0]):
            for j in range(self._processor.get_sequence_length()):
                # Network outputs LogSoftmax, so exp that back to get probabilities
                pred_ij = torch.exp(text_predictions[i, j, :])
                zero_class_prob = pred_ij[0].item()
                one_class_prob = pred_ij[1].item()
                dict_ij = pred_dicts[i + j]
                dict_ij['0'] += zero_class_prob
                dict_ij['1'] += one_class_prob
            # end for j
        # end for i

        text_labels = ['0'] * len(text_toks)

        for i in range(len(pred_dicts)):
            pd = pred_dicts[i]

            # Strategy 1: if accumulated prob for 1 is bigger, it's toxic
            #if pd['1'] > pd['0']:
            #    text_labels[i] = '1'
            # end if

            # Strategy 2: if label 1 (toxic) has been assigned in at least
            # two frames, it's toxic.
            if pd['1'] >= 1.:
                text_labels[i] = '1'
            # end if
        # end for

        return (text_toks, text_labels)


if __name__ == '__main__':
    
    # Load word embeddings to use as inputs
    we = dataset.WordEmbeddings()
    conf_input_size = we.get_vector_size()
    
    # This is the text processor, processing 10 words sequences
    txt_proc = dataset.TextProcessor(we, _conf_seqence_length)

    # The toxic detector NN PyTorch module
    toxic_detector_module = ToxicDetectorModule(
        conf_input_size, _conf_hidden_size, _conf_seqence_length)
    toxic_tagger = ToxicTagger(toxic_detector_module, txt_proc)

    if len(sys.argv) == 2 and sys.argv[1] == '-t':
        # Training model
        # Train and test sets
        td_train = dataset.ToxicData(
            os.path.join('data', 'tsd_train.csv'), txt_proc)
        td_test = dataset.ToxicData(
            os.path.join('data', 'tsd_test.csv'), txt_proc)
        # Train and test TensorDataset(s)
        train_tds = td_train.get_tensor_dataset()
        test_tds = td_test.get_tensor_dataset()

        # Load train/test datasets to train/test on
        toxic_tagger.load_dataset(train_tds, ml_type='train')
        toxic_tagger.load_dataset(test_tds, ml_type='test')

        # Do the training. Saving is done here
        toxic_tagger.train(epochs=3)
    elif len(sys.argv) == 3 and sys.argv[1] == '-e' and \
        os.path.isfile(sys.argv[2]):
        # Eval mode, on the test data
        td_test = dataset.ToxicData(
            os.path.join('data', 'tsd_test.csv'), txt_proc)
        toxic_tagger.load_model(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == '-r' and \
        os.path.isfile(sys.argv[2]) and os.path.isfile(sys.argv[3]):
        # Run mode, load model from the -r argument
        # and run on last file argument
        toxic_tagger.load_model(sys.argv[2])

        with open(sys.argv[3], mode='r', encoding='utf-8') as f:
            all_lines = ''.join(f.readlines())
        # end with

        tokens, labels = toxic_tagger.run_toxic_tagger(text=all_lines)

        # Print output
        for tok, lbl in zip(tokens, labels):
            print(f'{tok}\t{lbl}', flush=True)
        # end for
    else:
        print('Usage:\n  python|python3 offensive.py -t\n  python|python3 offensive.py -r <models/saved_model_file> <test_file.txt>')
    # end if
