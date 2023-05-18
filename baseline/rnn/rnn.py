import codecs
from torch import nn
import torch
import random
import sys

torch.manual_seed(0)
PAD = "PAD"
DIM_EMBEDDING = 100
LSTM_HIDDEN = 50
BATCH_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_data(file_name):
    data = []
    current_words = []
    current_tags = []

    token_map = {'B-scientist': 'B-musicalartist', 'I-scientist': 'I-musicalartist',
                 'B-university': 'B-band', 'I-university': 'I-band',
                 'B-astronomicalobject': 'B-musicalinstrument', 'I-astronomicalobject': 'I-musicalinstrument',
                 'B-protein': 'B-song', 'I-protein': 'I-song',
                 'B-chemicalcompound': 'B-musicgenre', 'I-chemicalcompound': 'I-musicgenre',
                 'B-academicjournal': 'B-album', 'I-academicjournal': 'I-album',
                 'B-enzyme': 'B-song', 'I-enzyme': 'I-song',
                 'B-discipline': 'B-musicgenre', 'I-discipline': 'I-musicgenre',
                 'B-theory': 'B-musicgenre', 'I-theory': 'I-musicgenre',
                 'B-chemicalelement': 'B-musicgenre', 'I-chemicalelement': 'I-musicgenre'}

    for line in codecs.open(file_name, encoding='utf-8'):

        line = line.strip()

        if line:
            if line[0] == '#':
                continue  # skip comments
            tok = line.split('\t')
            word = tok[0]
            tag = tok[1]
            if tag in token_map:
                tag = token_map[tag]
            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    if current_tags != []:
        data.append((current_words, current_tags))
    return data


print(sys.argv)
train_data = read_data(sys.argv[1])
max_len = max([len(x[0]) for x in train_data])

test_data = read_data(sys.argv[2])
output_file = sys.argv[3]
# Create vocabularies for both the tokens
# and the tags
id_to_token = [PAD]
token_to_id = {PAD: 0}
id_to_tag = [PAD]
tag_to_id = {PAD: 0}
for tokens, tags in train_data:
    for token in tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
            id_to_token.append(token)
    for tag in tags:
        if tag not in tag_to_id:
            tag_to_id[tag] = len(tag_to_id)
            id_to_tag.append(tag)
NWORDS = len(token_to_id)
NTAGS = len(tag_to_id)


# convert text data with labels to indices
def data2feats(inputData, word2idx, label2idx):
    feats = torch.zeros((len(inputData), max_len), dtype=torch.long)
    labels = torch.zeros((len(inputData), max_len), dtype=torch.long)
    for sentPos, sent in enumerate(inputData):
        for wordPos, word in enumerate(sent[0][:max_len]):
            wordIdx = word2idx[PAD] if word not in word2idx else word2idx[word]
            feats[sentPos][wordPos] = wordIdx
        for labelPos, label in enumerate(sent[1][:max_len]):
            labelIdx = word2idx[PAD] if label not in label2idx else label2idx[label]
            labels[sentPos][labelPos] = labelIdx
    return feats, labels


train_feats, train_labels = data2feats(train_data, token_to_id, tag_to_id)
# convert to batches
num_batches = int(len(train_feats) / BATCH_SIZE)
train_feats_batches = train_feats[:BATCH_SIZE * num_batches].view(num_batches, BATCH_SIZE, max_len)
train_labels_batches = train_labels[:BATCH_SIZE * num_batches].view(num_batches, BATCH_SIZE, max_len)


class TaggerModel(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super().__init__()
        # Create word embeddings
        self.word_embedding = nn.Embedding(nwords, DIM_EMBEDDING)
        # Create input dropout parameter
        self.word_dropout = torch.nn.Dropout(.2)
        # Create LSTM parameters
        self.rnn = torch.nn.RNN(DIM_EMBEDDING, LSTM_HIDDEN, num_layers=1,
                                batch_first=True, bidirectional=False)
        # Create output dropout parameter
        self.rnn_output_dropout = torch.nn.Dropout(.3)
        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN, ntags)

    def forward(self, sentences):
        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        dropped_word_vectors = self.word_dropout(word_vectors)
        rnn_out, _ = self.rnn(dropped_word_vectors, None)
        # Apply dropout
        rnn_out_dropped = self.rnn_output_dropout(rnn_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(rnn_out_dropped)
        # Calculate loss and predictions
        return output_scores


model = TaggerModel(NWORDS, NTAGS)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
for epoch in range(EPOCHS):
    model.train()
    model.zero_grad()
    # Loop over batches
    loss = 0
    match = 0
    total = 0
    for batchIdx in range(0, num_batches):
        output_scores = model.forward(train_feats_batches[batchIdx])
        output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
        flat_labels = train_labels_batches[batchIdx].view(BATCH_SIZE * max_len)
        batch_loss = loss_function(output_scores, flat_labels)
        predicted_tags = torch.argmax(output_scores, 1)
        predicted_tags = predicted_tags.view(BATCH_SIZE, max_len)
        # Prepare inputs
        input_array = train_feats_batches[batchIdx]
        output_array = train_labels_batches[batchIdx]
        # Construct computation
        output_scores = model(input_array)
        # Calculate loss
        output_scores = output_scores.view(BATCH_SIZE * max_len, -1)
        flat_labels = output_array.view(BATCH_SIZE * max_len)
        batch_loss = loss_function(output_scores, flat_labels)
        # Run computations
        batch_loss.backward()
        optimizer.step()
        model.zero_grad()
        loss += batch_loss.item()
        # Update the number of correct tags and total tags
        for goldSent, predSent in zip(train_labels_batches[batchIdx], predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel != 0:
                    total += 1
                    if goldLabel == predLabel:
                        match += 1
    print(epoch, loss, match / total)


def run_eval(feats_batches, labels_batches):
    model.eval()
    match = 0
    total = 0
    all_predictions = []
    for sents, labels in zip(feats_batches, labels_batches):
        output_scores = model.forward(sents)
        predicted_tags = torch.argmax(output_scores, 2)
        all_predictions.append(predicted_tags)
        for goldSent, predSent in zip(labels, predicted_tags):
            for goldLabel, predLabel in zip(goldSent, predSent):
                if goldLabel.item() != 0:
                    total += 1
                    if goldLabel.item() == predLabel:
                        match += 1
    return match / total, all_predictions


def save_predictions_to_file(data, predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        num_sentences = len(data)
        for i, (tokens, _) in enumerate(data):
            pred_tags = predictions[i].squeeze().tolist()
            for token, pred_tag in zip(tokens, pred_tags):
                pred_tag = id_to_tag[pred_tag]
                f.write(f"{token}\t{pred_tag}\n")
            if i < num_sentences - 1:  # Check if not the last sentence
                f.write("\n")


if __name__ == "__main__":
    dev_feats, dev_labels = data2feats(test_data, token_to_id, tag_to_id)
    num_batches2 = int(len(dev_feats) / BATCH_SIZE)
    score, predictions = run_eval(dev_feats.view(num_batches2, BATCH_SIZE, max_len),
                                  dev_labels.view(num_batches2, BATCH_SIZE, max_len))
    save_predictions_to_file(test_data, predictions, output_file)
