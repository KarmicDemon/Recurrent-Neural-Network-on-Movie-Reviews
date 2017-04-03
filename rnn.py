import numpy as np
import tensorflow as tf

from collections import Counter
from string import punctuation

# load reviews and labels
with open('reviews.txt', 'r') as f:
    reviews = f.read()
with open('labels.txt', 'r') as f:
    labels = f.read()

#remove punctuation from reviews, only use words
reviews = ''.join([c for c in reviews if c not in punctuation]).split('\n')
words = ' '.join(reviews).split()

#map words to integers by declining appearance
vocab_to_int = Counter(words)
vocab_to_int = sorted(vocab_to_int, key = vocab_to_int.get, reverse = True)
vocab_to_int = { word: order for order, word in enumerate(vocab_to_int, 1)}

#turn reviews into numbers
review_ints = []
for each in reviews:
    review_ints.append([vocab_to_int[word] for word in each.split()])

#convert labels to numbers
labels = np.array([1 if _ == 'positive' else 0 for _ in labels.split('\n')])

#remove empty reviews if there
non_zero_idx = [_ for _, review in enumerate(review_ints) if len(review) != 0]
review_ints = [review_ints[_] for _ in non_zero_idx]
labels = np.array([labels[_] for _ in non_zero_idx])

#truncate so that rnn only recognizes 200 words
features = np.zeros((len(review_ints), 200), dtype = int)
for _, row in enumerate(review_ints):
    features[_, -len(row):] = np.array(row)[:200]

_ = int(len(features) * .8)
train_x, val_x = features[:_], features[_:]
train_y, val_y = labels[:_], labels[_:]

_ = int(len(val_x) * .5)
val_x, test_x = val_x[:_], val_x[_:]
val_y, test_y = val_y[:_], val_y[_:]

num_words = len(vocab_to_int)

#define hyperparameters
batch_size = 500
embed_size = 300
epochs = 10
learning_rate = .001
lstm_layers = 1
lstm_size = 256

#define generator batch function
def get_batches(x, y, batch_size = 100):
    n_batches = len(x) // batch_size

    x, y = x[:(n_batches * batch_size)], y[:(n_batches * batch_size)]
    for i in range(0, len(x), batch_size):
        yield x[i:(i + batch_size)], y[i:(i + batch_size)]

#create network
nn_graph = tf.Graph()
with nn_graph.as_default():
    _inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    _labels = tf.placeholder(tf.int32, [None, None], name = 'labels')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    embedding = tf.Variable(tf.random_uniform((num_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, _inputs)

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, \
        initial_state = initial_state)

    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, \
        activation_fn = tf.sigmoid)
    cost = tf.losses.mean_squared_error(_labels, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), _labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

#pass through rnn
with tf.Session(graph = nn_graph) as s:
    s.run(tf.global_variables_initializer())
    _iter = 1

    for e in range(epochs):
        state = s.run(initial_state)

        for i, (x, y) in \
            enumerate(get_batches(train_x, train_y, batch_size), 1):

            feed = {
                _inputs : x,
                _labels : y[:, None],
                keep_prob: 0.5,
                initial_state: state
            }

            loss, state, _ = s.run([cost, final_state, optimizer], \
                feed_dict = feed)

            if _iter % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                    "Iteration: {}".format(_iter),
                    "Train loss: {:.3f}".format(loss))

            if _iter % 25 == 0:
                val_acc = []
                val_state = s.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {
                        _inputs : x,
                        _labels : y[:, None],
                        keep_prob: 1,
                        initial_state: val_state
                    }

                    batch_acc, val_state = s.run([accuracy, final_state], \
                        feed_dict = feed)
                    val_acc.append(batch_acc)

                print("Validation Accuracy: {:.3f}".format(np.mean(val_acc)))

            _iter += 1

    saver.save(s, 'checkpoints/sentiment.ckpt')

# Test
test_acc = []
with tf.Session(graph = nn_graph) as s:
    saver.restore(s, tf.train.latest_checkpoint('/output/checkpoints'))
    test_state = s.run(cell.zero_state(batch_size, tf.float32))

    for i, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {
            _inputs : x,
            _labels : y[:, None],
            keep_prob : 1,
            initial_state : test_state
        }

        batch_acc, test_state = s.run([accuracy, final_state], feed_dict = feed)
        test_acc.append(batch_acc)

    print("Testing Accuracy: {:.3f}".format(np.mean(test_acc)))
