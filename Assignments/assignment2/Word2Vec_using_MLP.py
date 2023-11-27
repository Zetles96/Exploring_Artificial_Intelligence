import re  # python's regular expression library.  A powerful library for text manipulation
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import Sequential
from keras.layers import Dense


def generate_variables(cleaned_text):
    """
    Generates the four required variables above.
    Args:
      cleaned_text: the cleaned text in the step above
    Returns:
      words_in_order, num_vocabs, word_to_index, index_to_word
    """
    # Split the cleaned text into words
    words = cleaned_text.split()

    # Create a list of all words in the order they appeared
    words_in_order = words

    # Get the number of unique words (vocabulary size)
    num_vocabs = len(set(words))

    # Create a dictionary mapping each unique word to an integer
    unique_words = sorted(set(words))
    word_to_index = {word: index for index, word in enumerate(unique_words)}

    # Create a dictionary mapping each integer back to the word
    index_to_word = {index: word for index, word in enumerate(unique_words)}

    return words_in_order, num_vocabs, word_to_index, index_to_word


def generate_training_data_cbow(words_in_order, word_to_index, window_size):
    """
    Generates the target word table and context word table based on the continuous bag of words approach.
    The table should contain the index of the word instead of the word itself.
    For words in the start and end of the whole text where the number of context words are less than window_size * 2,
    fill up the remaining entries of the array with 'special number' -1 instead.

    Args:
      words_in_order: the list of words appearing in order that we created in the previous step
      word_to_index: the dictionary that we created in the previous step
      window_size: how far we scan for context words
    Returns:
      a tuple of 2 numpy arrays
    """
    vocab_size = len(word_to_index)
    special_number = -1

    x_cbow = []
    y_cbow = []

    for i, target_word in enumerate(words_in_order):
        context_words = []

        # Collect context words within the window size
        start_index = max(0, i - window_size)
        end_index = min(len(words_in_order), i + window_size + 1)
        context_words = [word_to_index.get(words_in_order[j], special_number) for j in range(start_index, end_index) if
                         j != i]

        # Fill up missing context words if the window size is not met
        while len(context_words) < window_size * 2:
            context_words.append(special_number)

        # Append the target and context words to the training data
        x_cbow.append(context_words)
        y_cbow.append(word_to_index[target_word])

    # Convert lists to NumPy arrays
    x_cbow = np.array(x_cbow)
    y_cbow = np.array(y_cbow)

    return x_cbow, y_cbow


def generate_training_data_skipgram(words_in_order, word_to_index, window_size):
    """
    Generates the target word table and context word table based on the skipgram approach.
    The table should contain the index of the word instead of the word itself.

    Args:
      words_in_order: the list of words appearing in order that we created in the previous step
      word_to_index: the dictionary that we created in the previous step
      window_size: how far we scan for context words
    Returns:
      a tuple of 2 numpy arrays
    """
    vocab_size = len(word_to_index)

    x_skipgram = []
    y_skipgram = []

    for i, target_word in enumerate(words_in_order):
        # Context words within the window size
        start_index = max(0, i - window_size)
        end_index = min(len(words_in_order), i + window_size + 1)
        context_words = [word_to_index.get(words_in_order[j], -1) for j in range(start_index, end_index) if j != i]

        # Create training pairs for skipgram
        for context_word in context_words:
            x_skipgram.append(word_to_index[target_word])
            y_skipgram.append(context_word)

    # Convert lists to NumPy arrays
    x_skipgram = np.array(x_skipgram)
    y_skipgram = np.array(y_skipgram)

    return x_skipgram, y_skipgram


def one_hot(arr, num_vocabs):
    """
    Takes in a 1D array and generates its one-hot representation.
    Args:
      arr: a 1D numpy array
      num_vocabs: the number of unique vocabs
    Returns:
      a 2D numpy array of the one-hot representation of arr
    """
    # Create a 2D array of zeros with shape (len(arr), num_vocabs)
    one_hot_arr = np.zeros((len(arr), num_vocabs), dtype=int)

    # Fill in the one-hot representation
    one_hot_arr[np.arange(len(arr)), arr] = 1

    return one_hot_arr


def one_hot_multicategorical(arr, num_vocabs):
    """
    Takes in a 2D array and generates its multicategorical one-hot representation.
    Args:
      arr: a 2D numpy array
      num_vocabs: the number of unique vocabs
    Returns:
      a 2D numpy array of the multicategorical one-hot representation of arr
    """
    rows, cols = arr.shape
    one_hot_matrix = np.zeros((rows, num_vocabs), dtype=int)
    one_hot_matrix[np.arange(rows)[:, np.newaxis], arr] = 1

    return one_hot_matrix


def create_cbow_model(num_vocabs, dims_to_learn=50):
    model = Sequential()
    model.add(Dense(dims_to_learn, use_bias=False, input_shape=(num_vocabs,)))
    model.add(Dense(num_vocabs, activation='softmax', use_bias=False))
    return model


def create_skipgram_model(num_vocabs, dims_to_learn=50):
    model = Sequential()
    model.add(Dense(dims_to_learn, use_bias=False, input_shape=(num_vocabs,)))
    model.add(Dense(num_vocabs, activation='softmax', use_bias=False))
    return model


def get_embeddings(words, weights, word_to_index, num_vocabs):
    """
    Takes in a list of strings and returns an array of embeddings corresponding to those strings.
    Args:
      words: a list of words in which we want to get the embedding for
      weights: the embeddings learned by a model
      word_to_index: the dictionary used to map the words back to integers
      num_vocabs: the number of unique vocabs
    Returns:
      a 2D numpy array, where the n-th row corresponds to the embeddings of the n-th word in the list "words"
    """
    # Initialize an empty numpy array to store the embeddings
    embeddings = np.zeros((len(words), weights.shape[1]))

    # Iterate through the list of words
    for i, word in enumerate(words):
        # Check if the word is in the word-to-index dictionary
        if word in word_to_index:
            # Get the index of the word
            word_index = word_to_index[word]
            # Get the corresponding embedding from the weights array
            word_embedding = weights[word_index]
            # Store the embedding in the embeddings array
            embeddings[i] = word_embedding

    return embeddings


def plot_semantic_space(dim1, dim2, words, weights, word_to_index, num_vocabs):
    """
    Plots dim1 against dim2 of the embedding of the words.
    Args:
      dim1: the first dimension to slice
      dim2: the second dimension to slice
      words: a list of words to plot
      weights: the embeddings learned by a model
    """
    coordinates = get_embeddings(words, weights, word_to_index, num_vocabs)[:, [dim1, dim2]]
    x_points = coordinates[:, 0]
    y_points = coordinates[:, 1]

    # use matplotlib to plot a scatter plot of the points
    plt.figure(figsize=(15, 15))
    plt.scatter(x_points, y_points)
    # label the points with the word corresponding to it
    for i in range(0, len(words)):
        plt.annotate(text=words[i], xy=coordinates[i])


def context_words(word, top=10):
    """
    Finds the top few (default is 10) context words closest to the input word
    Args:
      word: the target word
      top:  specifies how many context words to find
    Returns:
      a list strings, which are the top few context words
    """
    # get the one-hot encoding of the word
    word_1hot = one_hot(np.array([word_to_index[word]]), num_vocabs)
    # get the model output
    output = model_skipgram(word_1hot)[0]
    # get an array for the would-be sorted the output, from closest context words to furthest away context words
    sorted = np.argsort(-output)
    # check out which words they are
    return [index_to_word[s] for s in sorted][0:top]


if __name__ == '__main__':
    text = '''In mathematics, an open set is a generalization of an open interval in the real line.
  In a metric space (a set along with a distance defined between any two points),
  an open set is a set that, along with every point P, contains all points that are sufficiently near to P
  (that is, all points whose distance to P is less than some value depending on P).
  More generally, an open set is a member of a given collection of subsets of a given set,
  a collection that has the property of containing every union of its members, every finite intersection of its members,
  the empty set, and the whole set itself. A set in which such a collection is given is called a topological space,
  and the collection is called a topology. These conditions are very loose, and allow enormous flexibility in the choice of open sets.
  For example, every subset can be open (the discrete topology), or no subset can be open except the space itself and the empty set
  (the indiscrete topology). In practice, however, open sets are usually chosen to provide a notion of nearness that is
  similar to that of metric spaces, without having a notion of distance defined. In particular,
  a topology allows defining properties such as continuity, connectedness, and compactness, which were originally defined by means of a distance.'''

    cleaning = text.lower()  # CAPS -> caps
    cleaning = re.sub(r'\n', ' ', cleaning)  # replace the newline characters with a space
    cleaning = re.sub(r'[^a-z0-9 -]', '',
                      cleaning)  # replace all characters except letters, numbers, space bars and hyphens with an empty string
    cleaned_text = re.sub(' +', ' ', cleaning)  # replace multiple spaces with one space

    words_in_order, num_vocabs, word_to_index, index_to_word = generate_variables(cleaned_text)
    x_cbow, y_cbow = generate_training_data_cbow(words_in_order, word_to_index, 2)
    x_skipgram, y_skipgram = generate_training_data_skipgram(words_in_order, word_to_index, 2)

    x_cbow_1hot = one_hot_multicategorical(x_cbow, num_vocabs)
    y_cbow_1hot = one_hot(y_cbow, num_vocabs)
    x_skipgram_1hot = one_hot(x_skipgram, num_vocabs)
    y_skipgram_1hot = one_hot(y_skipgram, num_vocabs)

    model_cbow = create_cbow_model(num_vocabs)
    model_cbow.summary()
    model_skipgram = create_skipgram_model(num_vocabs)
    model_skipgram.summary()

    epochs = 500
    model_cbow.compile(loss='categorical_crossentropy')
    model_cbow.fit(x_cbow_1hot, y_cbow_1hot, epochs=epochs, verbose=0)

    model_skipgram.compile(loss='categorical_crossentropy')
    model_skipgram.fit(x_skipgram_1hot, y_skipgram_1hot, epochs=epochs, verbose=0)

    # extract the weights from the model
    weights_cbow = model_cbow.layers[0].get_weights()[0]
    weights_skipgram = model_skipgram.layers[0].get_weights()[0]
    # visualization
    plot_semantic_space(10, 20, words_in_order, weights_skipgram, word_to_index, num_vocabs)

    for w in ['open', 'set', 'space']:
        print(f"The 10 closest context words for '{w}' are: \t {context_words(w)}")
