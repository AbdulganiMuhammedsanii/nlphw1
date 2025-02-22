# Name(s): Abdulgani Muhammedsani, Edwin Dake
# Netid(s): amm546, ed433
################################################################################
# NOTE: Do NOT change any of the function headers and/or specs!
# The input(s) and output must perfectly match the specs, or else your
# implementation for any function with changed specs will most likely fail!
################################################################################

################### IMPORTS - DO NOT ADD, REMOVE, OR MODIFY ####################
import numpy as np
import math
from collections import Counter, defaultdict


def handle_unknown_words(t, documents):
    """
    Replaces tokens in the given documents with <unk> (unknown) tokens if they occur
    less frequently than a certain threshold, based on the provided parameter 't'.
    Tokens are ordered first by frequency then alphabetically, so the tokens
    replaced are the least frequent tokens and earliest alphabetically.

    Input:
        t (float):
            A value between 0 and 1 representing the threshold for token frequency.
            The int(t * total_unique_tokens) least frequent tokens will be replaced.
        documents (list of lists):
            A list of documents, where each document is represented as a list of tokens.
    Output:
        new_documents (list of lists):
            A list of processed documents where the int(t * total_unique_tokens) least
            frequent tokens have been replaced with <unk> tokens and no other changes.
        vocab (list):
            A list of tokens representing the vocabulary, including both the most common tokens
            and the <unk> token.
    Example:
    t = 0.3
    documents = [["apple", "banana", "apple", "orange"],
                 ["apple", "cherry", "banana", "banana"],
                 ["cherry", "apple", "banana"]]
    new_documents, vocab = handle_unknown_words(t, documents)
    # new_documents:
    # [['apple', 'banana', 'apple', '<unk>'],
    #  ['apple', 'cherry', 'banana', 'banana'],
    #  ['cherry', 'apple', 'banana']]
    # vocab: ['banana', 'apple', 'cherry', '<unk>']
    """
    # YOUR CODE HERE
    # flatten documents and obtain frequencies of tokens
    counts = [Counter(document) for document in documents]
    frequencies = Counter()
    for count in counts:
        frequencies += count
    # unique tokens
    total_unique_tokens = len(frequencies)
    # find threshold for tokens to replace
    threshold = max(1, int(t * total_unique_tokens))
    # sort tokens by frequency then alphabetically
    sorted_tokens = sorted(frequencies, key=lambda token: (frequencies[token], token))
    # identify tokens we need to replace
    tokens_to_replace = set(sorted_tokens[:threshold])

    UNK_TOKEN = "<unk>"

    # Replace tokens in documents
    new_documents = []
    for document in documents:
        new_doc = [
            UNK_TOKEN if token in tokens_to_replace else token for token in document
        ]
        new_documents.append(new_doc)

    # Construct vocabulary (all remaining tokens + <unk>)
    vocab = sorted(set(token for doc in new_documents for token in doc))

    return new_documents, vocab


# Test Case 1: Basic Example
documents_1 = [
    ["apple", "banana", "apple", "orange"],
    ["apple", "cherry", "banana", "banana"],
    ["cherry", "apple", "banana"],
]
t_1 = 0.3

# Test Case 2: All Tokens are Equally Frequent
t_2 = 0.5
documents_2 = [
    ["dog", "cat", "mouse"],
    ["elephant", "cat", "dog"],
    ["mouse", "elephant", "dog"],
]

# Test Case 3: High Threshold (Replace Most Tokens)
t_3 = 0.75
documents_3 = [
    ["red", "blue", "green", "yellow"],
    ["red", "blue", "purple"],
    ["green", "purple", "yellow"],
]

# Test Case 4: No Tokens Replaced (t = 0)
t_4 = 0.0
documents_4 = [
    ["python", "java", "c++"],
    ["python", "ruby", "java"],
    ["c++", "ruby", "python"],
]

# Test Case 5: All Tokens Replaced (t = 1.0)
t_5 = 1.0
documents_5 = [
    ["sun", "moon", "stars"],
    ["galaxy", "moon", "comet"],
    ["blackhole", "comet", "nebula"],
]

test_samples = [
    (t_1, documents_1),
    (t_2, documents_2),
    (t_3, documents_3),
    (t_4, documents_4),
    (t_5, documents_5),
]

def test_func(test_cases):
    def test_helper(t, documents):
        new_documents, vocab = handle_unknown_words(t, documents)
        for doc in new_documents:
            print(doc)
        print()
        print(vocab)

    for t, doc in test_cases:
        test_helper(t, doc)
        print()
        print()


# test_func(test_samples)

def apply_smoothing(k, observation_counts, unique_obs):
    """
    Apply add-k smoothing to state-observation counts and return the log smoothed observation
    probabilities log[P(observation | state)].

    Input:
        k (float):
            A float number to add to each count (the k in add-k smoothing)
            Observation here can be either an NER tag or a word,
            depending on if you are applying_smoothing to transition_matrix or emission_matrix
        observation_counts (Dict[Tuple[str, str], float]):
            A dictionary containing observation counts for each state.
            Keys are state-observation pairs and values are numbers of occurrences of the key.
            Keys should contain  all possible combinations of (state, observation) pairs.
            i.e. if a `(NER tag, word)` doesn't appear in the training data, you should still include it as `observation_counts[(NER tag, word)]=0`
        unique_obs (List[str]):
            A list of string containing all the unique observation in the dataset.
            If you are applying smoothing to the transition matrix, unique_obs contains all the possible NER tags in the dataset.
            If you are applying smoothing to the emission matrix, unique_obs contains the vocabulary in the dataset

    Output:
        Dict<key Tuple[String, String]: value Float>
            A dictionary containing log smoothed observation **probabilities** for each state.
            Keys are state-observation pairs and values are the log smoothed
            probability of occurrences of the key.
            The output should be the same size as observation_counts.

    Note that the function will be applied to both transition_matrix and emission_matrix.
    """
    # YOUR CODE HERE 
    #first calculate the denominator for the add k smoothing
    # which is the sum of the state observation pair
    # the prob of the state observation over sum of the probs and using 
    #the log smoothed format
    log_prob = {}
    states = set( state for (state, _) in observation_counts.keys() )

    for state in states:
        denom = sum(k + observation_counts[(state, obs)] for obs in unique_obs)

        #diff of logs = prob / prob
        for obs in unique_obs:
            num = observation_counts[(state, obs)] + k
            
            log_prob[(state, obs)] = math.log(num) - math.log(denom)

    return log_prob
