# coding: utf-8
"""
Notes from: https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526

An embedding is a mapping of a discrete -categorical- variable to a vector of continuous numbers.
In the context of neural networks, embeddings are low-dimensional,
learned continuous vector representations of discrete variables.
Neural network embeddings are useful because they can reduce the dimensionality of
categorical variables and meaningfully represent categories in the transformed space.
"""
import numpy as np
import gc

def load_embedding_matrix(fpath, vocab_size, embedding_dim1, tokenizer):
    """ """
    def _get_coefs(word, *arr):
        """ """
        return word, np.asarray(arr, dtype='float32')

    embeddings_index1 = dict(
        _get_coefs(*o.rstrip().rsplit(' '))
            for o in open(fpath)
    )

    embedding_matrix1 = np.zeros((vocab_size, embedding_dim1))
    print(embedding_matrix1.shape)

    # Creating Embedding matrix
    for word, i in tokenizer.word_index.items():
        if word in embeddings_index1:
            embedding_vector = embeddings_index1[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix1[i] = embedding_vector

    print(embedding_matrix1.shape)
    del(embeddings_index1)
    gc.collect()
    return embedding_matrix1