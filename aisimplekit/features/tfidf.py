"""
TF-IDF and Count vectorizers.
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
import numpy as np

TRANSFORMER_TFIDF = 'tfidf'
TRANSFORMER_COUNT = 'count'

TRANSFORMERS_CLS = {
    TRANSFORMER_TFIDF: TfidfVectorizer,
    TRANSFORMER_COUNT: CountVectorizer
}

def compute_features(df, transformer_spec, analyzer="word", stop="english"):
    """ Given a transformation spec and data, apply
    either TFIDF and/or counts on words/chars.

    :param df: input dataframe.
    :type: pd.DataFrame
    :param transformer_spec: specifications of transforms to apply.
    :type: dict(str: dict)
    :param analyzer: either 'word' or 'char'
    :type: str
    :param : stop word vocab, default is 'english'.
    :type: str

    :returns: the fitted vectorizer, predicted data after transforms, and vocabulary
    :rtype: tuple(
                sklearn.pipeline.FeatureUnion,
                scipy.sparse.csr.csr_matrix,
                list
            )
    """
    global TRANSFORMERS_CLS

    stop_words = set(stopwords.words(stop))

    # default params
    _tfidf_params = {
        "stop_words": stop_words,
        "analyzer": analyzer,
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        "smooth_idf": False
    }

    def _get_col(col_name):
        """ """
        return lambda x: x[col_name]

    ## Union of tfidf and count features from differents cols.
    transformer_list = []
    for col in transformer_spec.keys():
        spec = transformer_spec[col]
        _cls = TRANSFORMERS_CLS[spec['vectorizer']]

        tfidf_params = spec['kwargs']
        if tfidf_params is {}: # override, and take defaults
            tfidf_params = _tfidf_params
        elif tfidf_params is None: # no additional params
            tfidf_params = {}

        vectorizer = _cls(
            ngram_range=spec['ngram_range'],
            max_features=spec['max_features'],
            preprocessor=_get_col(col),
            **tfidf_params
        )
        transformer_list.append((col, vectorizer))

    vectorizer = FeatureUnion(transformer_list)

    # Vectorization: fit records
    records = df.to_dict('records')
    vectorizer.fit(records)

    # Vectorization: transform
    df_tfidf = vectorizer.transform(records)

    if analyzer == 'word':
        tf_vocab = vectorizer.get_feature_names()
    else:
        tf_vocab = ['cfeat_%d'%k for k in range(len(vectorizer.get_feature_names()))]

    return (vectorizer, df_tfidf, tf_vocab)