"""
"""
from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, concatenate
from keras.layers import GRU, LSTM, Flatten
from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from sklearn.model_selection import train_test_split
import warnings
import os
import numpy as np
import pandas as pd
import gc

# from . import embeddings
from aisimplekit.dnn import embeddings

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'


def root_mean_squared_error(y_true, y_pred):
    """ Compute rmse loss. """
    return K.sqrt(K.mean(K.square(y_true-y_pred)))

class RnnModelType(object):
    """ Supported RNN model types (gru or lstm). """
    GRU = "gru"
    LSTM = "lstm"
    __SUPPORTED__ = [GRU, LSTM]

class RnnTextModel(object):
    """ Class for RNN Text Model. """
    def __init__(self, tokenizer_num_words, cat_cols=[],
                    text_seq_cols=[],
                    num_cols=[], num_transform_spec={},
                    max_seq_length=100,
                    embedding_file='../input/fasttest-common-crawl-russian/cc.ru.300.vec',
                    embedding_dim1=300, emb_out_size=10,
                    _prepare_df_handler=None,
                    batch_size=512*3, model_type=RnnModelType.GRU, n_units=50,
                    dropout_0=0.1, dropout_1=0.1, ndense_0=512, ndense_1=64,
                    final_layer_handler=None,
                    loss_fn=None, metrics_fns=None, learning_rates=(0.009, 0.0045),
                    optimizer="adam",
                    text_spec={}):
        """
        :param text_spec: Specification of text columns, num_word per col, embeddings..
        :type text_spec: dict
        """
        assert(batch_size > 0)
        assert(emb_out_size > 0)
        assert(model_type in RnnModelType.__SUPPORTED__)
        assert(n_units > 0)

        # Loss and metrics functions
        if loss_fn is None and (metrics_fns is None or len(metrics_fns)==0):
            print('No loss, nor metrics specified: using rmse by default!')
            loss_fn = root_mean_squared_error
            metrics_fns = [root_mean_squared_error]
        assert(loss_fn is not None)
        assert(metrics_fns is not None and len(metrics_fns) > 0)
        self.loss_fn = loss_fn
        self.metrics_fns = metrics_fns
        self.learning_rates = learning_rates
        self.optimizer = optimizer

        # Inputs and Preprocessing
        self.max_seq_length = max_seq_length
        self.cat_cols = cat_cols
        self.text_seq_cols = text_seq_cols
        self.num_cols = num_cols
        self.num_transform_spec = num_transform_spec
        # num_words: the maximum number of words to keep, based
        #   on word frequency. Only the most common `num_words-1` words will be kept.
        tokenizers = {}
        for col in self.text_seq_cols:
            tokenizers[col] = Tokenizer(num_words=tokenizer_num_words)
        self.tokenizers = tokenizers
        # self.tokenizer = Tokenizer(num_words=num_words)
        vocab_size = {}
        for col in self.text_seq_cols:
            vocab_size[col] = -1
        self.vocab_size = vocab_size
        self._prepare_df_handler = _prepare_df_handler

        # Embeddings for categorical
        self.embedding_dim1 = embedding_dim1 # from the pretrained vectors
        self.embedding_file = embedding_file
        self.emb_out_size = emb_out_size

        # Model: GRU or LSTM
        self.batch_size = batch_size
        self.model_type = model_type
        self.n_units = n_units
        self.model = None

        # Final layer
        self.dropout_0 = dropout_0
        self.dropout_1 = dropout_1
        self.ndense_0 = ndense_0
        self.ndense_1 = ndense_1
        self.final_layer_handler = final_layer_handler # possibility to override final layer composition.

    def _prepare_df(self, df):
        """ """
        if self._prepare_df_handler:
            return self._prepare_df_handler(df)
        return df

    def _fit_text(self, df, traindex):
        """ """
        for col in self.text_seq_cols:
            all_text = np.hstack([df.loc[traindex,:][col].str.lower()])
            self.tokenizers[col].fit_on_texts(all_text)
            self.vocab_size[col] = len(self.tokenizers[col].word_index)+2
            del(all_text)
            gc.collect()

    def _encode_categorical(self, df):
        """ """
        for col in self.cat_cols:
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
        return df

    def _build_text_sequences(self, df):
        """ """
        for col in self.text_seq_cols:
            df['seq_{}'.format(col)] = (
                self.tokenizers[col]
                    .texts_to_sequences(
                        df[col].str.lower()
                    )
            )
            del(df[col])
            gc.collect()
        return df

    def _preprocess_numerical(self, df):
        """ """
#        df['price'] = np.log1p(df['price']) # already transformed to log
        # if False:
        #     print('WITH USER AGG !')
        #     df['avg_days_up_user'] = np.log1p(df['avg_days_up_user'])
        #     df['avg_times_up_user'] = np.log1p(df['avg_times_up_user'])
        #     df['n_user_items'] = np.log1p(df['n_user_items'])
        for col in self.num_cols:
            if col in self.num_transform_spec.keys():
                transf_fn = self.num_transform_spec[col]
                df[col] = transf_fn(df[col])
        return df

    def prepare_df(self, df, traindex):
        """ """
        df = self._prepare_df(df)
        self._fit_text(df, traindex)
        df = self._encode_categorical(df)
        df = self._build_text_sequences(df)
        df = self._preprocess_numerical(df)
        return df

    def get_keras_data(self, dataset, max_seq_length):
        """ """
        data = {}
        for col in self.text_seq_cols:
            data['seq_{}'.format(col)] = pad_sequences(dataset['seq_{}'.format(col)],
                                        maxlen=max_seq_length)
                                        # FIXME: max_seq_title.. is common to all text_seq_cols => should not be !

        # checking that text_seq_cols dont contain any categorical simple cols => exclusive.
        assert(all([col not in self.cat_cols for col in self.text_seq_cols]))

        # categorical + numerical
        cols = self.cat_cols + self.num_cols
        for col in cols:
            data[col] = np.array(dataset[[col]])

        return data

    def build_rnn_model(self, embedding_matrixes):
        """ """
        # Inputs
        # 1) sequential columns
        arr_inputs_seq = []
        for col in self.text_seq_cols:
            in_seq = Input(shape=[self.max_seq_length], # FIXME: shouldnt be common to all text_seq_cols
                                    name="seq_{}".format(col))
            arr_inputs_seq.append(in_seq)

        # 2) categorical+numerical columns
        arr_inputs_cat = []
        for col in self.cat_cols:
            in_cat = Input(shape=[1], name=col)
            arr_inputs_cat.append(in_cat)

        arr_inputs_num = []
        for col in self.num_cols:
            in_num = Input(shape=[1], name=col)
            arr_inputs_num.append(in_num)

        # Embeddings layers
        col = self.text_seq_cols[0] # FIXME: why do we arbirarily choose the first column  ?????
        if self.vocab_size[col] < 0:
            self.vocab_size[col] = len(self.tokenizers[col].word_index)+2
        global_vocab_size = self.vocab_size[col] ## FIXME: vocab size is for the 1st seq column only ???
                                            ## FIXME: Why do we use it in all the embeddings ????

        embs_seqs = []
        for idx, col in enumerate(self.text_seq_cols):
            if self.vocab_size[col] < 0:
                self.vocab_size[col] = len(self.tokenizers[col].word_index)+2
            vocab_size = self.vocab_size[col] ## FIXME: vocab size is for the 1st seq column only ???

            emb_col = Embedding(
                vocab_size, self.embedding_dim1, weights=[embedding_matrixes[idx]],
                trainable=False
            )(arr_inputs_seq[idx])
            embs_seqs.append(emb_col)

        # emb_seq_title_description = Embedding(
        #     vocab_size, self.embedding_dim1, weights=[embedding_matrix1],
        #     trainable=False
        # )(seq_title_description)

        # For each categorical col, transform to vector of scalars using Embedding.
        emb_out_size = self.emb_out_size # embedding output size default

        embs_cat = []
        for idx, col in enumerate(self.cat_cols):
            emb_col = Embedding(global_vocab_size, emb_out_size)(arr_inputs_cat[idx])
            embs_cat.append(emb_col)

        # GRU Model (or LSTM)
        rnn_layers = []
        if self.model_type is RnnModelType.GRU:
            rnn_layers = [
                GRU(self.n_units)(emb)
                for emb in embs_seqs
            ]
        elif self.model_type is RnnModelType.LSTM:
            rnn_layers = [
                LSTM(self.n_units)(emb)
                for emb in embs_seqs
            ]
        else:
            raise Exception('[error] Unsupported Model Type:{}'.format(self.model_type))

        #main layer
        layers = [
            *rnn_layers,
            *[Flatten()(emb) for emb in embs_cat],
            *arr_inputs_num,
        ]
        main_l = concatenate(layers)

        if self.final_layer_handler is not None:
            # Possibility to override defaut double dense layers with dropout
            main_l = self.final_layer_handler(main_l)
        else:
            main_l = Dropout(self.dropout_0)(Dense(self.ndense_0, activation='relu') (main_l))
            main_l = Dropout(self.dropout_1)(Dense(self.ndense_1, activation='relu') (main_l))

        #output
        output = Dense(1, activation="sigmoid") (main_l)

        #model
        inputs = arr_inputs_seq + arr_inputs_cat + arr_inputs_num # order matters

        model = Model(inputs, output)
        model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics_fns)
        self.model = model

    def rmse(self, y, y_pred):
        """ """
        rsum = np.sum((y-y_pred)**2)
        n = y.shape[0]
        rmse = np.sqrt(rsum/n)
        return rmse

    def eval_model(self, X_test1, y_test1):
        """ """
        val_preds = self.model.predict(X_test1)
        y_pred = val_preds[:, 0]
        y_true = np.array(y_test1)
        yt = pd.DataFrame(y_true)
        yp = pd.DataFrame(y_pred)
        print(yt.isnull().any())
        print(yp.isnull().any())
        v_rmse = self.rmse(y_true, y_pred)
        print("rmse for validation set: "+str(v_rmse))
        return v_rmse

    def init_predictor(self, df, traindex):
        """ """
        df = self.prepare_df(df, traindex)

        embedding_matrixes = []
        for col in self.text_seq_cols:
            embedding_matrix1 = embeddings.load_embedding_matrix(
                self.embedding_file,
                self.vocab_size[col],
                self.embedding_dim1, # FIXME: same for all text_seq_cols ????
                self.tokenizers[col]
            )
            embedding_matrixes.append(embedding_matrix1)

        self.build_rnn_model(embedding_matrixes)

        return df

    def fit(self, train, y, n_iter=3, cv=False, test_size=0.10, random_state=23):
        """ """
        if cv is True:
            raise Exception('Not Yet Implemented !')
        X_train, X_valid, y_train, y_valid = train_test_split(
            train, y,
            test_size=test_size,
            random_state=random_state
        )

        # Fit the NN Model
        X_train = self.get_keras_data(X_train, self.max_seq_length)
        X_valid = self.get_keras_data(X_valid, self.max_seq_length)

        exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

        # Initializing a new model for current fold
        epochs = 1
        steps = (int(train.shape[0]/self.batch_size))*epochs
        (lr_init, lr_fin) = self.learning_rates
        lr_decay = exp_decay(lr_init, lr_fin, steps)
        K.set_value(self.model.optimizer.lr, lr_init)
        K.set_value(self.model.optimizer.decay, lr_decay)

        for i in range(n_iter):
            hist = self.model.fit(X_train, y_train,
                    batch_size=self.batch_size+(self.batch_size*(2*i)),
                    epochs=epochs, validation_data=(X_valid, y_valid),
                    verbose=1)

        v_rmse = self.eval_model(X_valid, y_valid)
        del(X_train)
        del(X_valid)
        del(y_train)
        del(y_valid)
        gc.collect()

        return v_rmse

    def predict(self, df_test, verbose=1):
        """ """
        X_test = self.get_keras_data(
            df_test,
            max_seq_length=self.max_seq_length
        )
        preds1 = self.model.predict(X_test, batch_size=self.batch_size, verbose=verbose)
        del(X_test)
        gc.collect()

        print("RNN Prediction is done.")
        preds = preds1.reshape(-1,1)
        preds = np.clip(preds, 0, 1)
        print(preds.shape)
        return preds