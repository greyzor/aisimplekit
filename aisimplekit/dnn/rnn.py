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
    def __init__(self, num_words, cat_cols=[], max_seq_title_description_length=100,
                    embedding_file='../input/fasttest-common-crawl-russian/cc.ru.300.vec',
                    embedding_dim1=300, emb_out_size=10,
                    _prepare_df_handler=None,
                    batch_size=512*3, model_type=RnnModelType.GRU, n_units=50,
                    dropout_0=0.1, dropout_1=0.1, ndense_0=512, ndense_1=64,
                    final_layer_handler=None,
                    loss_fn=None, metrics_fns=None, learning_rates=(0.009, 0.0045),
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

        # Inputs and Preprocessing
        self.max_seq_title_description_length = max_seq_title_description_length
        self.cat_cols = cat_cols
        # num_words: the maximum number of words to keep, based
        #   on word frequency. Only the most common `num_words-1` words will be kept.
        self.tokenizer = Tokenizer(num_words=num_words)
        self.vocab_size = -1
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
        all_text = np.hstack([df.loc[traindex,:]['title_description'].str.lower()])
        self.tokenizer.fit_on_texts(all_text)
        self.vocab_size = len(self.tokenizer.word_index)+2
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
        df['seq_title_description'] = self.tokenizer.texts_to_sequences(df['title_description'].str.lower())
        del(df['title_description'])
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
        df['item_seq_number'] = np.log(df['item_seq_number'])
        return df

    def prepare_df(self, df, traindex):
        """ """
        df = self._prepare_df(df)
        self._fit_text(df, traindex)
        df = self._encode_categorical(df)
        df = self._build_text_sequences(df)
        df = self._preprocess_numerical(df)
        return df

    def get_keras_data(self, dataset, max_seq_title_description_length):
        """ """
        data = {
            'seq_title_description': pad_sequences(dataset.seq_title_description,
                                        maxlen=max_seq_title_description_length),
            'region': np.array(dataset.region),
            'city': np.array(dataset.city),
            'category_name': np.array(dataset.category_name),
            'parent_category_name': np.array(dataset.parent_category_name),
            'param_1': np.array(dataset.param_1),
            'param123': np.array(dataset.param123),
            'image_top_1':np.array(dataset.image_top_1),
            'price': np.array(dataset[["price"]]),
            'item_seq_number': np.array(dataset[["item_seq_number"]]),
            # 'avg_ad_days': np.array(dataset[["avg_days_up_user"]]),
            # 'avg_ad_times': np.array(dataset[["avg_times_up_user"]]),
            # 'n_user_items': np.array(dataset[["n_user_items"]])
        }
        return data

    def build_rnn_model(self, embedding_matrix1):
        """ """
        #Inputs
        seq_title_description = Input(shape=[self.max_seq_title_description_length],
                                        name="seq_title_description")
        region = Input(shape=[1], name="region")
        city = Input(shape=[1], name="city")
        category_name = Input(shape=[1], name="category_name")
        parent_category_name = Input(shape=[1], name="parent_category_name")
        param_1 = Input(shape=[1], name="param_1")
        param123 = Input(shape=[1], name="param123")
        image_code = Input(shape=[1], name="image_top_1")
        price = Input(shape=[1], name="price")
        item_seq_number = Input(shape=[1], name='item_seq_number')
        # avg_ad_days = Input(shape=[1], name="avg_ad_days")
        # n_user_items = Input(shape=[1], name="n_user_items")
        # avg_ad_times = Input(shape=[1], name="avg_ad_times")

        #Embeddings layers
        if self.vocab_size < 0:
            self.vocab_size = len(self.tokenizer.word_index)+2
        vocab_size = self.vocab_size

        emb_seq_title_description = Embedding(
            vocab_size, self.embedding_dim1, weights=[embedding_matrix1],
            trainable=False
        )(seq_title_description)

        # For each categorical col, transform to vector of scalars using Embedding.
        emb_out_size = self.emb_out_size # embedding output size default
        emb_region = Embedding(vocab_size, emb_out_size)(region)
        emb_city = Embedding(vocab_size, emb_out_size)(city)
        emb_category_name = Embedding(vocab_size, emb_out_size)(category_name)
        emb_parent_category_name = Embedding(vocab_size, emb_out_size)(parent_category_name)
        emb_param_1 = Embedding(vocab_size, emb_out_size)(param_1)
        emb_param123 = Embedding(vocab_size, emb_out_size)(param123)
        emb_image_code = Embedding(vocab_size, emb_out_size)(image_code)

        # GRU Model (or LSTM)
        if self.model_type is RnnModelType.GRU:
            rnn_layer1 = GRU(self.n_units)(emb_seq_title_description)
        elif self.model_type is RnnModelType.LSTM:
            rnn_layer1 = LSTM(self.n_units)(emb_seq_title_description)
        else:
            raise Exception('[error] Unsupported Model Type:{}'.format(self.model_type))
        #main layer
        layers = [
            rnn_layer1,
            Flatten()(emb_region),
            Flatten()(emb_city),
            Flatten()(emb_category_name),
            Flatten()(emb_parent_category_name),
            Flatten()(emb_param_1),
            Flatten()(emb_param123),
            Flatten()(emb_image_code),
            price,
            item_seq_number,
            # avg_ad_days,
            # avg_ad_times,
            # n_user_items,
            # avg_ad_times
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
        inputs = [
            seq_title_description, region, city, category_name,
            parent_category_name, param_1, param123, image_code, price, item_seq_number,
            # avg_ad_days, avg_ad_times, n_user_items, avg_ad_times
        ]

        model = Model(inputs, output)
        model.compile(optimizer='adam', loss=self.loss_fn, metrics=self.metrics_fns)
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

        embedding_matrix1 = embeddings.load_embedding_matrix(
            self.embedding_file,
            self.vocab_size,
            self.embedding_dim1,
            self.tokenizer
        )
        self.build_rnn_model(embedding_matrix1)

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
        X_train = self.get_keras_data(X_train, self.max_seq_title_description_length)
        X_valid = self.get_keras_data(X_valid, self.max_seq_title_description_length)

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
            max_seq_title_description_length=self.max_seq_title_description_length
        )
        preds1 = self.model.predict(X_test, batch_size=self.batch_size, verbose=verbose)
        del(X_test)
        gc.collect()

        print("RNN Prediction is done.")
        preds = preds1.reshape(-1,1)
        preds = np.clip(preds, 0, 1)
        print(preds.shape)
        return preds