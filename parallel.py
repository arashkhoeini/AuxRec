import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Embedding, Flatten

class Parallel(object):

    def __init__(self, dataset, ns_size, layers, epoch_number, batch_size, validation_split):
        self.dataset = dataset
        self.ns_size = ns_size
        self.layers = layers
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.validation_split = validation_split

    def _generate_train_data(self):
        users = []
        items = []
        targets = []
        for u in range(self.dataset.user_size):
            user_pos_samples = []
            for i in range(self.dataset.item_size):
                if self.dataset.user_item[u,i] > 0:
                    user_pos_samples.append(i)
                    users.append(u)
                    items.append(i)
                    targets.append(1)
            p = self.dataset.user_item[u,:].copy()
            p = 1 - p
            p = p/sum(p)
            neg_samples = np.random.choice(self.dataset.item_size, 
                                            min( self.ns_size*len(user_pos_samples), sum(p>0)) , replace=False, p=p )
            for ns in neg_samples:

                users.append(u)
                items.append(ns)
                targets.append(0)

        items, users, targets = np.array(items), np.array(users), np.array(targets)

        #shuffling arrays
        randomize = np.arange(len(targets))
        np.random.shuffle(randomize)

        items, users, targets= items[randomize], users[randomize], targets[randomize]

        return users, items, targets

    def train_model(self):

        users, items, targets = self._generate_train_data()
        user_inputs = Input(shape=(1,))
        item_inputs = Input(shape=(1,))

        item_embbedings  =  Embedding(self.dataset.item_size, self.layers[0], input_length=1)(item_inputs)
        user_embbedings  =  Embedding(self.dataset.user_size, self.layers[0] , input_length=1)(user_inputs)

        user_emb_flat = Flatten()(user_embbedings)
        item_emb_flat = Flatten()(item_embbedings)

        
        for idx, size in enumerate(self.layers):
            if idx == 0:
                pass
            elif idx ==1:
                u_hidden = Dense(size, activation='relu')(user_emb_flat)
            else:
                u_hidden = Dense(size, activation='relu')(u_hidden)

        for idx, size in enumerate(self.layers):
            if idx == 0:
                pass
            elif idx ==1:
                i_hidden = Dense(size, activation='relu')(item_emb_flat)
            else:
                i_hidden = Dense(size, activation='relu')(i_hidden)

        concated = concatenate([i_hidden, u_hidden])

        pref_prediction = Dense(1, activation='sigmoid', name='pref')(concated)

        model = Model(inputs=[user_inputs, item_inputs], outputs=[pref_prediction])
        model.summary()

        model.compile(loss=['binary_crossentropy'],
                    optimizer='adam' 
                    , metrics=['accuracy']
                    )

        self.history = model.fit([users, items], [targets],
                                                epochs=self.epoch_number, batch_size = self.batch_size,
                                                validation_split=self.validation_split,
                                                verbose=1)

        return model
    
    

    