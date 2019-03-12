import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Embedding, Flatten

class Parallel_AUX(object):

    def __init__(self, dataset, ns_size, layers, epoch_number, batch_size, validation_split, user_sampling_size, core_number, sim_threshold):
        self.dataset = dataset
        self.ns_size = ns_size
        self.layers = layers
        self.epoch_number = epoch_number
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.user_sampling_size = user_sampling_size
        self.core_number = core_number
        self.sim_threshold = sim_threshold
        self.user_user_jaccard = self._get_user_user_jaccard(sim_threshold)

    def _calc_jacccard(self, user):

        user_jaccard_vector = []
        for user2 in range(self.dataset.user_size):
            user_user_dot = np.dot(self.dataset.user_item[user,:], self.dataset.user_item[user2,:])
            user_jaccard_vector.append(user_user_dot / \
                                (sum(self.dataset.user_item[user,:]) + sum(self.dataset.user_item[user2,:]) - user_user_dot))
        return np.array(user_jaccard_vector)


    def _get_user_user_jaccard(self, sim_threshold):

        path = 'data/user_user_jaccard_{}_{}.npy'.format(self.dataset.dataset_name, sim_threshold)
        try:
            return np.load(path)
        except FileNotFoundError:
            pool = Pool(processes=self.core_number)
            inputs = range(self.dataset.user_size)
            result = pool.map(self._calc_jacccard, inputs)
            user_user_jaccard = np.array(result, dtype=float)
            for user1 in range(self.dataset.user_size):
                for user2 in range(self.dataset.user_size):
                    if user_user_jaccard[user1,user2] >= self.sim_threshold:
                        user_user_jaccard[user1,user2] = 1
                    else:
                        user_user_jaccard[user1,user2] = 0
            np.save(path, user_user_jaccard) 
            return user_user_jaccard


    def _generate_train_data(self):
        data_temp = []

        users = []
        cusers = []
        items = []
        sims = []
        targets = []
        pos_samples = []
        for u in range(self.dataset.user_size):
            user_pos_samples = []
            for i in range(self.dataset.item_size):
                if self.dataset.user_item[u,i] > 0:
                    user_pos_samples.append(i)
                    data_temp.append((u,i,1))
            p = self.dataset.user_item[u,:].copy()
            p = 1 - p
            p = p/sum(p)
            neg_samples = np.random.choice(self.dataset.item_size, 
                                            min( self.ns_size*len(user_pos_samples), sum(p>0)) , replace=False, p=p )
            for ns in neg_samples:
                data_temp.append((u,ns,0))

        pos_samples = [ len(p) for p in pos_samples ]

        for user, item, target in data_temp:

            similar_users = np.where(self.user_user_jaccard[user,:] == 1)[0]
            different_users = np.where(self.user_user_jaccard[user,:] == 0)[0]                                 
            sampled_user = np.concatenate((np.random.choice(similar_users , min(self.user_sampling_size, 
                                                            len(similar_users)) , replace=False),
                                            np.random.choice(different_users, min(self.user_sampling_size, 
                                                            len(different_users)), replace=False)))
           
            for cuser in sampled_user:
                if cuser != user:
                    users.append(user)
                    cusers.append(cuser)
                    items.append(item)
                    targets.append(target)
                    sims.append(self.user_user_jaccard[user,cuser])

        items, users, cusers, targets, sims= np.array(items), np.array(users),np.array(cusers), np.array(targets), np.array(sims)

        #shuffling arrays
        randomize = np.arange(len(targets))
        np.random.shuffle(randomize)

        items, users, cusers, targets, sims= items[randomize], users[randomize], cusers[randomize], targets[randomize],  sims[randomize]

        return users, cusers, items, targets, sims

    def train_model(self):

        users, cusers, items, targets, sims = self._generate_train_data()
        cuser_inputs = Input(shape=(1,))
        user_inputs = Input(shape=(1,))
        item_inputs = Input(shape=(1,))

        item_embbedings  =  Embedding(self.dataset.item_size, self.layers[0] , input_length=1)(item_inputs)
        user_embbedings  =  Embedding(self.dataset.user_size, self.layers[0] , input_length=1)(user_inputs)
        cuser_embbedings  =  Embedding(self.dataset.user_size, self.layers[0] , input_length=1)(cuser_inputs)

        cuser_emb_flat = Flatten()(cuser_embbedings)
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
        
        for idx, size in enumerate(self.layers):
            if idx == 0:
                pass
            elif idx ==1:
                uc_hidden = Dense(size, activation='relu')(cuser_emb_flat)
            else:
                uc_hidden = Dense(size, activation='relu')(uc_hidden)


        user_item_concated = concatenate([i_hidden, u_hidden])
        user_user_concated = concatenate([u_hidden, uc_hidden])

        pref_prediction = Dense(1, activation='sigmoid', name='pref')(user_item_concated)
        sim_prediction  = Dense(1, activation='sigmoid',name='sim')(user_user_concated)

        model = Model(inputs=[user_inputs, cuser_inputs, item_inputs], outputs=[pref_prediction, sim_prediction])
        model.summary()

        model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                    optimizer='adam' 
                    , metrics=['accuracy']
                    )

        self.history = model.fit([users, cusers, items], [targets, sims],
                                                epochs=self.epoch_number, batch_size = self.batch_size,
                                                validation_split=self.validation_split,
                                                verbose=True)

        return model
    