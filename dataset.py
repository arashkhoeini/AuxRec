import pandas
import numpy as np


class Dataset(object):

    def __init__(self, dataset_name ):
        self.dataset_name = dataset_name
        self.user_item, self.test_data, self.test_negatives = self._read_data()
        self.user_size = self.user_item.shape[0]
        self.item_size = self.user_item.shape[1]

    def _read_data(self):
        return self._get_user_item_matrix(), self._get_test_data(), self._get_test_negatives()

    def _get_dataset_path(self, type):
        path = "data/{}/{}.{}"
        if type == 'train':
            return path.format(self.dataset_name, self.dataset_name, 'train.rating')
        elif type == 'test':
            return path.format(self.dataset_name, self.dataset_name, 'test.rating')
        elif type == 'negative':
            return path.format(self.dataset_name, self.dataset_name, 'test.negative')
        else:
            raise NameError('No such type for datasets.')

    def _get_user_item_matrix(self):
        train_df = pandas.read_csv(self._get_dataset_path('train'), 
                                    delim_whitespace=True , names= ['user', 'item', 'rate', 'date']
                                       ,dtype={'user':np.int32 , 'item': np.int32} )
        user_size = max(train_df.user) +1
        item_size = max(train_df.item) +1 
        user_item = np.zeros((user_size, item_size))
        for user, item in zip(train_df.user, train_df.item):
            user_item[user, item] = 1
        
        return user_item

    def _get_test_data(self):
        test_df = pandas.read_csv(self._get_dataset_path('test'), delim_whitespace=True , names= ['user', 'item', 'rate', 'date']
                                       ,dtype={'user':np.int32 , 'item': np.int32} )
        test_data = test_df.values[:,0:2]
        return test_data

    def _get_test_negatives(self):
        test_negatives = []
        with open(self._get_dataset_path('negative')) as file:
            for line in file:
                line = line.rstrip()
                line = line.split()
                test_negatives.append(list(map(int,line[1:])))      

        return test_negatives
