import numpy as np

import scipy.sparse as sp
import numpy as np
from time import time


class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, data_path, num_negatives, batch_size, fast_running=False):
        '''
        Constructor
        '''
        self.batch_size = int(batch_size)
        self.num_negatives = int(num_negatives)
        self.trainMatrix = self.load_training_file_as_matrix(data_path + ".train.rating")
        self.trainList = self.load_training_file_as_list(data_path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(data_path + ".test.rating")
        self.testNegatives = self.load_negative_file(data_path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape
        if fast_running:
            self.trainList = self.trainList[10000:20000]
            self.testRatings = self.testRatings[:2000]
            self.testNegatives = self.testNegatives[:2000]


    ########################  load data from the file #########################
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.rstrip().split("\t")
                negatives = []
                for x in arr[0:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        print("already load the trainMatrix...")
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
                index += 1
                if index < 210:
                    items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists

    ##################### generate positive instances
    def get_positive_instances(self):
        p_user_input_list, p_num_idx_list, p_item_input_list, p_labels_list = [], [], [], []
        p_batch_num = int(len(self.trainList) / self.batch_size)
        for batch in range(p_batch_num):
            u, n, i, l = self._get_positive_batch(batch)
            p_user_input_list.append(u)
            p_num_idx_list.append(n)
            p_item_input_list.append(i)
            p_labels_list.append(l)
        p_user_input_list = np.array(p_user_input_list)
        p_num_idx_list = np.array(p_num_idx_list)
        p_item_input_list = np.array(p_item_input_list)
        p_labels_list = np.array(p_labels_list)

        return [p_user_input_list, p_num_idx_list, p_item_input_list, p_labels_list, p_batch_num]

    def _get_positive_batch(self,i):
        user, number, item, label = [], [], [], []
        padding_number = self.trainMatrix.shape[1]
        begin = i * self.batch_size
        for idx in range(begin, begin + self.batch_size):
            sample = self.trainList[idx]
            i_i = sample[-1]
            sample.pop()
            u_i = sample
            user.append(u_i)
            number.append(len(u_i))
            item.append(i_i)
            label.append(1)
        user_input = self._add_mask(padding_number, user, max(number))
        return user_input, number, item, label

    

    ################# generate positive/negative instances for training

    def get_dataset_with_neg(self):  # negative sampling and shuffle the data

        self._get_train_data_fixed()
        iterations = len(self.user_input_with_neg)
        self.index_with_neg = np.arange(iterations)
        self.num_batch_with_neg = iterations / self.batch_size
        return self._preprocess(self._get_train_batch_fixed)

    # def batch_gen(batches, i):
    #     return [(batches[r])[i] for r in range(4)]


    def _preprocess(self,get_train_batch):  # generate the masked batch list
        user_input_list, num_idx_list, item_input_list, labels_list = [], [], [], []

        for i in range(int(self.num_batch_with_neg)):
            ui, ni, ii, l = get_train_batch(i)
            user_input_list.append(ui)
            num_idx_list.append(ni)
            item_input_list.append(ii)
            labels_list.append(l)

        return [user_input_list, num_idx_list, item_input_list, labels_list,self.num_batch_with_neg]


    def _get_train_data_fixed(self):
        self.user_input_with_neg, self.item_input_with_neg, self.labels_with_neg = [], [], []
        for u in range(len(self.trainList)):
            i = self.trainList[u][-1]
            self.user_input_with_neg.append(u)
            self.item_input_with_neg.append(i)
            self.labels_with_neg.append(1)
            # negative instances
            for t in range(self.num_negatives):
                j = np.random.randint(self.num_items)
                while j in self.trainList[u]:
                    j = np.random.randint(self.num_items)
                self.user_input_with_neg.append(u)
                self.item_input_with_neg.append(j)
                self.labels_with_neg.append(0)


    def _get_train_batch_fixed(self, i):
        # represent the feature of users via items rated by him/her
        user_list, num_list, item_list, labels_list = [], [], [], []
        trainList = self.trainList
        begin = i * self.batch_size
        for idx in range(begin, begin + self.batch_size):
            user_idx = self.user_input_with_neg[self.index_with_neg[idx]]
            item_idx = self.item_input_with_neg[self.index_with_neg[idx]]
            nonzero_row = []
            nonzero_row += self.trainList[user_idx]
            num_list.append(self._remove_item(self.num_items, nonzero_row, nonzero_row[-1]))
            user_list.append(nonzero_row)
            item_list.append(item_idx)
            labels_list.append(self.labels_with_neg[self.index_with_neg[idx]])
        user_input = self._add_mask(self.num_items, user_list, max(num_list))
        num_idx = num_list
        item_input = item_list
        labels = labels_list
        return (user_input, num_idx, item_input, labels)


    def _remove_item(self,feature_mask, users, item):
        flag = 0
        for i in range(len(users)):
            if users[i] == item:
                users[i] = users[-1]
                users[-1] = feature_mask
                flag = 1
                break
        return len(users) - flag


    def _add_mask(self, feature_mask, features, num_max):
        # uniformalize the length of each batch
        for i in range(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
        return features


 ################# generate positive/negative instances for test

    def get_test_instances(self):
        test_user_input, test_num_idx, test_item_input, test_labels = [], [], [], []

        for idx in range(len(self.testRatings)):
            test_user = self.testRatings[idx][0]
            rating = self.testRatings[idx][:]
            items = self.testNegatives[idx][:]
            user = self.trainList[test_user][:]

            # user.append(self.num_items)  # add padding number
            items.append(rating[1]) # add positive instance at the end of the negative instances 
            num_idx = np.full(len(items), len(user), dtype=np.int32) # the length of historical items are all the same, equaling to the length of the historical items of the positive instance
            user_input = np.tile(user, (len(items), 1)) # historical items are the same for the positive and negative instances
            item_input = np.array(items)
            labels = np.zeros(len(items)) 
            labels[-1] = 1 # the last label for the positive instance is 1

            test_user_input.append(user_input)
            test_num_idx.append(num_idx)
            test_item_input.append(item_input)
            test_labels.append(labels)

        return [test_user_input, test_num_idx, test_item_input, test_labels, len(self.testRatings)]





