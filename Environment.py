import numpy as np

class Environment():
    def __init__(self):
        self.gamma = 0.5

    def initilize_state(self, recommender, traindata, testdata, high_state_size, low_state_size, padding_number):
        self.high_state_size = high_state_size
        self.low_state_size = low_state_size
        self.padding_number = padding_number
        self.course_embedding_user, self.course_embedding_item = recommender.get_course_embedding()
        self.origin_train_rewards = recommender.get_rewards(traindata)
        self.origin_test_rewards = recommender.get_rewards(testdata)
        self.embedding_size = len(self.course_embedding_user[0])  # 16
        self.set_train_original_rewards()

    def set_train_original_rewards(self):
        self.origin_rewards = self.origin_train_rewards

    def set_test_original_rewards(self):
        self.origin_rewards = self.origin_test_rewards

    
    def reset_state(self, user_input, num_idx, item_input, labels, batch_size, max_course_num, batch_index):
        
        self.user_input = user_input
        self.num_idx = num_idx
        self.item_input = np.reshape(item_input, (-1,))
        self.labels = labels
        self.batch_size = batch_size
        self.max_course_num = max_course_num
        self.batch_index = batch_index

        self.origin_prob = np.zeros((self.batch_size, 1), dtype=np.float32)

        self.dot_product_sum = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.dot_product_mean = np.zeros((self.batch_size, 1), dtype=np.float32)

        self.element_wise_mean = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_sum = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)

        self.vector_sum = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_mean = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix = np.zeros((self.batch_size, self.max_course_num), dtype=np.int)
        self.state_matrix = np.zeros((self.batch_size, self.max_course_num, self.low_state_size), dtype=np.float32)
        self.selected_input = np.full((self.batch_size, self.max_course_num), self.padding_number)


    def get_overall_state(self):

        def _mask(i):
            return [True]*i[0] + [False]*(self.max_course_num - i[0])

        origin_prob = np.reshape(self.origin_rewards[self.batch_index], (-1, 1)) #(batch_size, 1)
        self.num_idx = np.reshape(self.num_idx, (-1,1))

        dot_product = self.rank_dot_product_bymatrix(self.user_input, self.item_input)
        element_wise = self.rank_element_wise_bymatrix(self.user_input, self.item_input)
        mask_mat = np.array(list(map(_mask, np.reshape(self.num_idx, (self.batch_size, 1)))))
        dot_product = np.reshape(np.sum(dot_product * mask_mat, 1), (-1,1)) / self.num_idx
        mask_mat = np.repeat(np.reshape(mask_mat, (self.batch_size, self.max_course_num, 1)), self.embedding_size, 2)
        element_wise = np.sum(element_wise * mask_mat, 1) / self.num_idx

        return np.concatenate((dot_product, element_wise, origin_prob),1)

   
    def get_state(self, step_index):
        self.origin_prob = np.reshape(self.origin_rewards[self.batch_index], (-1, 1))  # (batch_size, 1)
        self.dot_product = self.rank_dot_product(self.user_input, self.item_input, step_index)
        self.element_wise_current = self.rank_element_wise(self.user_input, self.item_input, step_index)
        self.vector_current = self.course_embedding_user[self.user_input[:, step_index]]
        self.vector_item = self.course_embedding_item[self.item_input]
        self.vector_current = np.abs(self.vector_current - self.vector_item)
        return np.concatenate((self.vector_mean, self.vector_current, self.dot_product, self.dot_product_mean), 1)

    def rank_element_wise(self, batched_user_input, item_input, step_index):
        self.train_item_ebd = self.course_embedding_user[batched_user_input[:, step_index]]
        self.test_item_ebd = np.reshape(self.course_embedding_item[item_input], (self.batch_size, self.embedding_size))
        return np.multiply(self.train_item_ebd, self.test_item_ebd)  # (batch_size, embedding_size)

    def rank_dot_product(self, batched_user_input, item_input, step_index):
        self.train_item_ebd = self.course_embedding_user[batched_user_input[:, step_index]]
        self.test_item_ebd = np.reshape(self.course_embedding_item[item_input], (self.batch_size, self.embedding_size))
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd,self.train_item_ebd),1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd,self.test_item_ebd),1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = np.where(norm != 0, dot_prod/norm, dot_prod)
        return np.reshape(cos_similarity, (-1, 1))  # (batch_size, 1)

    def rank_element_wise_bymatrix(self, batched_user_input, item_input):
        self.train_item_ebd = self.course_embedding_user[np.reshape(batched_user_input, (-1,1))]  # (batch_size, embedding_size)
        self.test_item_ebd =  self.course_embedding_item[np.reshape(np.tile(item_input, (1,self.max_course_num)), (-1,1))]  # (batch_size, embedding_size)
        return np.reshape(np.multiply(self.train_item_ebd, self.test_item_ebd), (-1,self.max_course_num, self.embedding_size))  # (batch_size, embedding_size)


    def rank_dot_product_bymatrix(self, batched_user_input, item_input):
        self.train_item_ebd = self.course_embedding_user[np.reshape(batched_user_input, (-1,))]  # (batch_size, embedding_size)
        self.test_item_ebd =  self.course_embedding_item[np.reshape(np.tile(item_input, (1,self.max_course_num)), (-1,))]  # (batch_size, embedding_size)
        # print self.train_item_ebd.shape, self.test_item_ebd.shape
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd,self.train_item_ebd),1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd,self.test_item_ebd),1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd, self.test_item_ebd), 1)
        cos_similarity = np.where(norm != 0, dot_prod/norm, dot_prod)
        return np.reshape( cos_similarity , (-1, self.max_course_num))  # (batch_size, 1)

    def update_state(self, low_action, low_state, step_index):
        self.action_matrix[:, step_index] = low_action
        self.state_matrix[:, step_index] = low_state

        self.num_selected = self.num_selected + low_action
        self.vector_sum = self.vector_sum + np.multiply(np.reshape(low_action, (-1, 1)), self.vector_current)
        self.element_wise_sum = self.element_wise_sum + np.multiply(np.reshape(low_action, (-1, 1)), self.element_wise_current)
        self.dot_product_sum = self.dot_product_sum + np.multiply(np.reshape(low_action, (-1,1)), self.dot_product)
        num_selected_array = np.reshape(self.num_selected, (-1, 1))
        self.element_wise_mean = np.where(num_selected_array != 0, self.element_wise_sum / num_selected_array, self.element_wise_sum)
        self.vector_mean = np.where(num_selected_array != 0, self.vector_sum / num_selected_array, self.vector_sum)
        self.dot_product_mean = np.where(num_selected_array != 0, self.dot_product_sum / num_selected_array, self.dot_product_sum)


    def get_action_matrix(self):
        return self.action_matrix

    def get_state_matrix(self):
        return self.state_matrix

    def get_selected_courses(self, high_action):
        notrevised_index = []
        revised_index = []
        delete_index = []
        keep_index = []
        select_user_input = np.zeros((self.batch_size, self.max_course_num), dtype=np.int)
        for index in range(self.batch_size):

            selected = []
            for course_index in range(self.max_course_num):
                if self.action_matrix[index, course_index] == 1:
                    selected.append(self.user_input[index, course_index])

            # revise
            if high_action[index] == 1:
                # delete
                if len(selected) == 0:
                    delete_index.append(index)
                # keep
                if len(selected) == self.num_idx[index]:
                    keep_index.append(index)
                revised_index.append(index)
            # not revise
            if high_action[index] == 0:
                notrevised_index.append(index)

            # random select one course from the original enrolled courses if no course is selected by the agent, change the number of selected courses as 1 at the same time
            if len(selected) == 0:
                original_course_set = list(set(self.user_input[index]))
                if self.padding_number in original_course_set:
                    original_course_set.remove(self.padding_number)
                random_course = np.random.choice(original_course_set, 1)[0]
                selected.append(random_course)
                self.num_selected[index] = 1

            for course_index in range(self.max_course_num - len(selected)):
                selected.append(self.padding_number)
            select_user_input[index, :] = np.array(selected)
        
        
        nochanged = notrevised_index + keep_index
        select_user_input[nochanged] = self.user_input[nochanged]
        self.num_selected[nochanged] = np.reshape(self.num_idx[nochanged],(-1,))
        return select_user_input, self.num_selected, notrevised_index, revised_index, delete_index, keep_index


    def get_reward(self, recommender, batch_index, high_actions, selected_user_input, batched_num_idx, batched_item_input, batched_label_input):
        batch_size = selected_user_input.shape[0]

        # difference between likelihood
        loglikelihood = recommender.get_reward(selected_user_input, np.reshape(self.num_selected, (-1, 1)), batched_item_input, batched_label_input)
        old_likelihood = self.origin_rewards[batch_index]
        likelihood_diff = loglikelihood - old_likelihood
        likelihood_diff = np.where(high_actions == 1, likelihood_diff, np.zeros(batch_size))


        #difference between average dot_product
        dot_product = self.rank_dot_product_bymatrix(selected_user_input, batched_item_input)
        # print dot_product
        new_dot_product = np.sum(np.multiply(dot_product, self.action_matrix),1) / self.num_selected
        old_dot_product = np.sum(dot_product, 1) /batched_num_idx

        dot_product_diff = new_dot_product - old_dot_product
        reward1 = likelihood_diff + self.gamma * dot_product_diff

        return reward1,old_dot_product,likelihood_diff
