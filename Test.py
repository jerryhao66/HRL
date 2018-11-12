import numpy as np
import tensorflow as tf
import random
import Train
import os
from collections import defaultdict

from Environment import Environment
from Agent import AgentNetwork
from Recommender import RecommenderNetwork
from DataGenerator import Dataset
from Evaluation import eval_rating
from Setting import setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
global env


def load_course(datapath):
    course_dict = defaultdict(str)
    with open(datapath+".course.csv") as f:
        for line in f.readlines():
            arr = line.strip().split(',')
            course_name, course_id = arr[0], int(arr[1])
            course_dict[course_id] = course_name
    return course_dict


def output(course_dict, originaldata, selectdata):
    user_input, num_idx, item_input, label_input, batch_num  = (originaldata[0],originaldata[1],originaldata[2],originaldata[3], originaldata[4])
    select_user_input, select_num_idx, attentions = (selectdata[0],selectdata[1],selectdata[4])
    writer = open("output.csv", "w")
    for batch_index in range(batch_num):
        original_inputs = user_input[batch_index]
        select_inputs = select_user_input[batch_index]
        original_num_idxs = num_idx[batch_index]
        s_num_idxs = select_num_idx[batch_index]
        items = item_input[batch_index]
        labels = label_input[batch_index]
        batch_attention = attentions[batch_index]
        original_input = original_inputs[-1]
        select_input = select_inputs[-1]
        original_num_idx = original_num_idxs[-1]
        s_num_idx = s_num_idxs[-1]
        item = items[-1]
        label = labels[-1]
        attention = batch_attention[-1]
        if original_num_idx>s_num_idx and s_num_idx > 1:
            writer.write(str(batch_index))
            writer.write("\t")
            for i in range(original_num_idx):
                writer.write(str(course_dict[int(original_input[i])])+"("+str(attention[i])+")")
                writer.write('||')
            writer.write(",")
            for j in range(s_num_idx):
                writer.write(str(course_dict[int(select_input[j])]))
                writer.write('||')
            writer.write(",")
            after_set = set(select_input)
            for i in range(original_num_idx):
                if original_input[i] not in after_set:
                    writer.write(str(course_dict[int(original_input[i])]))
                    writer.write('||')
            writer.write(",")
            writer.write(str(course_dict[int(item)]))
            writer.write(",")
            writer.write(str(label))

            writer.write('\n')
    writer.close()



def _get_high_action(prob, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where(random_number < prob, np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))
    else:
        return np.where(prob >= 0.5, np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))
    

def _get_low_action(prob, user_input_column, padding_number, Random):
    batch_size = prob.shape[0]
    if Random:
        random_number = np.random.rand(batch_size)
        return np.where((random_number < prob) & (user_input_column != padding_number), np.ones(batch_size,dtype=np.int),
                        np.zeros(batch_size,dtype=np.int))
    else:
        return np.where((prob >= 0.5) & (user_input_column != padding_number), np.ones(batch_size,dtype=np.int), np.zeros(batch_size,dtype=np.int))

def sampling_RL(user_input, num_idx, item_input, labels, batch_index, agent, Random=True):

    batch_size = user_input.shape[0]
    max_course_num = user_input.shape[1]
    env.reset_state(user_input, num_idx, item_input, labels, batch_size, max_course_num, batch_index)
    high_state = env.get_overall_state()
    high_prob = agent.predict_high_target(high_state)
    high_action = _get_high_action(high_prob, Random)

    for i in range(max_course_num):
        low_state = env.get_state(i)
        low_prob = agent.predict_low_target(low_state)
        low_action = _get_low_action(low_prob, user_input[:, i], padding_number, Random)
        env.update_state(low_action, low_state, i)
    select_user_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index = env.get_selected_courses(high_action)

    return high_action, high_state, select_user_input, select_num_idx, item_input, labels, notrevised_index, revised_index, delete_index, keep_index


def evalute(agent, recommender, testset):
    test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num  = (testset[0], testset[1], testset[2], testset[3], testset[4])
    env.set_test_original_rewards()
    select_user_input_list, select_num_idx_list, select_item_input_list, select_label_list, attetions = [],[],[],[],[]
    for i in range(test_batch_num):
        _, _, select_user_input, select_num_idx, select_item_input, select_label_input, _, _, _,_ = sampling_RL(test_user_input[i], test_num_idx[i], test_item_input[i], test_labels[i], i, agent, Random=False)

        batched_user_input_list = test_user_input[i]
        batched_user_input = np.array([u for u in batched_user_input_list])
        batched_item_input = np.reshape(test_item_input[i], (-1, 1))
        batched_label_input = np.reshape(test_labels[i], (-1, 1))
        batched_num_idx = np.reshape(test_num_idx[i],(-1,1))
        
        predictions, attention, loss = recommender.predict_with_atteionts(
            batched_user_input, batched_num_idx, batched_item_input, batched_label_input)
        select_user_input_list.append(select_user_input)
        select_item_input_list.append(select_item_input)
        select_num_idx_list.append(select_num_idx)
        select_label_list.append(select_label_input)
        attetions.append(attention)

    return [select_user_input_list,select_num_idx_list,select_item_input_list, select_label_list, attetions]


if __name__ == '__main__':
    args = setting()
    config = tf.ConfigProto()
    dataset = Dataset(args.datapath, args.num_neg, args.batch_size, args.fast_running)
    padding_number = dataset.num_items
    pos_instances = dataset.get_positive_instances()
    test_instances = dataset.get_test_instances()
    print "Load course names"
    course_dict = load_course(args.datapath)
    print "Loaded course names"
    env = Environment()
    
    with tf.Session(config = config) as sess:
        recommender = RecommenderNetwork(sess, padding_number, args)
        agent = AgentNetwork(sess, args)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.pre_recommender+'checkpoint')).model_checkpoint_path)
        print "recommender loaded"
        saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.agent+'checkpoint')).model_checkpoint_path)
        print "agent loaded"
        env.initilize_state(recommender, pos_instances, test_instances, args.high_state_size, args.low_state_size, padding_number)
        print "Envoriment initialized."
        select_instances = evalute(agent, recommender, test_instances)
        output(course_dict, test_instances, select_instances)

