import numpy as np
import tensorflow as tf
import os
from time import time
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from Environment import Environment
from Agent import AgentNetwork
from Recommender import RecommenderNetwork
from DataGenerator import Dataset
from Evaluation import eval_rating
from Setting import setting

global padding_number
global env



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


def train(sess, agent, recommender, trainset, testset, args, recommender_trainable=True, agent_trainable=True):
    train_user_input, train_num_idx, train_item_input, train_labels, train_batch_num  = (trainset[0], trainset[1], trainset[2], trainset[3], trainset[4])
    sample_times = args.sample_cnt
    high_state_size = args.high_state_size
    low_state_size = args.low_state_size
    avg_loss = 0

    shuffled_batch_indexes = np.random.permutation(int(train_batch_num))
    for batch_index in shuffled_batch_indexes:

        batched_user_input = np.array([u for u in train_user_input[batch_index]])
        batched_item_input = np.reshape(train_item_input[batch_index], (-1, 1))
        batched_label_input = np.reshape(train_labels[batch_index], (-1, 1))
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))


        batch_size = batched_user_input.shape[0]
        max_course_num = batched_user_input.shape[1]


        train_begin = time()
        train_loss = 0
        agent.assign_active_high_network()
        agent.assign_active_low_network()
        recommender.assign_active_network()
        if agent_trainable:

            sampled_high_states = np.zeros((sample_times, batch_size, high_state_size), dtype=np.float32)
            sampled_high_actions = np.zeros((sample_times, batch_size), dtype=np.int)

            sampled_low_states = np.zeros((sample_times, batch_size, max_course_num, low_state_size), dtype=np.float32)
            sampled_low_actions = np.zeros((sample_times, batch_size, max_course_num), dtype=np.float32)

            sampled_high_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)
            sampled_low_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)

            sampled_revise_index = []

            avg_high_reward = np.zeros((batch_size), dtype=np.float32)
            avg_low_reward = np.zeros((batch_size), dtype=np.float32)

            for sample_time in range(sample_times):
                high_action, high_state, select_user_input, select_num_idx, item_input, label_input, notrevised_index, revised_index, delete_index, keep_index =  sampling_RL(batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent)
                sampled_high_actions[sample_time, :] = high_action
                sampled_high_states[sample_time, :] = high_state
                sampled_revise_index.append(revised_index)


                _, _, reward = env.get_reward(recommender, batch_index, high_action, select_user_input, select_num_idx, batched_item_input, batched_label_input)

                # reward = np.sqrt(np.multiply(reward,env.num_selected)/env.num_idx[batch_index])   # Geometric mean
                avg_high_reward += reward
                avg_low_reward += reward
                sampled_high_rewards[sample_time, :] = reward
                sampled_low_rewards[sample_time, :] = reward
                sampled_low_actions[sample_time, :] = env.get_action_matrix()
                sampled_low_states[sample_time, :] = env.get_state_matrix()

            avg_high_reward = avg_high_reward / sample_times
            avg_low_reward = avg_low_reward / sample_times
            high_gradbuffer = agent.init_high_gradbuffer()
            low_gradbuffer = agent.init_low_gradbuffer()
            for sample_time in range(sample_times):
                high_reward = np.subtract(sampled_high_rewards[sample_time], avg_high_reward)
                high_gradient = agent.get_high_gradient(sampled_high_states[sample_time], high_reward,sampled_high_actions[sample_time] )
                agent.train_high(high_gradbuffer, high_gradient)

                revised_index = sampled_revise_index[sample_time]
                low_reward = np.subtract(sampled_low_rewards[sample_time], avg_low_reward)
                low_reward_row = np.tile(np.reshape(low_reward[revised_index], (-1, 1)), max_course_num)
                low_gradient = agent.get_low_gradient(
                    np.reshape(sampled_low_states[sample_time][revised_index], (-1, low_state_size)),
                    np.reshape(low_reward_row, (-1,)),
                    np.reshape(sampled_low_actions[sample_time][revised_index], (-1,)))
                agent.train_low(low_gradbuffer, low_gradient)

            if recommender_trainable:
                _, _, select_user_input, select_num_idx, _, _, _, _, _, _ =  sampling_RL(
                     batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent, Random=False)
                train_loss,_ = recommender.train(select_user_input, np.reshape(select_num_idx,(-1,1)), batched_item_input,  batched_label_input)
        else:
            train_loss,_ = recommender.train(batched_user_input,batched_num_idx , batched_item_input, batched_label_input)
        avg_loss += train_loss
        train_time = time() - train_begin


        # Update parameters
        if agent_trainable:
            agent.update_target_high_network()
            agent.update_target_low_network()
            if recommender_trainable:
                recommender.update_target_network()
        else:
            recommender.assign_target_network()

    return avg_loss / train_batch_num


def get_avg_reward(agent, trainset):
    train_user_input, train_num_idx, train_item_input, train_labels, train_batch_num  = (trainset[0], trainset[1], trainset[2], trainset[3], trainset[4])
    avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances = 0,0,0,0,0,0
    total_instances = 0
    test_begin = time()
    for batch_index in range(train_batch_num):
        batched_user_input = np.array([u for u in train_user_input[batch_index]])
        batched_item_input = np.reshape(train_item_input[batch_index], (-1, 1))
        batched_label_input = np.reshape(train_labels[batch_index], (-1, 1))
        batched_num_idx = np.reshape(train_num_idx[batch_index], (-1,1))

        high_action, high_state, select_user_input, select_num_idx, _, _, notrevised_index, revised_index, delete_index, keep_index =  sampling_RL(batched_user_input, batched_num_idx, batched_item_input, batched_label_input, batch_index, agent, Random=False)
        _, _, reward = env.get_reward(recommender, batch_index, high_action, select_user_input, select_num_idx, batched_item_input, batched_label_input)

        avg_reward += np.sum(reward)
        total_selected_courses += np.sum(select_num_idx)
        total_revised_instances += len(revised_index)
        total_notrevised_instances += len(notrevised_index)
        total_deleted_instances += len(delete_index)
        total_keep_instances += len(keep_index)
        total_instances += batched_user_input.shape[0]
    test_time = time() - test_begin
    avg_reward = avg_reward / total_instances
    return avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances, test_time


def evalute(agent, recommender,testset, noAgent=False):
    test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num  = (testset[0], testset[1], testset[2], testset[3], testset[4])
    if noAgent:
        return eval_rating(recommender, test_user_input, test_num_idx, test_item_input, test_labels, test_batch_num)
    else:
        env.set_test_original_rewards()
        select_user_input_list, select_num_idx_list, select_item_input_list, select_label_list = [],[],[],[]
        for i in range(test_batch_num):
            # print test_user_input[i].shape, test_num_idx[i].shape, test_item_input[i].shape, test_labels[i].shape
            _, _, select_user_input, select_num_idx, select_item_input, select_label_input, _, _, _,_ = sampling_RL(test_user_input[i], test_num_idx[i], test_item_input[i], test_labels[i], i, agent, Random=False)
            select_user_input_list.append(select_user_input)
            select_item_input_list.append(select_item_input)
            select_num_idx_list.append(select_num_idx)
            select_label_list.append(select_label_input)
        env.set_train_original_rewards()
        return eval_rating(recommender, select_user_input_list, select_num_idx_list, select_item_input_list, select_label_list, test_batch_num)


def print_recommender_message(unit, index, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, train_loss, train_time):
    logging.info(
        "%s %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
            unit,  index, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, train_loss, train_time))
    print(
        "%s %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
            unit, index, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, train_loss,
            train_time))

def print_agent_message(epoch, avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time, train_time ):
    partial_revised = total_revised_instances-total_deleted_instances-total_keep_instances
    logging.info(
        "Epoch %d : avg reward = %.4f, courses (keep = %d), instances (revise = %d, notrevise = %d, delete = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
            epoch,  avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
    print(
        "Epoch %d : avg reward = %.4f, courses (keep = %d), instances (revise = %d, notrevise = %d, delete = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
            epoch,  avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances, total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                filename='log.txt',
                filemode='w',
                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    args = setting()
    config = tf.ConfigProto()
    dataset = Dataset(args.datapath, args.num_neg, args.batch_size, args.fast_running)
    padding_number = dataset.num_items
    pos_instances = dataset.get_positive_instances()
    test_instances = dataset.get_test_instances()
    pos_and_neg_instances = dataset.get_dataset_with_neg()

    env = Environment()

    with tf.Session(config=config) as sess:
        recommender = RecommenderNetwork(sess, padding_number, args)
        agent = AgentNetwork(sess, args)
        # print variables
        # for item in tf.trainable_variables():
        #     print(item.name, item.get_shape())
        pre_recommender_saver = tf.train.Saver()
        pre_agent_saver = tf.train.Saver()
        recommender_saver = tf.train.Saver()
        agent_saver = tf.train.Saver()    

        #Recommender pretrain
        if args.recommender_pretrain:
            best_ndcg10 = 0.0
            best_hr = 0.0
            sess.run(tf.global_variables_initializer())

            for epoch in range(args.recommender_epochs):
                train_begin = time()
                train_loss = train(sess, agent, recommender, pos_and_neg_instances, test_instances, args, recommender_trainable=True, agent_trainable=False)
                train_time = time() - train_begin
                recommender.assign_target_network()
                pos_and_neg_instances = dataset.get_dataset_with_neg()

                # recommender.assign_target_network()
                if epoch % args.recommender_verbose == 0:
                    test_begin = time()
                    (hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss) = evalute(agent, recommender, test_instances, noAgent=True)
                    test_time = time() - test_begin
                    print_recommender_message("Epoch", epoch,hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, train_loss, train_time)
                    if hr10 >=best_hr or ndcg10 >= best_ndcg:
                        best_hr = hr10
                        best_ndcg = ndcg10
                        pre_recommender_saver.save(sess, args.pre_recommender, global_step=epoch)
            print ("Recommender pretrain OK")
            logging.info("Recommender pretrain OK")
        print ("Load best pre-trained recommender from ", args.pre_recommender)
        logging.info("Load best pre-trained recommender from %s " % args.pre_recommender)
        pre_recommender_saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.pre_recommender+'checkpoint')).model_checkpoint_path)
        print ("Evaluate pre-trained recommender based on original test instances")
        logging.info("Evaluate pre-trained recommender based on original test instances")
        test_begin = time()
        (hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss) = evalute(agent, recommender, test_instances, noAgent=True)
        test_time = time() - test_begin
        print_recommender_message("Epoch", -1, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, 0, 0)


        #Agent pretrain
        env.initilize_state(recommender, pos_instances, test_instances, args.high_state_size, args.low_state_size,padding_number)
        if args.agent_pretrain:
            best_avg_reward = -1000

            for epoch in range(args.agent_epochs):
                train_begin = time()
                train_loss = train(sess, agent, recommender, pos_instances, test_instances, args, recommender_trainable=False, agent_trainable=True)
                train_time = time() - train_begin
                if epoch % args.agent_verbose == 0:
                    test_begin = time()
                    avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances, test_time = get_avg_reward(agent, pos_instances)
                    test_time = time() - test_begin
                    print_agent_message(epoch, avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time,train_time)
                    if avg_reward >= best_avg_reward:
                        best_avg_reward = avg_reward
                        pre_agent_saver.save(sess, args.pre_agent, global_step=epoch)
            print ("Agent pretrain OK")
        print ("Load best pre-trained agent from", args.pre_agent)
        logging.info("Load best pre-trained agent from %s" % args.pre_agent)
        pre_agent_saver.restore(sess, tf.train.get_checkpoint_state(os.path.dirname(args.pre_agent+'checkpoint')).model_checkpoint_path)
        print ("Evaluate pre-trained recommender based on the selected test instances by the pre-trained agent")
        logging.info("Evaluate pre-trained recommender based on the selected test instances by the pre-trained agent")
        test_begin = time()
        (hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss) = evalute(agent, recommender, test_instances)
        test_time = time() - test_begin
        print_recommender_message("Epoch", -1, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, 0, 0)


        #Agent and recommender jointly train
        print ("Begin to jointly train")
        logging.info("Begin to jointly train")
        best_ndcg10 = 0.0
        best_hr = 0.0
        best_avg_reward = -1000
        print (agent.tau, agent.lr)
        agent.udpate_tau(args.agent_tau)
        agent.update_lr(args.agent_lr)
        print (agent.tau, agent.lr)
        for epoch in range(0,5):
            train_begin = time()
            train_loss = train(sess, agent, recommender, pos_instances, test_instances, args, recommender_trainable=True, agent_trainable=True)
            train_time = time() - train_begin
            if epoch % 1 == 0:

                test_begin = time()
                avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances, test_time = get_avg_reward(agent, pos_instances)
                test_time = time() - test_begin
                print_agent_message(epoch, avg_reward, total_selected_courses, total_revised_instances,total_notrevised_instances, total_deleted_instances,total_keep_instances,test_time,train_time)
                if avg_reward >= best_avg_reward:
                    best_avg_reward = avg_reward
                    agent_saver.save(sess, args.agent, global_step=epoch)

            if epoch % 1 == 0:
                test_begin = time()
                (hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss) = evalute(agent, recommender, test_instances)
                test_time = time() - test_begin
                print_recommender_message("Epoch", epoch, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, train_loss, train_time)
                if hr10 >=best_hr or ndcg10 >= best_ndcg:
                    best_hr = hr10
                    best_ndcg = ndcg10
                    recommender_saver.save(sess, args.recommender, global_step=epoch)
            # To update the embeddings and the original rewards or not?
            env.initilize_state(recommender, pos_instances, test_instances, args.high_state_size, args.low_state_size,padding_number)
        print ("Jointly train OK")
        logging.info("Jointly train OK")
        print ("Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        logging.info("Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        test_begin = time()
        (hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss) = evalute(agent, recommender, test_instances)
        test_time = time() - test_begin
        print_recommender_message("Epoch", -1, hr5, ndcg5, hr10, ndcg10, map, mrr, test_loss, test_time, 0, 0)




