"""Linear QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False


GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 5 # 10(previous value)
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# pragma: coderesponse template name="linear_epsilon_greedy"
def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here
    # initial
    action_index, object_index = None, None

    # Generate a non-uniform random result as if a were np.arange(a)
    # https://www.sharpsightlabs.com/blog/numpy-random-choice/
    result = np.random.choice(a=[0, 1], p=[epsilon, 1 - epsilon])

    # optimal policy
    if result == 1:
        Q = np.inner(theta, state_vector)  # for each value of Q is superimposed by a state vector and theta (weighting)
        # fetch max argument index by np.unravel_index
        # https://www.geeksforgeeks.org/numpy-unravel_index-function-python/
        index = np.unravel_index(np.argmax(Q, axis=None), Q.shape)[0]  # (n,) -> into numpy.int64
        action_index, object_index = index2tuple(index)  # Converts an index c to a tuple (a,b)
    # random select action-object
    if result == 0:
        action_index = np.random.randint(NUM_ACTIONS)
        object_index = np.random.randint(NUM_OBJECTS)

    return (action_index, object_index)

# pragma: coderesponse end


# pragma: coderesponse template
def linear_q_learning(theta, current_state_vector, action_index, object_index,
                      reward, next_state_vector, terminal):
    """Update theta for a given transition

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    # TODO Your code here

    # https://stats.stackexchange.com/questions/187110/how-to-fit-weights-into-q-values-with-linear-function-approximation
    # https://www.baeldung.com/cs/epsilon-greedy-q-learning

    # Renaming to shorter name
    a_idx = action_index    # rename for shorter name for action index
    o_idx = object_index    # rename for shorter name for object index

    # Q(s,a,??)
    q_value = (theta @ current_state_vector)[tuple2index(a_idx, o_idx)]

    # Q(s???,c???,??)
    parametrized_function = theta @ next_state_vector

    if terminal:
        max_q_value_next = 0
    else:
        max_q_value_next = np.max(parametrized_function)


    #  y = R(s,c) + ??maxc???Q(s???,c???,??)
    y = reward + (GAMMA * max_q_value_next)

    # phi(s,c) = current_state_vector
    delta_theta = (y - q_value) * current_state_vector

    # update theta
    # ??	?????+??g(??)=??+??[R(s,c)+??maxc???Q(s???,c???,??)???Q(s,c,??)]??(s,c)
    theta[tuple2index(a_idx, o_idx)] = theta[tuple2index(a_idx, o_idx)] + (ALPHA * delta_theta)

# pragma: coderesponse end


def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    # initialize for each episode
    # TODO Your code here

    # Look into framework.py file for hint
    # A tuple where the first element is a description of the initial room,
    # the second element is a description of the quest for this new game episode, and
    # the last element is a Boolean variable with value False implying that the game is not over.
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    # initial value
    count = 0
    epi_reward = 0

    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(current_state, dictionary)
        # TODO Your code here
        (action_index, object_index) = epsilon_greedy(current_state_vector, theta, epsilon)

        # Renaming to shorter name
        a_idx = action_index  # rename for shorter name for action index
        o_idx = object_index  # rename for shorter name for object index
        crd = current_room_desc  # rename for shorter name for current room description
        cqd = current_quest_desc  # rename for shorter name for current quest description

        # the system next state when the selected command is applied at the current state
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(crd, cqd, a_idx, o_idx)

        next_state = next_room_desc + next_quest_desc
        # Look into utils.py for the bag-of-words vector representation of the state
        next_state_vector = utils.extract_bow_feature_vector(next_state, dictionary)

        if for_training:
            # update Q-function.
            # TODO Your code here
            linear_q_learning(theta, current_state_vector, a_idx, o_idx, reward, next_state_vector, terminal)
            pass

        if not for_training:
            # update reward
            # TODO Your code here
            epi_reward += np.power(GAMMA, count) * reward
            pass

        # prepare next step
        # TODO Your code here
        count += 1
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global theta
    theta = np.zeros([action_dim, state_dim])

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)
    action_dim = NUM_ACTIONS * NUM_OBJECTS

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()

