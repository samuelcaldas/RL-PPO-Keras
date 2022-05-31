from keras.models import Model, model_from_json, load_model
from keras.optimizers import Adam, RMSprop
import os
from keras.layers import Input, Dense
import keras.backend as K
import time
from copy import deepcopy
import numpy as np


class Memory:
    def __init__(self):
        self.batch_state = []
        self.batch_action = []
        self.batch_reward = []
        self.batch_new_state = []
        self.batch_done = []

    def store(self, state, action, new_state, reward, done):
        self.batch_state.append(state)
        self.batch_action.append(action)
        self.batch_reward.append(reward)
        self.batch_new_state.append(new_state)
        self.batch_done.append(done)

    def clear(self):
        self.batch_state.clear()
        self.batch_action.clear()
        self.batch_reward.clear()
        self.batch_new_state.clear()
        self.batch_done.clear()

    @property
    def cnt_samples(self):
        return len(self.batch_state)


class Agent:
    def __init__(self, dictionary_agent_configuration, dictionary_path, dictionary_env_configuration):
        self.dictionary_agent_configuration = dictionary_agent_configuration
        self.dictionary_path = dictionary_path
        self.dictionary_env_configuration = dictionary_env_configuration

        self.num_actions = self.dictionary_agent_configuration["ACTION_DIM"]

        self.actor_network = self._build_actor_network()
        self.actor_old_network = self.build_network_from_copy(self.actor_network)

        self.critic_network = self._build_critic_network()

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self.num_actions))

        self.memory = Memory()

    def choose_action(self, state):
        assert isinstance(state, np.ndarray), "state must be numpy.ndarry"

        state = np.reshape(state, [-1, self.dictionary_agent_configuration["STATE_DIM"][0]])
        prob = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        action = np.random.choice(self.num_actions, p=prob)
        return action

    def train_network(self):
        n = self.memory.cnt_samples
        discounted_reward = []
        if self.memory.batch_done[-1]:
            value = 0
        else:
            value = self.get_value(self.memory.batch_new_state[-1])
        for r in self.memory.batch_reward[::-1]:
            value = r + self.dictionary_agent_configuration["GAMMA"] * value
            discounted_reward.append(value)
        discounted_reward.reverse()

        batch_state, batch_action, batch_discounted_reward = np.vstack(self.memory.batch_state), \
                     np.vstack(self.memory.batch_action), \
                     np.vstack(discounted_reward)

        batch_value = self.get_value(batch_state)
        batch_advantage = batch_discounted_reward - batch_value
        batch_old_prediction = self.get_old_prediction(batch_state)

        batch_action_final = np.zeros(shape=(len(batch_action), self.num_actions))
        batch_action_final[:, batch_action.flatten()] = 1
        # print(batch_s.shape, batch_advantage.shape, batch_old_prediction.shape, batch_a_final.shape)
        self.actor_network.fit(x=[batch_state, batch_advantage, batch_old_prediction], y=batch_action_final, verbose=0)
        self.critic_network.fit(x=batch_state, y=batch_discounted_reward, epochs=2, verbose=0)
        self.memory.clear()
        self.update_target_network()

    def get_old_prediction(self, state):
        state = np.reshape(state, (-1, self.dictionary_agent_configuration["STATE_DIM"][0]))
        return self.actor_old_network.predict_on_batch(state)

    def store_transition(self, state, action, new_state, reward, done):
        self.memory.store(state, action, new_state, reward, done)

    def get_value(self, state):
        state = np.reshape(state, (-1, self.dictionary_agent_configuration["STATE_DIM"][0]))
        value = self.critic_network.predict_on_batch(state)
        return value

    def save_model(self, file_name):
        self.actor_network.save(os.path.join(self.dictionary_path["PATH_TO_MODEL"], "%s_actor_network.h5" % file_name))
        self.critic_network.save(os.path.join(self.dictionary_path["PATH_TO_MODEL"], "%s_critic_network.h5" % file_name))

    def load_model(self):
        self.actor_network = load_model(self.dictionary_path["PATH_TO_MODEL"], "%s_actor_network.h5")
        self.critic_network = load_model(self.dictionary_path["PATH_TO_MODEL"], "%s_critic_network.h5")
        self.actor_old_network = deepcopy(self.actor_network)

    def _build_actor_network(self):

        state = Input(shape=self.dictionary_agent_configuration["STATE_DIM"], name="state")

        advantage = Input(shape=(1, ), name="Advantage")
        old_prediction = Input(shape=(self.num_actions,), name="Old_Prediction")

        shared_hidden = self._shared_network_structure(state)

        action_dim = self.dictionary_agent_configuration["ACTION_DIM"]

        policy = Dense(action_dim, activation="softmax", name="actor_output_layer")(shared_hidden)

        actor_network = Model(inputs=[state, advantage, old_prediction], outputs=policy)

        if self.dictionary_agent_configuration["OPTIMIZER"] is "Adam":
            actor_network.compile(optimizer=Adam(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]),
                                  loss=self.proximal_policy_optimization_loss(
                                    advantage=advantage, old_prediction=old_prediction,
                                  ))
        elif self.dictionary_agent_configuration["OPTIMIZER"] is "RMSProp":
            actor_network.compile(optimizer=RMSprop(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]))
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            actor_network.compile(optimizer=Adam(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]))
        print("=== Build Actor Network ===")
        actor_network.summary()

        time.sleep(1.0)
        return actor_network

    def update_target_network(self):
        alpha = self.dictionary_agent_configuration["TARGET_UPDATE_ALPHA"]
        self.actor_old_network.set_weights(alpha*np.array(self.actor_network.get_weights())
                                           + (1-alpha)*np.array(self.actor_old_network.get_weights()))

    def _build_critic_network(self):
        state = Input(shape=self.dictionary_agent_configuration["STATE_DIM"], name="state")
        shared_hidden = self._shared_network_structure(state)

        if self.dictionary_env_configuration["POSITIVE_REWARD"]:
            q = Dense(1, activation="relu", name="critic_output_layer")(shared_hidden)
        else:
            q = Dense(1, name="critic_output_layer")(shared_hidden)

        critic_network = Model(inputs=state, outputs=q)

        if self.dictionary_agent_configuration["OPTIMIZER"] is "Adam":
            critic_network.compile(optimizer=Adam(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]),
                                   loss=self.dictionary_agent_configuration["CRITIC_LOSS"])
        elif self.dictionary_agent_configuration["OPTIMIZER"] is "RMSProp":
            critic_network.compile(optimizer=RMSprop(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]),
                                   loss=self.dictionary_agent_configuration["CRITIC_LOSS"])
        else:
            print("Not such optimizer for actor network. Instead, we use adam optimizer")
            critic_network.compile(optimizer=Adam(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]),
                                   loss=self.dictionary_agent_configuration["CRITIC_LOSS"])
        print("=== Build Critic Network ===")
        critic_network.summary()

        time.sleep(1.0)
        return critic_network

    def build_network_from_copy(self, actor_network):
        network_structure = actor_network.to_json()
        network_weights = actor_network.get_weights()
        network = model_from_json(network_structure)
        network.set_weights(network_weights)
        network.compile(optimizer=Adam(lr=self.dictionary_agent_configuration["ACTOR_LEARNING_RATE"]), loss="mse")
        return network

    def _shared_network_structure(self, state_features):
        dense_d = self.dictionary_agent_configuration["D_DENSE"]
        hidden1 = Dense(dense_d, activation="relu", name="hidden_shared_1")(state_features)
        hidden2 = Dense(dense_d, activation="relu", name="hidden_shared_2")(hidden1)
        return hidden2

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        loss_clipping = self.dictionary_agent_configuration["CLIPPING_LOSS_RATIO"]
        entropy_loss = self.dictionary_agent_configuration["ENTROPY_LOSS_RATIO"]

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage) + entropy_loss * (
                           prob * K.log(prob + 1e-10)))

        return loss
