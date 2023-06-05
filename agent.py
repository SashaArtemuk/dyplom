import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, state_size, eval_mode=False, model_name=None, lr=0.001):
        self.state_size = state_size
        self.action_size = 3 # hold, buy, sell
        self.memory = deque(maxlen=500)
        self.inventory = []
        self.model_name = model_name
        self.eval_mode = eval_mode
        self.lr = lr

        self.gamma = 0.95
        self.epsilon = 1.0 if not model_name else 0.005
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.995

        if model_name:
            self.model = load_model(f"models/{model_name}.h5")
        else:
            self.model = self.get_model()

    def get_model(self) -> Model:
        model = Sequential()
        model.add(Dense(units=256, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))

        return model

    def perform_action(self, state: np.array) -> int:
        if not self.eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    def experience_replay(self, batch_size):
        memory_len = len(self.memory)
        mini_batch = [self.memory[i] for i in range(memory_len - batch_size + 1, memory_len)]

        # memory index to value mapping: 0:state, 1:action, 2:reward, 3:next_state, 4:is_done
        next_states = tf.data.Dataset.from_tensor_slices([el[3] for el in mini_batch])
        states = tf.data.Dataset.from_tensor_slices([el[0] for el in mini_batch])
        targets = self.model.predict(next_states)
        target_fs = self.model.predict(states) #, workers=8,use_multiprocessing=True,

        x = []
        y = []
        for i, (state, action, reward, next_state, is_done) in enumerate(mini_batch):
            target = reward
            if not is_done:
                target = reward + self.gamma * np.amax(targets[i][0])
            target_f = np.array([target_fs[i]])

            target_f[0][action] = target
            x.append(state)
            y.append(target_f)

        train_data = tf.data.Dataset.from_tensor_slices((x, y))
        self.model.fit(train_data, epochs=1, verbose=0,) # workers=8,use_multiprocessing=True,)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
