import time
import os
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from agent import Agent
import sys
from typing import Tuple, List
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def format_price(price) -> str:
    return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))


def get_data(dataset_name: str) -> List[Tuple[float, float, float, float, float]]:
    result = []
    lines = open("data/" + dataset_name + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        split_line = line.split(",")
        open_ = float(split_line[1])
        high = float(split_line[2])
        low = float(split_line[3])
        close_ = float(split_line[4])
        volume = float(split_line[6])
        result.append((open_,high,low,close_,volume))

    return result


def get_state(data: List[Tuple[int]], t: int, window_size: int) -> np.array:
    """
    Returns state representation ending at time t and with size of window_size
    """

    d = t - window_size + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with data[0]

    return np.array([block])


# disable useless logs
tensorflow.keras.utils.disable_interactive_logging()
# disable gpu usage
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
CUDA_VISIBLE_DEVICES=""

# parameters
stock_name = "BTCUSD_day_2"
window_size = 100
episode_count = 2000

# model number to continue training
model_number = 190
start_episode = model_number + 1 if model_number else 0
draw_graphs = False

# get and scale data
data = get_data(stock_name)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(np.float32(data).reshape(-1, 1)).flatten().tolist()

# model init
agent = Agent(window_size, model_name=f"model_ep{model_number}") if model_number else Agent(window_size)


data_length = len(data) - 1
batch_size = 32
buy_amount = 0.1 # buy 0.1 of share


for episode in range(start_episode, episode_count + 1):
    print(f"Episode {episode} / {episode_count}")

    state = get_state(scaled_data, 0, window_size)

    total_profit = 0
    balance = 30000 # initial balance

    agent.inventory = []
    agent.memory.clear()

    buy_history = []
    sell_history = []

    for t in range(data_length):
        # perform action
        action = agent.perform_action(state)

        # hold
        next_state = get_state(scaled_data, t + 1, window_size)
        reward = 10

        # buy
        # data[t][3] == close price for t-th element of data
        if action == 1 and data[t][3]*buy_amount<balance:
            # add bought share to inventory
            agent.inventory.append(data[t][3]*buy_amount)
            balance -= data[t][3]*buy_amount
            reward = 11
            buy_history.append((t,data[t][3]))
            print(f"{(t / data_length) * 100}% | Buy: {format_price(data[t][3] * buy_amount)} | Balance: {format_price(balance)}")

        # sell
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t][3] * buy_amount - bought_price, 0)*2
            total_profit += data[t][3] * buy_amount - bought_price
            balance += data[t][3] * buy_amount
            sell_history.append((t,data[t][3]))
            print(f"{(t / data_length) * 100}% | Sell: {format_price(data[t][3] * buy_amount)} | Profit: {format_price(data[t][3] * buy_amount - bought_price)} | Balance: {format_price(balance)}")

        is_done = True if t == data_length - 1 else False
        agent.memory.append((state, action, reward, next_state, is_done))
        state = next_state

        if is_done:
            # sell all shares at the end of data
            while len(agent.inventory):
                bought_price = agent.inventory.pop(0)
                reward = max(data[t][3] * buy_amount - bought_price, 0)/5
                total_profit += data[t][3] * buy_amount - bought_price
                balance += data[t][3] * buy_amount
                sell_history.append((t, data[t][3]))
                print(
                    f"{(t / data_length) * 100}% | Sell: {format_price(data[t][3] * buy_amount)} | Profit: {format_price(data[t][3] * buy_amount - bought_price)} | Balance: {format_price(balance)}")

            print("--------------------------------")
            print("Total Profit: " + format_price(total_profit))
            print("--------------------------------")

        # perform training
        if len(agent.memory) > batch_size:
            agent.experience_replay(batch_size)


    # save model
    if episode % 10 == 0:
        agent.model.save(f"models/model_ep{episode}.h5")

    if draw_graphs and len(sell_history)>3:
        plt.figure()
        plot_data = [(index, *el[0:4]) for index, el in enumerate(data)]
        df = pd.DataFrame(data=plot_data,  # values
                     index=[el[0] for el in plot_data],  # 1st column as index
                     columns = ["index","open","high","low","close"])
        up = df[df.close >= df.open]
        down = df[df.close < df.open]
        color1 = 'green'
        color2 = 'red'
        width = 2
        width2 = .5

        plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=color1)
        plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=color1)
        plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=color1)

        # Plotting down prices of the stock
        plt.bar(down.index, down.close - down.open, width, bottom=down.open, color=color2)
        plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=color2)
        plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=color2)

        plt.scatter([el[0] for el in buy_history], [el[1] for el in buy_history], marker="^", s=60, c="blue")
        plt.scatter([el[0] for el in sell_history], [el[1] for el in sell_history], marker="v", s=60,
                 c="purple",)

        plt.show()
