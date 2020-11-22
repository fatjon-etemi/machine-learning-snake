import random
import sys

import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

from Snake import Game

LR = 1e-3
goal_steps = 200
games_to_evaluate = 20
games_to_render = 3
env = Game()
balancing = True
balancing_factor = 4
selected_dataset = 'saved_expert_player.npy'
train_data_factor = 0.75
kernels = ['poly', 'rbf', 'linear']


def create_dnn_model():
    model = Sequential()
    model.add(Flatten(input_shape=(24, 1)))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_dnn(model, render=False):
    global df
    global dataset
    info = {'Model': "DNN"}
    shape_second_parameter = len(dataset[0][0])

    X = np.array([i[0] for i in dataset])
    X = X.reshape(-1, shape_second_parameter)
    y = [i[1] for i in dataset]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_data_factor, shuffle=False)

    start = time.time()
    history = model.fit(X_train.tolist(), y_train, validation_data=(X_test.tolist(), y_test), epochs=10, batch_size=16)
    info['Training time'] = time.time() - start
    info['Accuracy'] = history.history['val_accuracy'][-1]
    info['Average score'], info['Games'], info['Steps'] = evaluate(model=model, render=render, nn=True)
    df = df.append(info, ignore_index=True)

    return history, model


def get_train_and_test():
    global dataset
    shape_second_parameter = len(dataset[0][0])
    X = np.array([i[0] for i in dataset])
    X = X.reshape(-1, shape_second_parameter)
    y = np.array([np.argmax(i[1]) for i in dataset])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_data_factor, shuffle=False)
    return X_train, y_train, X_test, y_test


def train_svm(kernel, render=False):
    global df
    info = {'Model': "SVM (kernel=%s)" % kernel}
    x_train, y_train, x_test, y_test = get_train_and_test()
    model = SVC(kernel=kernel)
    start = time.time()
    model.fit(x_train, y_train)
    info['Training time'] = time.time() - start
    info['Accuracy'] = model.score(x_test, y_test)
    info['Average score'], info['Games'], info['Steps'] = evaluate(model=model, render=render)
    df = df.append(info, ignore_index=True)
    return model


def train_decision_tree(render=False):
    global df
    info = {'Model': 'Decision Tree'}
    x_train, y_train, x_test, y_test = get_train_and_test()
    model = DecisionTreeClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    info['Training time'] = time.time() - start
    info['Accuracy'] = model.score(x_test, y_test)
    info['Average score'], info['Games'], info['Steps'] = evaluate(model=model, render=render)
    df = df.append(info, ignore_index=True)
    return model


def train_gnb(render=False):
    global df
    info = {'Model': 'Gaussian Naive Bayes'}
    x_train, y_train, x_test, y_test = get_train_and_test()
    model = GaussianNB()
    start = time.time()
    model.fit(x_train, y_train)
    info['Training time'] = time.time() - start
    info['Accuracy'] = model.score(x_test, y_test)
    info['Average score'], info['Games'], info['Steps'] = evaluate(model=model, render=render)
    df = df.append(info, ignore_index=True)
    return model


def train_random_forest(render=False):
    global df
    info = {'Model': 'Random Forest'}
    x_train, y_train, x_test, y_test = get_train_and_test()
    model = RandomForestClassifier()
    start = time.time()
    model.fit(x_train, y_train)
    info['Training time'] = time.time() - start
    info['Accuracy'] = model.score(x_test, y_test)
    info['Average score'], info['Games'], info['Steps'] = evaluate(model=model, render=render)
    df = df.append(info, ignore_index=True)
    return model


def evaluate(model, nn=False, render=True):
    scores = []
    choices = []
    if render:
        games = games_to_render
    else:
        games = games_to_evaluate
    for _ in range(games):
        print('Evaluations ', _, " out of ", str(games), '\r', end='')
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            if render:
                env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, 3)
            else:
                if nn:
                    prediction = model.predict(prev_obs.reshape(-1, len(prev_obs), 1))
                    action = np.argmax(prediction[0])
                else:
                    prediction = model.predict(prev_obs.reshape(1, -1))
                    action = prediction[0]

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)
    return sum(scores) / len(scores), games, goal_steps


if __name__ == '__main__':
    df = pd.DataFrame(columns=['Model', 'Training time', 'Accuracy', 'Average score', 'Games', 'Steps'])
    dataset = np.load(selected_dataset, allow_pickle=True)[0]
    # balancing the data
    if balancing:
        dataset_0 = [i for i in dataset if np.argmax(i[1]) == 0]
        dataset_1 = [i for i in dataset if np.argmax(i[1]) == 1]
        dataset_2 = [i for i in dataset if np.argmax(i[1]) == 2]
        dataset = dataset_0 + dataset_1 + random.choices(dataset_2, k=int(len(dataset_2) / balancing_factor))
    print("Training data length:", len(dataset))
    result = input('1 = all, 0 = one')
    if result == '0' or result == '':
        train_dnn(create_dnn_model(), True)
        # train_svm('rbf', True)
    elif result == '1':
        for k in kernels:
            train_svm(kernel=k)
        train_decision_tree()
        train_gnb()
        train_random_forest()
        train_dnn(create_dnn_model())
        df.sort_values('Average score', axis=0, inplace=True, ascending=True, ignore_index=True)
        df.to_excel('table.xlsx')
    print(df)
    sys.exit()
