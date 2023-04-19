import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import glob
import random
import time
import matplotlib.pyplot as plt
correct = 0


def read_csv_file(file_path, nrows):
    column_names = ['timestamp', 'action_index', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    dtype = {'acc_x': 'int', 'acc_y': 'int', 'acc_z': 'int', 'gyro_x': 'int', 'gyro_y': 'int', 'gyro_z': 'int'}
    df = pd.read_csv(file_path, header=None, names=column_names, nrows=nrows, dtype=dtype)
    return df


def calculate_average_difference(df1, df2, column):
    return int(abs(df1[column] - df2[column]).mean())


def create_fuzzy_system():
    max_range = 32767
    acc_x_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'acc_x')
    acc_y_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'acc_y')
    acc_z_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'acc_z')
    gyro_x_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'gyro_x')
    gyro_y_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'gyro_y')
    gyro_z_diff = ctrl.Antecedent(np.arange(0, max_range + 1, 1), 'gyro_z')

    similarity = ctrl.Consequent(np.arange(0, 101, 1), 'similarity')

    for antecedent in [acc_x_diff, acc_y_diff, acc_z_diff, gyro_x_diff, gyro_y_diff, gyro_z_diff]:
        antecedent['low'] = fuzz.trimf(antecedent.universe, [0, 0, max_range / 2])
        antecedent['high'] = fuzz.trimf(antecedent.universe, [max_range / 2, max_range, max_range])

    similarity['low'] = fuzz.trimf(similarity.universe, [0, 0, 50])
    similarity['high'] = fuzz.trimf(similarity.universe, [50, 100, 100])

    rules = []

    for antecedent in [acc_x_diff, acc_y_diff, acc_z_diff, gyro_x_diff, gyro_y_diff, gyro_z_diff]:
        rules.append(ctrl.Rule(antecedent['low'], similarity['high']))
        rules.append(ctrl.Rule(antecedent['high'], similarity['low']))

    rules.append(ctrl.Rule(
        acc_x_diff['low'] & acc_y_diff['low'] & acc_z_diff['low'] & gyro_x_diff['low'] & gyro_y_diff['low'] &
        gyro_z_diff['low'], similarity['high']))

    similarity_ctrl = ctrl.ControlSystem(rules)
    similarity_sim = ctrl.ControlSystemSimulation(similarity_ctrl)

    return similarity_sim


def check_similarity(fuzzy_system, avg_diffs):
    for key, value in avg_diffs.items():
        fuzzy_system.input[key] = value

    fuzzy_system.compute()

    return fuzzy_system.output['similarity']


def verify_result(df, result):
    global correct
    action_list = ["None", "grenade", "reload", "shield", "logout"]
    target = df['action_index'][0]
    pred = max(result, key=result.get)

    if target == pred:
        correct = correct + 1
        return True
    return False


def check_csv_similarity_with_base(file_path):
    base_path = './base_actions/'
    base_files = glob.glob('./base_actions/*.csv')
    nrows = 50  # adjust as needed
    file2 = file_path
    df2 = read_csv_file(file2, nrows)
    fuzzy_system = create_fuzzy_system()
    result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for base in base_files:
        file1 = base
        df1 = read_csv_file(file1, nrows)

        columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        avg_diffs = {col: calculate_average_difference(df1, df2, col) for col in columns}
        # print(avg_diffs)
        similarity_score = check_similarity(fuzzy_system, avg_diffs)

        # print(f"Similarity score for {df1['action_index'][0]}: {similarity_score}")
        if similarity_score > result[df1['action_index'][0]]:
            result[df1['action_index'][0]] = similarity_score

    verify_result(df2, result)
    # print(verify_result(df2, result))


def main():
    # List all files with a '.csv' extension in the folder
    csv_files = glob.glob('data_2023-04-08/*.csv')
    #csv_files = glob.glob('*.csv')
    random.shuffle(csv_files)
    acc = 0
    acc_values = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    for n, csv_file in enumerate(csv_files):
        start_time = time.time()
        check_csv_similarity_with_base(csv_file)
        elapsed_time = time.time() - start_time
        acc = correct * 100 / (n + 1)
        acc_values.append(acc)
        print(elapsed_time, csv_file, acc)
        ax.clear()
        ax.set_ylim(0, 120)
        ax.plot(acc_values, label="Accuracy")
        ax.legend(loc='upper right')

        plt.draw()
        plt.pause(0.1)  # Adjust the time interval between updates

    plt.ioff()  # Turn off interactive mode
    plt.show()


if __name__ == "__main__":
    main()
