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


def create_fuzzy_system():
    max_range = 32767
    max_similarity = 100

    acc_x_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'acc_x_diff')
    acc_y_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'acc_y_diff')
    acc_z_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'acc_z_diff')
    gyro_x_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'gyro_x_diff')
    gyro_y_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'gyro_y_diff')
    gyro_z_diff = ctrl.Antecedent(np.arange(-max_range, max_range + 1, 1), 'gyro_z_diff')

    similarity = ctrl.Consequent(np.arange(0, max_similarity + 1, 1), 'similarity')

    # define fuzzy set
    for antecedent in [acc_x_diff, acc_y_diff, acc_z_diff, gyro_x_diff, gyro_y_diff, gyro_z_diff]:
        antecedent['NL'] = fuzz.trimf(antecedent.universe, [-max_range, -max_range, -max_range * 2 / 3])
        antecedent['NM'] = fuzz.trimf(antecedent.universe, [-max_range, -max_range * 2 / 3, -max_range / 3])
        antecedent['NS'] = fuzz.trimf(antecedent.universe, [-max_range * 2 / 3, -max_range / 3, 0])
        antecedent['AZ'] = fuzz.trimf(antecedent.universe, [-max_range / 3, 0, max_range / 3])
        antecedent['PS'] = fuzz.trimf(antecedent.universe, [0, max_range / 3, max_range * 2 / 3])
        antecedent['PM'] = fuzz.trimf(antecedent.universe, [max_range / 3, max_range * 2 / 3, max_range])
        antecedent['PL'] = fuzz.trimf(antecedent.universe, [max_range * 2 / 3, max_range, max_range])

    similarity['AZ'] = fuzz.trimf(similarity.universe, [0, 0, max_similarity / 3])
    similarity['PS'] = fuzz.trimf(similarity.universe, [0, max_similarity / 3, max_similarity * 2 / 3])
    similarity['PM'] = fuzz.trimf(similarity.universe, [max_similarity / 3, max_similarity * 2 / 3, max_similarity])
    similarity['PL'] = fuzz.trimf(similarity.universe, [max_similarity * 2 / 3, max_similarity, max_similarity])

    rules = []
    for antecedent in [acc_x_diff, acc_y_diff, acc_z_diff, gyro_x_diff, gyro_y_diff, gyro_z_diff]:
        rules.append(ctrl.Rule(antecedent['NL'], similarity['AZ']))
        rules.append(ctrl.Rule(antecedent['NM'], similarity['PS']))
        rules.append(ctrl.Rule(antecedent['NS'], similarity['PM']))
        rules.append(ctrl.Rule(antecedent['AZ'], similarity['PL']))
        rules.append(ctrl.Rule(antecedent['PS'], similarity['PM']))
        rules.append(ctrl.Rule(antecedent['PM'], similarity['PS']))
        rules.append(ctrl.Rule(antecedent['PL'], similarity['AZ']))
    similarity_ctrl = ctrl.ControlSystem(rules)

    similarity_sim = ctrl.ControlSystemSimulation(similarity_ctrl)
    return similarity_sim


def check_similarity(fuzzy_system, col_diffs):
    for key, value in col_diffs.items():
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
        similarity_score = 0
        for i in range(nrows):
            columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            col_diffs = {col + '_diff': df1[col][i] - df2[col][i] for col in columns}
            similarity_score = similarity_score + check_similarity(fuzzy_system, col_diffs)
            # print(col_diffs)
            # # print(f"Similarity score for {df1['action_index'][0]}: {similarity_score}")
        similarity_score = similarity_score / nrows
        if similarity_score > result[df1['action_index'][0]]:
            result[df1['action_index'][0]] = similarity_score

    verify_result(df2, result)
    print(result)


def main():
    # List all files with a '.csv' extension in the folder
    csv_files = glob.glob('data_2023-04-08/*.csv')
    # csv_files = glob.glob('*.csv')
    random.shuffle(csv_files)
    acc = 0
    acc_values = []

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    for n, csv_file in enumerate(csv_files):
        start_time = time.time()
        check_csv_similarity_with_base(csv_file)
        elapsed_time = time.time() - start_time
        acc = correct * 100 / (n+1)
        acc_values.append(acc)
        print(elapsed_time, csv_file,acc)
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
