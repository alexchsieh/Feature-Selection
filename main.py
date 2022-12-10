import pandas as table
import copy
import numpy as np


def findDist(a, b, c):
    dist = 0
    for i in range(1, len(a)):
        if i in c:
            continue
        # distance formula calculation
        dist = dist + pow((a[i] - b[i]), 2)
    return np.sqrt(dist)


def feature_search(data, alg):

    current_set_of_features = []
    acc = 0
    isMax = False
    s = None

    # Accounts for the levels of the tree
    for i in range(1, len(data[0])):
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        # Accounts for each feature of the data set
        for k in range(1, len(data[0])):
            # Don't repeat for the same feature
            if k not in current_set_of_features:

                # Returns accuracy with new feature
                accuracy = leave_one_out_cross_validation(
                    data, current_set_of_features, k, alg)

                print(f'---Using features {current_set_of_features} and adding {k} has an accuracy of {round(accuracy * 100, 1)}%') if alg == 1 else print(
                    f'---Removing features {current_set_of_features} and also {k} has an accuracy of {round(accuracy * 100, 1)}%')

                # Keeps track of proper accuracy
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        # Adds feature to list to not use again
        current_set_of_features.append(feature_to_add_at_this_level)

        # Checks to see if your current accuracy beats best, if so replace best and save set
        if best_so_far_accuracy > acc:
            acc = best_so_far_accuracy
            s = copy.deepcopy(current_set_of_features)
            isMax = False

        # Once you've hit the peak, it'll start decreasing unless it's local maximum
        elif best_so_far_accuracy < acc and isMax == False:
            isMax = True
            print(
                '(Warning, accuracy has decreased! Continuing search in case of local maxima)')

        print(f'Feature set {current_set_of_features} was the best, accuracy is {round(best_so_far_accuracy * 100, 1)}%') if alg == 1 else print(
            f'Removing feature set {current_set_of_features} was the best, accuracy is {round(best_so_far_accuracy * 100, 1)}%')
        print()

    print(f'\nFinished search! The best feature subset is {s}, which has an accuracy of {round(acc * 100, 1)}%') if alg == 1 else print(
        f'\nFinished search! The best feature subset to remove is {s}, which has an accuracy of {round(acc * 100, 1)}%')


def leave_one_out_cross_validation(data, current_set_of_features, add_feature, alg):
    number_correctly_classified = 0

    # Implements a copy to manipulate in the cross validation
    a = copy.deepcopy(current_set_of_features)

    # Adds the new feature in question to test accuracy
    a.append(add_feature)

    # New set to keep track of which columns not to use in validation
    ignoreColumns = None

    # If forward, save all the columns you are ignoring from the total data set
    if alg == 1:
        # FIX THIS
        ignoreColumns = list(range(1, len(data[0])))
        ignoreColumns = [x for x in ignoreColumns if x not in a]

    # If backwards, give it current data set as you're going to remove
    else:
        ignoreColumns = a

    for i, currRow in enumerate(data):
        label_object_to_classify = currRow[0]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = float('inf')

        for k, currRowSkip in enumerate(data):
            # skips itself from classification
            if k == i:
                continue

            dist = findDist(currRow, currRowSkip, ignoreColumns)

            if dist < nearest_neighbor_distance:
                nearest_neighbor_distance = dist
                nearest_neighbor_label = currRowSkip[0]

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    return number_correctly_classified / len(data)


def main():
    print("Welcome to Alex Hsieh's Feature Selection Algortihm.")

    file = input('Type in the name of the file to test: ')
    algo = int(
        input('Type the numbef of the algorithm you wan to run. \n1) Forward Selection\n2) Backward Elimination\n'))

    # Reading file using pandas library
    data = table.read_table(file, delim_whitespace=True, header=None)

    # Returns the dimensions of the table
    print(
        f'\n{file} has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.')

    # Saves to a dictionary in this form [{column -> value}, … , {column -> value}]
    save = data.to_dict('records')

    # Looks for the accuracy
    acc = leave_one_out_cross_validation(
        save, [], [], '1' if algo == 1 else '2')

    print(
        f'\nRunning nearest neighbor with all {data.shape[1] - 1} features, using "leave-one-out" evaluation, I get an accuracy of {acc}%')
    print('\nBeginning search.')
    feature_search(save, algo)


main()
