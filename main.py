import pandas as table
import copy


def feature_search(data, alg):

    current_set_of_features = []
    acc = 0
    isMax = False
    s = None

    # Accounts for the levels of the tree
    for i in range(1, len(data[0])):
        print('On the ', i, ' th level of the search tree')
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        # Accounts for each feature of the data set
        for k in range(1, len(data[0])):
            # Don't repeat for the same feature
            if k not in current_set_of_features:
                print(' -- Considering adding the ', k, ' feature')

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
        ignoreColumns = list(range(1, len(data[0])))
        for i in ignoreColumns:
            if i not in a:
                ignoreColumns.append(i)
    # If backwards, give it current data set as you're going to remove
    else:
        ignoreColumns = a

    for i in range(1, len(data[0])):
        object_to_classify = data(i, 2: end)
        label_object_to_classify = data(i, 1)
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        nearest_neighbor_label = float('inf')

        for k in range(1, len(data[0])):
            # skips itself from classification
            if k == i:
                continue

            # figure out distance function
            distance = sqrt(sum((object_to_classify - data(k, 2: end)). ^ 2))

            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = k
                nearest_neighbor_label = data(nearest_neighbor_location, 1)

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    return number_correctly_classified / len(data[0])


def main():
    print("Welcome to Alex Hsieh's Feature Selection Algortihm.")

    file = input('Type in the name of the file to test: ')
    algo = int(
        input('Type the numbef of the algorithm you wan to run. \n1) Forward Selection\n2) Backward Elimination'))

    # Reading file using pandas library
    data = table.read_table(file, delim_whitespace=True, header=None)

    # Returns the dimensions of the table
    print(
        f'\n{file} has {data.shape[1] - 1} features (not including the class attribute), with {data.shape[0]} instances.')

    # Saves to a dictionary in this form [{column -> value}, … , {column -> value}]
    save = data.to_dict('records')

    # Looks for the accuracy
    acc = leave_one_out_cross_validation(
        save, [i for i in range(1, len(save[0]) - 1)], len(save[0]), '1')

    print(
        f'\nRunning nearest neighbor with all {data.shape[1] - 1} features, using "leave-one-out" evaluation, I get an accuracy of {acc * 100}%')
    print('\nBeginning search.')
    feature_search(save, algo)


main()
