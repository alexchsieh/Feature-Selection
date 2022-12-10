import pandas as table


def feature_search(data, alg):

    current_set_of_features = []

    for i in range(1, len(data[0])):
        print('On the ', i, ' th level of the search tree')
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0

        for k in range(1, len(data[0])):
            if k not in current_set_of_features:
                print(' -- Considering adding the ', k, ' feature')
                accuracy = leave_one_out_cross_validation(
                    data, current_set_of_features, k, alg)

                print(f'---Using features {current_set_of_features} and adding {k} has an accuracy of {round(accuracy * 100, 1)}%') if alg == 1 else print(
                    f'---Removing features {current_set_of_features} and also {k} has an accuracy of {round(accuracy * 100, 1)}%')

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        current_set_of_features.append(feature_to_add_at_this_level)
        print('On level ', i, ' i added feature ',
              feature_to_add_at_this_level, ' to current set')


def leave_one_out_cross_validation(data, current_set_of_features, add_feature, alg):
    number_correctly_classified = 0

    for i in range(1, len(data[0])):
        object_to_classify = data(i, 2: end)
        label_object_to_classify = data(i, 1)
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        for k in range(1, len(data[0])):
            if k != i:
                distance = sqrt(sum((object_to_classify - data(k, 2: end)). ^ 2))
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor_location = k
                nearest_neighbor_label = data(nearest_neighbor_location, 1)

        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified = number_correctly_classified + 1

    accuracy = number_correctly_classified / len(data[0])


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
