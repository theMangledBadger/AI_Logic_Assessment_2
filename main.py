import csv
import numpy

# Import file
training_file_data = "training_data.csv"


def training_data_file_reader(file_path):
    # The neural net module we have requires two inputs:
    # a ist of training data lists as input on
    # a list of expected outputs, ie the 'Type' as identified

    # We have two lists we need to store:
    training_list = []
    glass_type_list = []

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Each item identified as 'Type', we append the value to the list
            glass_type_list.append([int(row['Type'])])
            # Each item identified as not 'id' or not 'Type', we append to the inputs list
            temp = [float(row['RI']),
                    float(row['Na']),
                    float(row['Mg']),
                    float(row['Al']),
                    float(row['Si']),
                    float(row['K']),
                    float(row['Ca']),
                    float(row['Ba']),
                    float(row['Fe'])]
            training_list.append(temp)

    # At the moment, we are just printing the data to the screen in two lists:
    print(training_list)
    print(glass_type_list)
    return training_list


if __name__ == '__main__':
    training_data_file_reader(training_file_data)
