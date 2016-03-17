import csv
# Adding numpy dependency
import numpy as np
from sklearn import svm

# Import file
training_file_data = "training_data.csv"
testing_file_data = "testing_data.csv"

classifier = svm.SVC()


def file_reader(file_path):
    # The neural net module we have requires two inputs:
    # a ist of training data lists as input on
    # a list of expected outputs, ie the 'Type' as identified

    data=[]

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
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
            data.append(temp)

    return data

def get_expected_outputs(file_path):
    
    data=[]

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            data.append(int(row['Type']))

    return data

def main():
    # get training/testing sets and expected outputs from file
    training_set = file_reader(training_file_data)
    testing_set = file_reader(testing_file_data)
    training_expected_outputs = get_expected_outputs(training_file_data)
    testing_expected_outputs = get_expected_outputs(testing_file_data)
    actual_output = []

    # create a numpy array for training set/expected outputs
    X = np.array(training_set)
    Y = np.array(training_expected_outputs)

    # run fitness function on training set array/expected output array
    classifier.fit(X,Y)

    # for each record in testing set, run prediction function and append output to expected output list
    for record in testing_set:
        test = np.array([record])
        output = classifier.predict(test)
        actual_output.append(output[0])

    # print output
    print('predicted outputs:')
    print(testing_expected_outputs)
    print('actual outputs:')
    print(actual_output)

    pass


if __name__ == '__main__':
    main()
