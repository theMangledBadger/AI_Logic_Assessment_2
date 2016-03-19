import csv
# Adding numpy dependency
import numpy as np
from sklearn import svm
from flask import Flask, render_template

# Import file
training_file_data = "training_data.csv"
testing_file_data = "testing_data.csv"

classifier = svm.SVC()

#=== Flask Set Up ===#
app = Flask(__name__)

#=== Flask Routes ===#
@app.route('/')
def default_route():
    # Ok, need to get al the testing data, including the ID
    # Store this and pass it into the template
    # Get the actual (calculated) outputs
    # Also pass this
    #

    training_set = file_reader(training_file_data)
    testing_set = file_reader(testing_file_data)
    training_expected_outputs = get_expected_outputs(training_file_data)
    testing_expected_outputs = get_expected_outputs(testing_file_data)

    get_all_testing_data = get_all_data_of_testing_file(testing_file_data)

    actual_output = []
    actual_output_values = []

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
        for x in output:
            item = {}
            item['value'] = x
            actual_output_values.append(item)

    # print output
    print('predicted outputs:')
    print(testing_expected_outputs)
    print('actual outputs:')
    print(actual_output)

    glassTypeList = [1,2,3,4,5,6,7]


    return render_template("index.html", 
        glassTypeList = glassTypeList,
        get_all_testing_data = get_all_testing_data,
        actual_output =actual_output)

#=== Neural Net Functions ===#
def file_reader(file_path):
    # The neural net module we have requires two inputs:
    # a ist of training data lists as input on
    # a list of expected outputs, ie the 'Type' as identified

    data=[]

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Each item identified as not 'id' or not 'Type', we append to the inputs list
            temp = [
                    float(row['RI']),
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

def get_all_data_of_testing_file(file_path):

    data = []

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            temp = [
                int(row['id']),
                float(row['RI']),
                float(row['Na']),
                float(row['Mg']),
                float(row['Al']),
                float(row['Si']),
                float(row['K']),
                float(row['Ca']),
                float(row['Ba']),
                float(row['Fe']),
                int(row['Type'])]
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
    # main()
    app.run(debug=True)