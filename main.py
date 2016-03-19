# =====================================================================================================================
# AI & Logic Assessment 2
# Emmett Daly | Sam Devlin | Alan Higgins | David Youster
#
# Name: main.py
# Purpose:  Main python file.
#           Takes two files as an input, training data and testing data
#           Both files are parsed and converted into lists
#           The training list is then ran through the neural network
#           The testing list is then ran through the neural network, calculations are performed based on the results of
#           the training set
#           The data is then fed to a flask template to visualise the data
# =====================================================================================================================

import csv
# Adding numpy dependency
import numpy as np
from sklearn import svm
from flask import Flask, render_template

# Import file
training_file_data = "training_data.csv"
testing_file_data = "testing_data.csv"

classifier = svm.SVC()

# ================================================= Flask Set Up ===================================================== #
app = Flask(__name__)


# ================================================= Flask Routes ===================================================== #
@app.route('/')
def default_route():
    # The main function, where the magic happens.
    # Here, we take the input files, parse and run the neural network on both sets.
    # Then we pass both sets to a template, where it is rendered in a simple flask web applciation
    training_set = file_reader(training_file_data)
    testing_set = file_reader(testing_file_data)
    training_expected_outputs = get_expected_outputs(training_file_data)

    get_all_testing_data = get_all_data_of_testing_file(testing_file_data)

    actual_output = []
    actual_output_values = []

    # create a numpy array for training set/expected outputs
    X = np.array(training_set)
    Y = np.array(training_expected_outputs)

    # run fitness function on training set array/expected output array
    classifier.fit(X, Y)

    # for each record in testing set, run prediction function and append output to expected output list
    for record in testing_set:
        test = np.array([record])
        output = classifier.predict(test)
        actual_output.append(output[0])
        for x in output:
            item = {'value': x}
            actual_output_values.append(item)

    glass_type_list = [1, 2, 3, 4, 5, 6, 7]

    return render_template("index.html",
                           glassTypeList=glass_type_list,
                           get_all_testing_data=get_all_testing_data,
                           actual_output=actual_output)


# ============================================= Neural Net Functions ================================================= #
def file_reader(file_path):
    # Here we only want to the chemical values for each entry.
    # We simply iterate over the csv file, extracting all the data
    # We than add the data to a python list and return

    data = []

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Each item identified as not 'id' or not 'Type', we append to the data list
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


# ========================================== Retrieve the testing data =============================================== #
def get_all_data_of_testing_file(file_path):
    # Here we want to get all the data that was tested to provide a comparison of the data against the expected output.
    # We simply iterate over the csv file, extracting all the data
    # We than add the data to a python list
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


# =============================== Get the expected outputs, for comparison only ====================================== #
def get_expected_outputs(file_path):
    data = []

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            data.append(int(row['Type']))

    return data


# ============================================= Run the application ================================================== #
if __name__ == '__main__':
    app.run(debug=True)
