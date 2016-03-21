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
from sklearn.metrics import mean_squared_error
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
    testing_expected_outputs = get_expected_outputs(testing_file_data)
    all_ids = get_expected_id(testing_file_data)
    get_all_testing_data = get_all_data_of_testing_file(testing_file_data)

    actual_output = []
    successful_hits = 0;

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

    # Calculate the number of successful hits after training
    for x,y in zip(actual_output,testing_expected_outputs):
        if x == y:
            successful_hits +=1

    # Calculate the precentage
    percentage_of_hits = float((float(successful_hits)/float(len(testing_expected_outputs))*100))

    # Calculate the mean squared error, how incorrect the outputs are
    mean_square_error = calculate_mean_square_error(testing_expected_outputs, actual_output)

    return render_template("index.html", 
        get_all_testing_data = get_all_testing_data,
        actual_output =actual_output,
        all_ids=all_ids,
        mean_square_error =mean_square_error,
        percentage_of_hits=percentage_of_hits,
        testing_expected_outputs=testing_expected_outputs)

# ============================================= Neural Net Functions ================================================= #
def file_reader(file_path):
    # Here we only want to the chemical values for each entry.
    # We simply iterate over the csv file, extracting all the data
    # We than add the data to a python list and return

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
    
    data=[]

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            data.append(int(row['Type']))

    return data



# ======================================== Get the id of the training dataset ======================================== #
def get_expected_id(file_path):
    # Here we just want to get the id for each entry to append to the graph within the index.html file
    
    data=[]

    with open(file_path) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            data.append(int(row['id']))

    return data



# ======================================= Calculate the mean square error ============================================= #
def calculate_mean_square_error(expected_output, actual_output):

    return mean_squared_error(actual_output,expected_output)



# ============================================= Run the application ================================================== #
if __name__ == '__main__':
    # main()
    app.run(debug=True)