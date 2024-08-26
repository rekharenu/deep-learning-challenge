# deep-learning-challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organisation, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organisation classification
USE_CASE—Use case for funding
ORGANIZATION—Organisation type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

Instructions
Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organisation will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Step 3: Optimise the Model
Using your knowledge of TensorFlow, optimise your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimise your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimising your model, you will not lose points if your model does not achieve target performance.

Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimisation.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimising the model.

Design a neural network model, and be sure to adjust for modifications that will optimise the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimisation.h5.

Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
   Targey Variable: The target variable for the model was identified as the column IS_Successful.

What variable(s) are the features for your model?
    Feature Variable:The following columns were used as features for the model:
    1 NAME
    2 APPLICATION_TYPE
    3 AFFILIATION
    4 CLASSIFICATION
    5 USE_CASE
    6 ORGANIZATION
    7 STATUS
    8 INCOME_AMT
    9 SPECIAL_CONSIDERATIONS
    10 ASK_AMT


What variable(s) should be removed from the input data because they are neither targets nor features?
     Variable to Remove:The EIN column was removed,as it serves only as an identifier for the application prganization and does not affect the models behaviour


Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
     Model Architecture:The optimization model used 3 hidden layers with multiple neurons,which increased the accuracy from under 72% to 78%. The initial model has 2 layers,epochs are same ,but when we add the new layers the accuracy will improve.
     
Were you able to achieve the target model performance?
     Target Performance:Yes ,by optimizing the madel the accuracy increases to 78%.
What steps did you take in your attempts to increase model performance?
    1 Steps to Increae Performance:
         **Instead of drooping both the EIN and NAME columns,only the EIN colums was dropped . However,only names that appeared more than 5 times were considered.
         **A third activation layer was added to the model in the following order to boost accuracy to over 72%:
               1: ReLU layer
               2: Tanh layer
               3: Sigmoid layer
Summary: Summarise the overa


     overall results of the deep learning model 77% .This means the test data has improved of time . 
     The applicant name appears more than 5 times (indicating they have applied
      more than 5 times).
     The application type is one of the following: T3, T4, T5, T6, or T19.
     The application has one of the following classification values:   C1000,       C1200,
C2000, C2100, or C3000.Include a recommendation for how a different model could solve this classification problem, and then explain your recommendat

Alternative method

We can use the Random Forest also achive accuray close .



