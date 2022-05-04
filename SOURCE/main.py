#Import library
import numpy as np #Import for fun :DD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

#Preparing the datasets
def prepare_dataset():
    #Initialize 2 arrays: feature and label
    #Feature is a 2D array to store n games (rows), with each game has 42 attributes (columns)
    # with 3 types: blank (b), player1 (x) and player2 (o)
    #Label is a 1D array to store the result of n games, with 3 types of label: win, loss or draw
    feature = []
    label = []
    #Open the data file
    f = open("connect-4.data", "r")
    lines = f.readlines()
    #Store the first 42 columns of each line into feature array, representing numbers for 3 types:
    # 0 for b (blank), 1 for x (player1), -1 for o (player2)
    #Store the last column (column 43) of each line into label arrary, representing numbers for 3 labels type:
    # 1 for win, -1 for loss and 0 for draw
    for i in lines:
        temp = []
        for j in i.split(','):
            #Append to the feature array through a temp array
            if j != "win\n" and j != "draw\n" and j != "loss\n":
                if j == 'b':
                    temp.append(0)
                elif j == 'x':
                    temp.append(1)
                elif j == 'o':
                    temp.append(-1)
                else: continue
            #Append to the label array
            else:
                if j == "win\n":
                    label.append(1)
                elif j == "draw\n":
                    label.append(0)
                elif j == "loss\n":
                    label.append(-1)
                else: continue
        feature.append(temp)
    return feature, label

#main function
def main():
    #Get the datasets
    feature, label = prepare_dataset()

    #Proportions for the datasets
    #Train proportion
    prop = [4/10, 6/10, 8/10, 9/10]

    #Processing the datasets
    for i in range(4):
        #Building the decision tree classifiers
        #Use train_test_split function to split the dataset into 4 smaller subsets: 
        # feature_train, feature_test, label_train and label_test, where parameter X and Y of the function is 
        # feature and label, with train_size of test_size = (1 - train_size), and turn on stratify, shuffle is True as default
        feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=prop[i], shuffle = True, stratify=label)

        #Use DecisionTreeClassifier function with criterion parameter set to "entropy" in order to get information gain
        # with max_depth is None (we want to classify the whole decision tree)
        #Then we use .fit(X, Y) function to build the tree with X is feature_train and Y is label_train
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=None)
        clf = clf.fit(feature_train, label_train)

        #After the tree is constructed, we use graphviz library to export the tree visualization 
        # using export_graphviz function with color filled, then use the source from the dot_data 
        # to render the tree into a pdf output file
        dot_data = export_graphviz(clf, out_file=None, filled=True)
        graph = graphviz.Source(dot_data)
        graph.render("output_" + str(prop[i]))

        #Evaluating the decision tree classifiers: classification_report
        #Next step, using the .predict function we take the feature_test as an input to predict the result of 
        # the label_test which we have a predict_test array as an output, then compare predict_test with label_test to build
        # a classification report and a confusion matrix using classification_report(y_true, y_pred) 
        # and confusion_matrix(y_true, y_pred) with y_true is label_test and y_pred is predict_test
        predict_test = clf.predict(feature_test)
        print(classification_report(label_test, predict_test))
        print(confusion_matrix(label_test, predict_test))

        #After sketching the classification report and the confusion matrix, we want to calculate
        # the accuracy of the prediction test at max_depth = None, which is the whole decision tree
        # with the respective proportion of train/test subset
        print("\nAccuracy at max_depth = None: ", accuracy_score(label_test, predict_test), "\n")

        #Evaluating the decision tree classifiers: confusion_matrix
        #We also want to visualize the confusion matrix using ConfusionMatrixDisplay from matplotlib, using the 
        # previous predictions (predict_test) to display the visualization as a png file using .show function
        display = ConfusionMatrixDisplay.from_predictions(label_test, predict_test)
        display.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{display.confusion_matrix}")

        #Visualize the confusion matrix using matplotlib
        plt.show()

        #Used for 8/2 train/test
        #The depth and accuracy of a decision tree
        #We need to also visualize the decision tree at max_depth = [2,3,4,5,6,7] with
        # the proportion of train/test is 8/2
        if i == 2:
            #Run a for loop for max_depth from 2 to 7
            for j in range(2, 8):
                #Using DecisionTreeClassifier with max_depth parameter from 2 to 7
                clf = DecisionTreeClassifier(criterion="entropy", max_depth=j)
                clf = clf.fit(feature_train, label_train)

                #Export the decision tree graph visualization using graphviz
                #Use the .Source function from dot_data as input argument and then render to graph
                dot_data = export_graphviz(clf, out_file=None, filled=True)
                graph = graphviz.Source(dot_data)
                graph.render("output_" + str(prop[i]) + "-max_depth_" + str(j))

                #We also predict the test subset at the respective max_depth and calculate the accuracy
                # with the respective max_depth
                predict_test = clf.predict(feature_test)
                print("\nAccuracy at max_depth = ", j, " : ", accuracy_score(label_test, predict_test), "\n")

#Call the main function
main()