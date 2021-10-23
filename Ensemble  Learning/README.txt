This is a folder for Ensemble Learning and Adaboost/Bagged/Random Forest implementations: main_EL to run this class

I do apoligize for any confusion regarding my main method or helper methods.  I have used a DecisionTree class import along with methods from the DecisionTree folder in order to alter specifically for weighted examples and or random forests.

Here are some roadmaps to follow if you wish to run my code:

Firstly, the main method will have the csv read in files for train and test data.  I do reset these often which could be bad practice rather than using copy however it is there for this assignment submission.

In my main_EL.py file, I have multiple methods ranging from new adaboost and bagged to adjusted ID3 in the form of Random_forest.


Similar to my Linear Regression folder, I do have my functions commented out on the main method and if you wish to run, please uncomment one at a time.


First : Adaboost.  Please uncomment out the call to adaboost and the two following print statements which will print figures

Second : Bagged.  Please uncomment out the call to bagged and the following print statements which will print figures

Third : Random Forest.  This will be the default call.  Please comment out to run other methods.  NOTE: for this implementation, I change the random forest manually         in my 'random_forest' method.  This is done by selecting the random number amount then adding those located attributes into a new array and passing that             into find the best splits.  Sorry for the inconvinience.  It is currently set at '6'


