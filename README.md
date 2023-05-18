# second_year_project

The rnn model can be run by in the terminal. 
To run it make sure you are in the current working directory (baseline/rnn). 
Then the path to the training data, test_data and output file should be specified as command line arguments such as:

python rnn.py ../../datasets/entity_swapped_datasets/science_randomly_replaced.txt ../../datasets/music_test.txt outfile.txt


To evaulate of the model can be done by setting the current working directory to the predictions folder. 
And providing the path to the test data (gold labels) and the predictions as:

python evaluate.py ../datasets/music_test.txt ../predictions/randomly_music_test_preds.txt
