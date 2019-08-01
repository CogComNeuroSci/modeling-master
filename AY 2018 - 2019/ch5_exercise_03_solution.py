#!/usr/bin/python3
# -*- coding: utf-8 -*-


'''
@author: Pieter
Pieter.Huycke@UGent.be

- - - - - - - - - - - - 

Stuck using a function?
No idea about the arguments that should be defined?

Type:
help(function_name)
to let Python help you!
'''

import matplotlib.pyplot          as plt
import numpy                      as np

from   sklearn.metrics            import accuracy_score
from   sklearn.model_selection    import train_test_split
from   sklearn.neural_network     import MLPClassifier


#%%
'''
*The biased model*

In this code, we are going to recreate the Stroop model.
Here, the idea is that a human is biased towards reading words.
Because of this, we have to this kind of bias.
To do this, we have to present the 'word naming task' more than the 
'color naming task'
This approach is similar to previous work where we presented more dogs than 
cats when training our model

To keep track of our model, we will first have to assign interpretations to
our unit activations (like we did in the previous exercise).
Then, we can define what each pattern represents, and what response is 
appropriate for each pattern.
Remember to learn the 'word task' more to install the bias towards reading

Then, we want to train our model with the data, which leads to a model that 
is able to complete the task, but has a bias towards reading the words
vs naming the color
'''

# Unit interpretation:
    # units 1 and 2 represent the cue that is given (this one is fixed)
        # [0, 1] = 'color' is relevant dimension
        # [1, 0] = 'word' is relevant dimension
    # units 3 and 4 represent the color of the word:
        # [1, 0] = green
        # [0, 1] = red
    # units 5 and 6 represent the written format:
        # [1, 0] = green
        # [0, 1] = red
    # response units:
        #  [1] if the response should be 'red'
        # [-1] if the response should be 'green'

# task: 'name the color of the presented word'
red_red_color     = [0, 1, 0, 1, 0, 1] # Answer should be: 'red'
green_green_color = [0, 1, 1, 0, 1, 0] # Answer should be: 'green'
red_green_color   = [0, 1, 0, 1, 1, 0] # Answer should be: 'red'
green_red_color   = [0, 1, 1, 0, 0, 1] # Answer should be: 'green'

# task: 'read the presented word'
red_red_word      = [1, 0, 0, 1, 0, 1] # Answer should be: 'red'
green_green_word  = [1, 0, 1, 0, 1, 0] # Answer should be: 'green'
red_green_word    = [1, 0, 0, 1, 1, 0] # Answer should be: 'green'
green_red_word    = [1, 0, 1, 0, 0, 1] # Answer should be: 'red'

# appropriate responses for each case
response_1, response_2, response_3, response_4 =  [1], [-1],  [1], [-1]
response_5, response_6, response_7, response_8 =  [1],  [1], [-1],  [1]

# bundle them together
color_inputs  = np.array([red_red_color, green_green_color, 
                          red_green_color, green_red_color])
word_inputs   = np.array([red_red_word,  green_green_word,  
                          red_green_word,  green_red_word])

# make a copy of the original arrays, because I believe you might want to use 
# those later on...
color_inputs_copy, word_inputs_copy = np.copy(color_inputs), \
                                      np.copy(word_inputs)

# define the appropriate responses for the 'name the color' task
color_outputs = np.array([response_1, response_2, response_3, response_4])

# define the appropriate responses for the 'read the word' task
word_outputs  = np.array([response_5, response_6, response_7, response_8])

# again, we make a copy of the original arrays
color_outputs_copy, word_outputs_copy = np.copy(color_outputs), \
                                        np.copy(word_outputs)

# amount of possible inputs (the same for the word task of course)
length_inputs = len(color_inputs)

# repeat the input patterns (repeat the word task more, for the bias)
# we defined the color inputs 2 times, and the word inputs 7 times
color_inputs   = np.tile(color_inputs, (1 * 2, 1))
word_inputs    = np.tile(word_inputs,  (1 * 7, 1))

# repeat the responses the same amount of times: our responses should match
# our defined input patterns
color_outputs  = np.tile(color_outputs, (1 * 2, 1))
word_outputs   = np.tile(word_outputs,  (1 * 7, 1))

# stack the input patterns and their associated responses
inputted_patterns  = np.vstack((color_inputs, word_inputs))
outputs            = np.ravel(np.vstack((color_outputs, word_outputs)))

#%%
'''
* Fitting the model *

In this part, we use a multilayer perceptron to train our model
Similar to the previous exercise, we will use a hidden layer with 4 hidden 
units
Our learning rate, the random_state, ... and other parameters remain constant
'''

# define MultiLayerPerceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), 
                    max_iter=500,
                    solver='sgd', 
                    verbose=0,
                    random_state=1234,
                    learning_rate_init=.3, 
                    activation='logistic')

# train the model based on the data
mlp.fit(inputted_patterns, outputs)

#%%
'''
* Testing on the congruent trials *

Now, we will use our trained model to perform the Stroop task, but only for the
congruent trials. 

In this case, a congruent trial is when the ink color is the 
same as the word (i.e., green written in green ink) and an incongruent trial is
when the ink color is different to the word meaning 
(i.e., green written in red ink)
'''

color_inputs_congr  = np.tile(color_inputs_copy[0:2], (1 * 10, 1))
words_inputs_congr  = np.tile(word_inputs_copy[0:2], (1 * 10, 1))

color_outputs_congr = np.tile(color_outputs_copy[0:2], (1 * 10, 1))
word_outputs_congr  = np.tile(word_outputs_copy[0:2], (1 * 10, 1))

congruent_input     = np.vstack((color_inputs_congr, words_inputs_congr))
congruent_output    = np.ravel(np.vstack((color_outputs_congr, word_outputs_congr)))

# predict y based on x for the test data
y_pred = mlp.predict(congruent_input)

# print accuracy using dedicated function
print('Accuracy percentage in the congruent trials: {0:.2f}%'.format(accuracy_score(congruent_output, y_pred) * 100))

#%%
'''
* Testing on the incongruent trials *

Now, we will use our trained model to perform the Stroop task, but only for the
incongruent trials. 

The same definitions with respect to (in)congruency hold
'''

color_inputs_incongr  = np.tile(color_inputs_copy[2:4], (1 * 10, 1))
words_inputs_incongr  = np.tile(word_inputs_copy[2:4], (1 * 10, 1))

color_outputs_incongr = np.tile(color_outputs_copy[2:4], (1 * 10, 1))
word_outputs_incongr  = np.tile(word_outputs_copy[2:4], (1 * 10, 1))

incongruent_input     = np.vstack((color_inputs_incongr, words_inputs_incongr))
incongruent_output    = np.ravel(np.vstack((color_outputs_incongr, word_outputs_incongr)))

# predict y based on x for the test data
y_pred = mlp.predict(incongruent_input)

# print accuracy using dedicated function
print('Accuracy percentage in the incongruent trials: {0:.2f}%'.format(accuracy_score(incongruent_output, y_pred) * 100))

#%%
'''
* Test entire model performance (all data available) *

Retrain exactly the same model on the same data
Split the data in a train part and a test part.
Keep every parameter exactly the same for the MLP model, train it on the 
part of the data reserved for training, and test it on the part reserved for
testing
Check the accuracy of your trained model using the earlier defined code
'''

# split data in training and testing set
X_train, X_test, y_train, y_test = train_test_split(inputted_patterns, 
                                                    outputs)


# define MultiLayerPerceptron
mlp = MLPClassifier(hidden_layer_sizes=(4,), 
                    max_iter=500,
                    solver='sgd', 
                    verbose=0,
                    random_state=1234,
                    learning_rate_init=.7, 
                    activation='logistic')

# fit ('train') MLP classifier to the training data
mlp.fit(X_train, y_train)

# predict y based on x for the test data
y_pred = mlp.predict(X_test)

# print accuracy using dedicated function
print('Accuracy percentage for the entire dataset: {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))

#%%
'''
* Add a lesion to the model *

Here, we will simulate the performance on this very Stroop task of a subject 
that suffers from schizophrenia.
This is in line with the seminal paper of Cohen and Servan-Schreiber (1992).
While a thorough review of this paper is beyond the scope of this comment block,
we stress that this paper aimed to model the decreased performance of 
schizophrenia patients in the Stroop task. 
For the sake of parsimony, it should suffice to know that making the model 
perform worse is an approximation of a subject suffering from schizophrenia.
This because these patients also perform worse than healthy subjects.
(i.e. the distorted model also performs worse than the healthy model)

In this final part of this code we will also add a lesion to our trained model.
In other words, we will make sure that our model performs worse by adding 
random (normally distributed) noise to the weights from our input layer to the
hidden layer of our trained model.
Obviously, our model will perform worse when the weights are distorted, as
our weight values after learning represent that the model will be able to 
output an appropriate response for a certain input.
Example: due to our configured weights, the model will respond with 'red', when
the task is 'respond with the color' to the word GREEN written in a red color.
After training, our model represents a human that is able to do this task 
without any errors. Thus, our subject makes no errors when performing the 
Stroop task.
This degradation in model performance reflects how a schizophrenic person would
perform on this task, as we highlighted earlier.

In each step, we will add more noise (reflecting a patient that has more issues
in processing).
Investigate how the performance of the model evolves with increasing noise.

Reference:
Cohen, J. D., & Servan-Schreiber, D. (1992). Context, cortex, and dopamine: 
    a connectionist approach to behavior and biology in schizophrenia. 
    Psychological review, 99(1), 45.
'''

# suppress scientific notation
np.printoptions(suppress=True)

# copy the original weights to a new variable
original_weights_to_hidden = np.copy(mlp.coefs_[0])

# create an array with values in the interval [0.01, 1], equally spaced
standard_deviations        = np.linspace(1, 5)

# number of simulations we want to run
simulation_number          = 50

# create an array of zeros that will contain the accuracy scores
accuracy_general           = np.zeros(standard_deviations.shape)
accuracy_congruent         = np.zeros(standard_deviations.shape)
accuracy_incongruent       = np.zeros(standard_deviations.shape)

# create an array that will contain all the accuracy scores for all simulations
    # this can be of use when we want to plot per simulations after running the
    # simulation
simulations_general        = np.zeros((standard_deviations.shape[0], simulation_number))
simulations_congruency     = np.zeros((standard_deviations.shape[0], simulation_number))
simulations_incongruency   = np.zeros((standard_deviations.shape[0], simulation_number))

# loop for n amount of cycles, where n = simulation_number
for simulation in range(simulation_number): 

    cycle_number = 0
    random_array = np.random.randn(6, 4)
    
    # loop over different standard deviations
    for std in standard_deviations:
            
            # add random noise to the weights that connect the input layer and 
            # the hidden layer
            # mind that the standard deviation has a huge influence in this:
                # multiplying by 100 has huge effects, while mulltiplying by 1 
                # yields no differences at all when it comes to accuracy
            mlp.coefs_[0] = mlp.coefs_[0] + (random_array * std)
            
            # *****
            # Accuracy on all data
            # *****

            # make a new prediction with the model, keeping in mind that the 
            # weights have been altered by the random noise
            y_pred = mlp.predict(inputted_patterns)
            
            # print the result: how is the accuracy of the model affected by
            # the change in weights?
            # (commented out for now, but please uncomment it to see how it works)
            '''
            print('Simulation {0:2d} - Cycle {1:2d}: Accuracy = {2:.2f}%'.format(simulation + 1,
                                                                                 cycle_number + 1,
                                                                                 accuracy_score(outputs, y_pred) * 100))
            '''
            # assign the computed accuracy score to the array that keeps track
            # of this
            accuracy_general[cycle_number] = accuracy_score(outputs, y_pred) * 100
            
            # *****
            # Accuracy on congruent trials
            # *****
            y_pred = mlp.predict(congruent_input)
            accuracy_congruent[cycle_number] = accuracy_score(congruent_output, y_pred) * 100
            
            # *****
            # Accuracy on incongruent trials
            # *****
            y_pred = mlp.predict(incongruent_input)
            accuracy_incongruent[cycle_number] = accuracy_score(incongruent_output, y_pred) * 100        
            
            # reset the altered weights to their original value
            mlp.coefs_[0] = original_weights_to_hidden
            
            # increment the variable 'cycle_number' with 1
            cycle_number += 1
    
    # when the previous loop is done, we have a filled 'accuracy_score' array
    # we now assign the entire array to one of the all-zero arrays in 
    # 'simulations_scores'
    # at the end, we will have replaced all n all-zero arrays in this variable
    # these arrays reflect the fluctuations in accuracy scores depending on the
    # standard deviations, with n different (random) starting distortions 
    # in the weights
    # by doing so, we can assess how often the decrease in model accuracy occurs
    simulations_general[simulation]      = accuracy_general
    simulations_congruency[simulation]   = accuracy_congruent
    simulations_incongruency[simulation] = accuracy_incongruent
    
    # rest 'accuracy_scores' = make it all-zero again
    # if this is not done, we will use a non-zero array in the next 
    # 'standard_deviation' loop
    accuracy_general           = np.zeros(standard_deviations.shape)
    accuracy_congruent         = np.zeros(standard_deviations.shape)
    accuracy_incongruent       = np.zeros(standard_deviations.shape)

#%%
'''
* Plot me what you got *

general accuracy 

In this part, we will plot how the performance of the model decreases with 
heavier distortions of the weights between the input layer and the hidden layer
We will use data that comes from the previous part, where we simulated the
decrease in performance n times
'''

# compute both the median- and the mean accuracy of the altered model for each
# noise-adding cycle
mean_accuracy = np.mean(simulations_general, axis = 0)
median_accuracy = np.median(simulations_general, axis = 0)

# plot both lines separately
# basically, we plot all the values in each specific array
# we also assign a color to the line, and label the line (this will be used 
# by the legend() function we call immediately after this))
median_acc, = plt.plot(median_accuracy, color = 'royalblue', label='Median accuracy')
mean_acc,   = plt.plot(mean_accuracy, color = 'forestgreen', label='Mean accuracy')

# create a legend, Python knows the colors and the labels due to the previous
# piece of code, the legend will be plotted in the 'best' location
plt.legend(loc='best')

# some definitions that impact the looks of the y-axis
plt.ylabel('Accuracy of the model')
plt.ylim(0, 105)
plt.yticks(np.arange(0, 105, step=10))

# a definition that impact the looks of the y-axis
plt.xlabel('Number of loops (more noise added / loop)')

# a definition that creates a title for our plot
plt.title('The impact of distorted weights on model accuracy')

# show me the plot
plt.show()

# let the plot close when clicking on it
plt.waitforbuttonpress(0)
plt.close()


#%%
'''
* Plot me what you got *

congruent vs. incongruent

In this part, we will investigate how the performance of the model changes
both on the congruent and the incongruent trials
Plot the mean accuracy for both types of trials, and look at what we observe
'''

# compute mean accuracy on the congruent and incongruent trials for each
# noise-adding cycle
mean_congruent = np.mean(simulations_congruency, axis = 0)
mean_incongruent = np.mean(simulations_incongruency, axis = 0)

# plot both lines separately
# basically, we plot all the values in each specific array
# we also assign a color to the line, and label the line (this will be used 
# by the legend() function we call immediately after this))
mean_congr,     = plt.plot(mean_congruent, color = 'Indigo', label='Congruent trials')
mean_incongr,   = plt.plot(mean_incongruent, color = 'sandybrown', label='Incongruent trials')

# create a legend, Python knows the colors and the labels due to the previous
# piece of code, the legend will be plotted in the 'best' location
plt.legend(loc='best')

# some definitions that impact the looks of the y-axis
plt.ylabel('Accuracy of the model')
plt.ylim(0, 105)
plt.yticks(np.arange(0, 105, step=10))

# a definition that impact the looks of the y-axis
plt.xlabel('Number of loops (more noise added / loop)')

# a definition that creates a title for our plot
plt.title('The impact of distorted weights on model accuracy\n' \
          'Distinction between congruent and incongruent trials')

# show me the plot
plt.show()

# let the plot close when clicking on it
plt.waitforbuttonpress(0)
plt.close()
