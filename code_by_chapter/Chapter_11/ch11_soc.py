# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:16:08 2021

@author: Lenovo
uses the pymdp module
adapted by tom verguts to model a small social cognition situation
"""


import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from pymdp import utils
from pymdp.agent import Agent

verbose = False

context_names = ['Good', 'Bad']
choice_names = ['Start', 'Stay', 'Sample']

""" Define `num_states` and `num_factors` below """
num_states = [len(context_names), len(choice_names)]
num_factors = len(num_states)

context_action_names = ['Do-nothing']
choice_action_names = ['Start', 'Stay', 'Sample']

""" Define `num_controls` below """
num_controls = [len(context_action_names), len(choice_action_names)]

behavior_obs_names = ['Null', 'good_behavior', 'bad_behavior']
choice_obs_names = ['Start', 'Stay', 'Sample']

""" Define `num_obs` and `num_modalities` below """
num_obs = [len(behavior_obs_names), len(choice_obs_names)]
num_modalities = len(num_obs)

A = utils.obj_array( num_modalities )

prob_good = [0, 1] # what is the probability of being good (element 0) and being bad (element 1)
p_consist_model = 0.8 # consistency of the behavior with the trait according to the model


def entropy(q):
    print(q)
    ent = np.ndarray(num_factors)
    for loop in range(num_factors):
        ent[loop] = -np.dot(q[loop], np.log(q[loop]))
    return ent

A_behavior = np.zeros((len(behavior_obs_names), len(context_names), len(choice_names)))

for choice_id, choice_name in enumerate(choice_names):

  if choice_name == 'Start':

    A_behavior[0,:,choice_id] = 1.0
  
  elif choice_name == 'Stay':

    A_behavior[0,:,choice_id] = 1.0
  
  elif choice_name == 'Sample':

    A_behavior[1:,:,choice_id] = np.array([ [p_consist_model, 1-p_consist_model], 
                                          [1-p_consist_model, p_consist_model]])
  
A[0] = A_behavior

A_choice = np.zeros((len(choice_obs_names), len(context_names), len(choice_names)))

for choice_id in range(len(choice_names)):

  A_choice[choice_id, :, choice_id] = 1.0 # you can observe your actions without ambiguity

A[1] = A_choice


B = utils.obj_array(num_factors)

B_context = np.zeros( (len(context_names), len(context_names), len(context_action_names)) )

B_context[:,:,0] = np.eye(len(context_names))

B[0] = B_context

B_choice = np.zeros( (len(choice_names), len(choice_names), len(choice_action_names)) )

for choice_i in range(len(choice_names)):
  
  B_choice[choice_i, :, choice_i] = 1.0 # you can control your actions without ambiguity

B[1] = B_choice

# prior preferences C over X
C = utils.obj_array_zeros(num_obs) # note: num_obs is a list, which is why we can use obj_array_zeros here
#from pymdp.maths import softmax

C_behavior = np.zeros(len(behavior_obs_names))
C_behavior[1] = +1.0 # good
C_behavior[2] = -0.1 # bad

C[0] = C_behavior  # C[1] can remain all zeros

# prior beliefs D over Z
D = utils.obj_array(num_factors)
D_context = np.array([0.5,0.5])

D[0] = D_context

D_choice = np.zeros(len(choice_names))

D_choice[choice_names.index("Start")] = 1.0

D[1] = D_choice

print('-------------A')
print(A)
print('-------------B')
print(B)
print('-------------C')
print(C)
print('-------------D')
print(D)
print('-------------')
print(f'Beliefs about goodness: {D[0]}')
print(f'Beliefs about starting location: {D[1]}')

my_agent = Agent(A = A, B = B, C = C, D = D)

class Knowthyself(object):

  def __init__(self, context = None, p_consist = 0.8):

    self.context_names = ["Good", "Bad"]

    if context == None:
      self.context = self.context_names[utils.sample(np.array(prob_good))] # randomly sample which trait you have (good or bad)
    else:
      self.context = context

    self.p_consist = p_consist

    self.behavior_obs_names = ['Null', 'good_behavior', 'bad_behavior']


  def step(self, action):

    if action == "Start":
      observed_behavior = "Null"
      observed_choice   = "Start"
    elif action == "Stay":
      observed_behavior = "Null"
      observed_choice   = "Stay" 	
    elif action == "Sample":
      observed_choice = "Sample"
      if self.context == "Good":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, self.p_consist, 1 - self.p_consist]))]
      elif self.context == "Bad":
        observed_behavior = self.behavior_obs_names[utils.sample(np.array([0.0, 1 - self.p_consist, self.p_consist]))]
    
    obs = [observed_behavior, observed_choice]

    return obs

def run_active_inference_loop(my_agent, my_env, T = 5):

  """ Initialize the first observation """
  obs_label = ["Null", "Start"]  # agent observes a `Null` behavior, and seeing itself in the `Start` location
  obs = [behavior_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]
  ent = np.zeros((T, num_factors))
  
  for t in range(T):
    qs = my_agent.infer_states(obs)
    #print("***", qs[0], qs[1])
    ent[t,:] = entropy(qs)
    #print(ent[t,:])
    # if t == T-1:
    #     fig, ax = plt.subplots()
    #     ax.bar([0, 1], qs[0])
    #plot_beliefs(qs[0], title_str = f"Beliefs about the context at time {t}")

    q_pi, efe = my_agent.infer_policies()
    chosen_action_id = my_agent.sample_action()

    movement_id = int(chosen_action_id[1])

    choice_action = choice_action_names[movement_id]

    obs_label = my_env.step(choice_action)

    obs = [behavior_obs_names.index(obs_label[0]), choice_obs_names.index(obs_label[1])]

    print(f'Action at time {t}: {choice_action}')
    print(f'Behavior at time {t}: {obs_label[0]}')
  print("Q = ", qs[0]) 
  return ent

p_consist_process = 0.8 # This is how consistent behavior is with actual character in reality (ie, the generative process)
env = Knowthyself(p_consist = p_consist_process)

T = 10

if verbose:
	print('----------------A')
	print(A)
	print('----------------B')
	print(B)
	print('----------------C')
	print(C)
	print('----------------D')
	print(D)

my_agent = Agent(A = A, B = B, C = C, D = D) # redefine the agent with the new preferences

entr = run_active_inference_loop(my_agent, env, T = T)
fig, ax = plt.subplots()

ax.plot(range(T), entr[:,0], color = "black")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Time")
ax.set_ylabel("Entropy")
ax.set_ylim([0, 0.73])

