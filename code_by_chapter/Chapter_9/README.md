
# Markov decision process approaches to RL

As for chapter 8, you will need to install Open AI Gym for some scripts (pip install gym).
The code in this repo uses "pure" MDP (tabular) approaches to RL.
Some use dynamic programming (all code with "optimal" in its name), others use Rescorla-Wagner, Q-learning, Sarsa, or Sarsa-lambda (e.g., frozen_lake, ch9_RL_Taxi). Mountaincar has a continuous input space, but this input space is discretized in ch9_RL_mountaincar in order to apply tabular RL. mountaincar_cont in addition has a continuous output space, which is (also) discretised in ch9_RL_mountaincar_cont.

Ch9_RL_Taxi_2 uses a tabular actor-critic agent, which may be argued to belong to chapter 8... The boundaries can be fuzzy.