
# gradient-based approaches to RL

Some code in this chapter (bandit, taxi) is direct application of chapter 8 material. Bandit implements REINFORCE prediction errors; ch8_RL_taxi implements a policy gradient algorithm via Keras (which is also basically REINFORCE, just implicitly defined via its goal function).

Instead, some code uses principles in between chapter 8 and 9. In particular, the deep-Q (DQN) models for polecart (pole), lunarlander (lunar), taxi_2, and mountaincar problems combine aspects of chapter 8 and 9. They are value-based (as in chapter 9), but the values are computed based on gradients (as in chapter 8).

