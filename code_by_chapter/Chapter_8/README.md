
# Gradient-based approaches to RL

Several of the scripts in chapters 8 and 9 folders (attempt to) solve RL problems from the Open AI gym environment. You can simply install gym using pip (pip install gym).

Some code in this chapter is direct application of chapter 8 material. Bandit scripts implement REINFORCE via explicitly defined prediction errors.
ch8_tf2_taxi, ch8_tf2_lunar, and ch8_tf2_mountaincar_cont implement a policy gradient algorithm via Keras (also basically REINFORCE, just implicitly defined via its goal function).

Instead, some code uses principles in between chapter 8 and 9. In particular, the deep-Q (DQN) models for polecart (ch8_tf2_pole_x), lunarlander (ch8_tf2_lunar_2), ch8_tf2_taxi_2, and ch8_tf2_mountaincar problems combine aspects of chapter 8 and 9. They are value-based (as in chapter 9), but the values are computed based on gradients (as in chapter 8).

The script ch8_tf2_tf2_lunar_2 implements actor-critic; this is standardly considered as policy gradient (as in chapter 8), but arguably also contains a value-based perspective (as in chapter 9; see Bennett et al., 2021).

