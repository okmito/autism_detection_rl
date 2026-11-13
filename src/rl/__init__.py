"""Adaptive screening as a sequential decision problem (ASK one item, or STOP).

Modules
-------
belief    calibrated naive-Bayes belief model over partial Q-CHAT-10 histories
mdp       state codes, legal actions, reward definitions, batch replay environment
planner   model-based RL: exact value iteration on the learned belief MDP
dqn       model-free RL: Double DQN trained on replayed real response histories
baselines fixed-length, threshold-stopping and adapters for the existing policies
evaluate  run any agent over held-out records and compute screening/burden metrics
explain   per-decision ASK/STOP explanations derived from the acting agent's own values
served    the artifact the application loads, with its integration gate

Research prototype. Nothing here diagnoses autism.
"""
