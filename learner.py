import random
import numpy as np
import time

class Q_Learner:

    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ######
    ###### Class variables / Constructor ######
    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ######
    
    # These will have to be passed when the learner is created
    total_states = None
    total_actions = None
    Q_table = None

    # These are default values, but can be overwritten
    learning_rate = 0.2
    rewards_rate = 0.9
    exploration_rate = 0.5
    exploration_rate_decay = 0.99

    # These are training values used to hold the previous state and action
    train_state = None
    train_action = None

    def __init__(
        self,
        inp_total_states, # positive int - total number of states the learner could be in
        inp_total_actions, # positive int - total number of actions the learner can take in each state
        inp_learning_rate_OPT=None, # see update_learner_preferences()
        inp_rewards_rate_OPT=None, # see update_learner_preferences()
        inp_exploration_rate_OPT=None, # see update_learner_preferences()
        inp_exploration_rate_decay_OPT=None # see update_learner_preferences()
    ):
        
        # Set required variables for states and actions
        self.total_states = inp_total_states
        self.total_actions = inp_total_actions

        # Create the initial Q-table, start with all zeros
        self.Q_table = np.zeros((self.total_states, self.total_actions), dtype=float)

        # Update optional varaibles for learning, reward, and random action rates
        self.set_learner_preferences(inp_learning_rate_OPT, inp_rewards_rate_OPT, inp_exploration_rate_OPT, inp_exploration_rate_decay_OPT)
        
        # Set random seed based on current time
        random.seed(time.time() * 1000)

        return

    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ######
    ###### Accessor and Mutator Functions ######
    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ######

    # Mutator function that updates the mechanical weights of the Q-Learner *cannot update # states or # actions*
    def set_learner_preferences(
        self,
        inp_learning_rate_OPT=None, # Between 0 and 1 - how quickly the Q-table updates with information it observes
        inp_rewards_rate_OPT=None, # Between 0 and 1 - how much future calculated rewards are worth to the Q-table
        inp_exploration_rate_OPT=None, # Between 0 and 1 - how often the learner takes random actions when starting out
        inp_exploration_rate_decay_OPT=None # Between 0 and 1 - the quickly the learner stops taking random actions as it learns
    ):
        # Update optional class variables
        for arg_name, arg_value in locals().items():
            if arg_name.endswith('OPT') and arg_value is not None:
                setattr(self, '_'.join(arg_name.split('_')[1:-1]), arg_value)

    # Accessor function to get any of mechanical weights of the Q-learner
    def get_learner_preferences(
        self,
        inp_total_states_OPT=None,
        inp_total_actions_OPT=None,
        inp_learning_rate_OPT=None,
        inp_rewards_rate_OPT=None,
        inp_exploration_rate_OPT=None,
        inp_exploration_rate_decay_OPT=None
    ):
        results = []
        for arg_name, arg_value in locals().items():
            if arg_name.endswith('OPT') and arg_value is not None:
                results.append(getattr(self, '_'.join(arg_name.split('_')[1:-1])))
        return results
    
    # Helper method to calculate convergence between the current Q-table and an old copy passed to the function
    # Note: the Q-learner does not store its own copies to compare, but you can grab on with get_q_table()
    def get_convergence(
        self,
        inp_old_q_table
    ):
        return np.mean((self.Q_table - inp_old_q_table) ** 2)

    # Accessor function to get the current Q table values
    def get_q_table(
        self,
        inp_state_OPT=None, # If this is passed, return a set of action rewards for a given state
        inp_action_OPT=None # If this is passed, return a specific reward for a given action and state
    ):
        if(inp_state_OPT is None):
            return self.Q_table.copy()
        elif(inp_action_OPT is None):
            return self.Q_table[inp_state_OPT]
        else:
            return self.Q_table[inp_state_OPT][inp_action_OPT]
    
    # toString method to print relevant data quickly
    def __str__(self):
        s = "Learner data:\n"
        s += f"{self.total_states=}\n"
        s += f"{self.total_actions=}\n"
        s += f"{self.learning_rate=}\n"
        s += f"{self.rewards_rate=}\n"
        s += f"{self.exploration_rate=}\n"
        s += f"{self.exploration_rate_decay=}\n"
        return s

    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~ ######
    ###### Machine Learning Functions ######
    ###### ~~~~~~~~~~~~~~~~~~~~~~~~~~ ######

    # Function to calculate the best action to take given the current state
    def calculate_action(
        self,
        inp_new_state,
        training
    ):
        # Determine if a random action will be taken
        rng = random.random()
        if training:
            self.exploration_rate *= self.exploration_rate_decay

        if rng < self.exploration_rate and training:
            action = random.randint(0, self.total_actions - 1)
        else:
            # If not a random action, pick the best action based on the Q-Table for the current state
            action = np.argmax(self.Q_table[inp_new_state])

        # Set current state and action to use in next iteration for updating the Q-table
        # (these get ignored if we're querying the model without training)
        self.train_state = inp_new_state
        self.train_action = action

        return action
       
    # Method to update the Q-table with an experience tuple, requires a new state and reward for that state, and adds those to previously saved state and action to get to those
    def train_step(
        self,
        inp_new_state, # The new state entered
        inp_reward # The reward for entering that state
    ):
        # Update Q-table with experience tuple
        current_Q_value = self.Q_table[self.train_state][self.train_action]
        future_rewards = inp_reward + self.rewards_rate * (self.Q_table[inp_new_state, np.argmax(self.Q_table[inp_new_state])])
        self.Q_table[self.train_state][self.train_action] = (1.0 - self.learning_rate) * current_Q_value + self.learning_rate * future_rewards

        action = self.calculate_action(inp_new_state=inp_new_state, training=True)
        return action

    # Helper method for testing the model, call this each time a new state is entered and the next action is desired
    def test_step(
        self,
        inp_new_state
    ):
        return self.calculate_action(inp_new_state=inp_new_state, training=False)
