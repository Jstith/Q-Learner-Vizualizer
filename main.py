import pygame
import time
import random
from board import Board
from learner import Q_Learner

random.seed(int(time.time() * 1000))

def create_q_learner():

    num_states = 100
    num_actions = 4
    learning_rate = 0.2
    future_rewards_rate = 0.9
    exploration_rate = 0.5
    exploration_rate_decay = 0.99

    learner = Q_Learner(inp_total_states=num_states, inp_total_actions=num_actions, inp_learning_rate_OPT=learning_rate, inp_rewards_rate_OPT=future_rewards_rate, inp_exploration_rate_OPT=exploration_rate, inp_exploration_rate_decay_OPT=exploration_rate_decay)
    return learner

def draw_button(screen, text, x, y, width, height, font, color, text_color):
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

def is_button_clicked(mouse_pos, button_rect):
    return button_rect.collidepoint(mouse_pos)

pygame.init()

# Define GUI size variables

tile_size = 60
map_size = 10
padding = 30
board_length = tile_size * map_size
screen_width = board_length * 2 + padding * 3
screen_height = board_length * 2 + padding * 3

# Create screen
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
last_tick_time = pygame.time.get_ticks()
pygame.display.set_caption("Q-Learner")

# Create boards
b_environment = Board(tile_size=tile_size, map_size=map_size, offset_x=padding, offset_y=padding)
b_learner = Board(tile_size=tile_size, map_size=map_size, offset_x=(padding * 2 + board_length), offset_y=padding)

font = pygame.font.SysFont("Arial", 35, bold=True)
env_title = font.render("Environment", True, (245, 245, 245))
learner_title = font.render("Learner", True, (245, 245, 245))

alpha_text = font.render("Learning rate: 0.2", True, (245, 245, 245))
screen.blit(alpha_text, (700, 1000))

gamma_text = font.render("Rewards rate: 0.9", True, (245, 245, 245))
screen.blit(gamma_text, (700, 1050))

explored_text = font.render("Exploration rate decay: 0.99", True, (245, 245, 245))
screen.blit(explored_text, (700, 1100))

# Set state variables before starting main loop

weights_toggle = False # When false, weights are not displayed
weights_text = {
    True: "Weights: ON",
    False: "Weights: OFF"
}

train_toggle = False
train_text = {
    True: "Stop Training",
    False: "Start Training"
}

run_toggle = False
run_text = {
    True: "Stop Model",
    False: "Start Model"
}

speed_toggle = False
speed_check = 100
speed_text = {
    True: "Set to 1x speed",
    False: "Set to 2x speed"
}

# Declare state booleans to control flow
terrain_set_toggle = False # When false, you can change terrain. When true, you cannot change terrain
any_terrain_made = False
first_train_step = True
first_run_step = True

# Declare Q Leraner variables
learner = create_q_learner()
state = None
action = None
next_state = None
reward = None
state_weight = None
direction = None
old_q = learner.get_q_table()

num_episodes = 1
num_steps = 0
convergence = 0

while True:

    # Allows the environment to run at double speed wanted
    if(speed_toggle):
        speed_check = 50
    else:
        speed_check = 100

    current_time = pygame.time.get_ticks()
    if(current_time - last_tick_time) >= speed_check:
        last_tick_time = current_time

        if(train_toggle):
            # Taking actions to train the Q-learner

            # discretize the current state
            player_pos = list(b_environment.get_player_location())
            state = player_pos[0] * 10 + player_pos[1]

            if first_train_step: # First step, we don't have anything yet. Set our initial state and get an action back
                action = learner.test_step(state)
                first_train_step = False
                num_episodes = 1
                num_steps = 1
            else: # All other circumstances, we have an action, and the learner has a prior state. Now, take that action, and pass the new state and reward from that state to the learner
                # it will return a new action to take based on that.
                reward = b_environment.move_player(direction=action)
                b_learner.move_player(direction=action)
                state_weight = learner.get_q_table(inp_state_OPT=state, inp_action_OPT=action)
                direction = ((action + 2) % 4 )
                
                new_player_pos = b_environment.get_player_location()
                new_state = new_player_pos[0] * 10 + new_player_pos[1]
                
                # Takes a train step
                action = learner.train_step(inp_new_state=new_state, inp_reward=reward)
                
                num_steps += 1
                if(reward == 100): # We won the round
                    convergence = learner.get_convergence(inp_old_q_table=old_q)
                    old_q = learner.get_q_table()
                    num_episodes += 1
                    num_steps = 0

        elif(run_toggle):

            # Reset our homie for the big show
            if first_run_step:
                first_run_step = False
                b_environment.set_player_location((1,1))
                b_learner.set_player_location((1,1))

            # discretize the current state
            player_pos = list(b_environment.get_player_location())
            state = player_pos[0] * 10 + player_pos[1]

            # Get the action, we can take it right away no need to wait for the next rep, we aren't saving anything.
            action = learner.test_step(inp_new_state=state)
            b_environment.move_player(action)
            b_learner.move_player(action)

            # Get the Q-table value that got us to the new spot we're at (Q table value of the action we just took).
            state_weight = learner.get_q_table(inp_state_OPT=state, inp_action_OPT=action)
            direction = ((action + 2) % 4 )

        b_environment.draw_map(screen, weights_toggle)
        b_learner.draw_map(screen, weights_toggle, custom_weight=state_weight, direction=direction) # , custom_weights=state_weights

        # Add map titles
        screen.blit(env_title, (padding + board_length // 2 - env_title.get_width() // 2, padding // 2 - env_title.get_height() // 2))
        screen.blit(learner_title, (padding * 2 + board_length + board_length // 2 - learner_title.get_width() // 2, padding // 2 - env_title.get_height() // 2))

        # Line between maps and bottom
        pygame.draw.line(screen, (245, 245, 245), (padding, board_length + padding * 2), (screen_width - padding, board_length + padding * 2), 2)  
    
        # Add walls button
        b_wall_add_rect = pygame.Rect(30, (screen_height // 2 + 30), 300, 80)
        if(terrain_set_toggle):
            draw_button(screen, "Generate Walls", 30, (screen_height // 2 + 30), 270, 80, font, (100, 100, 100), (70, 70, 70))
        else:
            draw_button(screen, "Generate Walls", 30, (screen_height // 2 + 30), 270, 80, font, (70, 70, 70), (245, 245, 245))

        b_double_speed_rect = pygame.Rect(30, (screen_height // 2 + 120), 300, 80)
        draw_button(screen, speed_text[speed_toggle], 30, (screen_height // 2 + 120), 300, 80, font, (70, 70, 70), (245, 245, 245))

        # Add Train Learner Button
        b_train_learner = pygame.Rect(330, (screen_height // 2 + 30), 300, 80)
        if(not any_terrain_made):
            draw_button(screen, train_text[train_toggle], 330, (screen_height // 2 + 30), 270, 80, font, (100, 100, 100), (70, 70, 70))
        else:
            draw_button(screen, train_text[train_toggle], 330, (screen_height // 2 + 30), 270, 80, font, (70, 70, 70), (245, 245, 245))

        # Add start button
        b_start_rect = pygame.Rect(660, (screen_height // 2 + 30), 300, 80)
        if(not any_terrain_made):
            draw_button(screen, run_text[run_toggle], 660, (screen_height // 2 + 30), 270, 80, font, (100, 100, 100), (70, 70, 70))
        else:
            draw_button(screen, run_text[run_toggle], 660, (screen_height // 2 + 30), 270, 80, font, (70, 70, 70), (245, 245, 245))

        # Add weights toggle
        b_weights_rect = pygame.Rect(990, (screen_height // 2 + 30), 270, 80)
        draw_button(screen, weights_text[weights_toggle], 990, (screen_height // 2 + 30), 270, 80, font, (70, 70, 70), (245, 245, 245))

        # Text that changes all the time:
        episodes_text = font.render(f"Episode: {num_episodes}", True, (245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), (30, 1000, episodes_text.get_width()+50, episodes_text.get_height()))
        screen.blit(episodes_text, (30, 1000))
        
        steps_text = font.render(f"Step: {num_steps}", True, (245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), (30, 1050, steps_text.get_width()+50, steps_text.get_height()))
        screen.blit(steps_text, (30, 1050))

        explore = "{:.4g}".format(float(learner.get_learner_preferences(inp_exploration_rate_OPT=1)[0]))
        explore_text = font.render(f"Exploration Rate: {explore}", True, (245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), (30, 1100, explore_text.get_width()+50, explore_text.get_height()))
        screen.blit(explore_text, (30, 1100))

        convergence = "{:.4g}".format(float(convergence))
        convergence_text = font.render(f"Q-Table Convergence (MSE): {convergence}", True, (245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), (30, 1150, convergence_text.get_width()+50, convergence_text.get_height()))
        screen.blit(convergence_text, (30, 1150))

        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if is_button_clicked(event.pos, b_wall_add_rect): # Clicked to add a wall
                    any_terrain_made = True
                    if(not terrain_set_toggle): # Simulation hasn't started yet
                        
                        # Generate random terrain
                        b_environment.gen_new_map()
                        m_pointer = b_environment.get_game_map()
                        player_loc = b_environment.get_player_location()
                        for row in range(len(m_pointer)):
                            for col in range(len(m_pointer)):
                                tile = m_pointer[row][col]
                                rand = random.random()
                                if(tile == 'floor' and rand < 0.25 and not (row, col) == player_loc):
                                    tile = 'wall'
                                m_pointer[row][col] = tile
                        
                        # Set terrain (and weights internally) for both maps
                        b_learner.set_game_map(m_pointer)
                        b_environment.set_game_map(m_pointer)

                elif is_button_clicked(event.pos, b_start_rect): # Clicked start
                    if(not terrain_set_toggle): # Simulation hasn't started yet
                        terrain_set_toggle = True
                    if(not run_toggle):
                        run_toggle = True
                    else:
                        run_toggle = False
                elif is_button_clicked(event.pos, b_weights_rect):
                    if(not weights_toggle):
                        weights_toggle = True
                    else:
                        weights_toggle = False
                elif is_button_clicked(event.pos, b_train_learner):

                    # Now that we're going to train, make the terrain set
                    if(not terrain_set_toggle):
                        terrain_set_toggle = True

                    if(not train_toggle): # Start training the model
                        train_toggle = True

                    else:
                        train_toggle = False
                elif is_button_clicked(event.pos, b_double_speed_rect):
                    if(speed_toggle):
                        speed_toggle = False
                    else:
                        speed_toggle = True            

        pygame.display.flip()
