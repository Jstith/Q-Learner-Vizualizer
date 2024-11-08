import pygame

class Board:

    def __init__(self, tile_size = 60, map_size = 10, offset_x = 0, offset_y = 0, board_name="Board"):

        # Define map sizes
        self.tile_size = tile_size
        self.map_size = map_size
        
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Define colors for entities
        self.colors = {
            'wall': (70, 70, 70),
            'floor': (255, 255, 255),
            'player': (0, 255, 0),
            'goal': (255, 0, 0)
        }

        self.weight_map = {
            'wall': -1000,
            'floor': -1,
            'goal': 100
        }

        self.gen_new_map()
        self.assign_weights()
        self.font = pygame.font.SysFont("Arial", 12, bold=False)

    def gen_new_map(self):
        
        # Create game map with all floors
        self.game_map = [['floor' for _ in range(self.map_size)] for _ in range(self.map_size)]

        # Add walls to game map
        for i in range(self.map_size):
            self.game_map[0][i] = 'wall'
            self.game_map[self.map_size - 1][i] = 'wall'
            self.game_map[i][0] = 'wall'
            self.game_map[i][self.map_size - 1] = 'wall'

        # Place player and goal
        self.player_location = (1, 1)
        self.game_map[9][8] = 'goal'

    def assign_weights(self):
        self.weights = [[self.weight_map[tile] for tile in row] for row in self.game_map]
        self.custom_weight_map = [[None for _ in row] for row in self.game_map]

    def get_player_location(self):
        return self.player_location
    
    def set_player_location(self, location):
        self.player_location = location

    def get_game_map(self):
        return self.game_map
        
    def set_game_map(self, inp_map):
        self.game_map = inp_map
        self.assign_weights()

    def draw_map(self, canvas, draw_weights, custom_weight=None, direction=None):
        pos = list(self.player_location)
        for row in range(self.map_size):
            for col in range(self.map_size):
                tile = self.game_map[row][col]
                color = self.colors[tile]
                if((row, col) == self.player_location):
                    color = self.colors['player']
                pygame.draw.rect(canvas, color, 
                                 (self.offset_x + col * self.tile_size,
                                  self.offset_y + row * self.tile_size,
                                  self.tile_size, self.tile_size))
                
                if draw_weights:
                    
                    if custom_weight is None:
                        weight = self.weights[row][col]
                        weight_text = self.font.render(str(weight), True, (0, 0, 0))
                        text_rect = weight_text.get_rect(center=(self.offset_x + col * self.tile_size + self.tile_size // 2, self.offset_y + row * self.tile_size + self.tile_size // 2))
                        canvas.blit(weight_text, text_rect)  # Blit the weight text onto the canvas
                    
                    else: # Using custom weights

                        if(row == pos[0] and col == pos[1]):
                            
                            formatted_weight = "{:.4g}".format(custom_weight)
                            
                            if(direction == 0):
                                formatted_weight = formatted_weight + " R"
                            elif(direction == 1):
                                formatted_weight = formatted_weight + " U"
                            elif(direction == 2):
                                formatted_weight = formatted_weight + " L"
                            elif(direction == 3):
                                formatted_weight = formatted_weight + " D"

                            self.custom_weight_map[row][col] = formatted_weight
                            weight = self.custom_weight_map[row][col]
                            weight_text = self.font.render(weight, True, (0, 0, 0))
                            text_rect = weight_text.get_rect(center=(self.offset_x + col * self.tile_size + self.tile_size // 2, self.offset_y + row * self.tile_size + self.tile_size // 2))
                            canvas.blit(weight_text, text_rect)  # Blit the weight text onto the canvas

                        else:

                            weight = self.custom_weight_map[row][col]
                            if weight is not None:
                                weight_text = self.font.render(weight, True, (0, 0, 0))
                                text_rect = weight_text.get_rect(center=(self.offset_x + col * self.tile_size + self.tile_size // 2, self.offset_y + row * self.tile_size + self.tile_size // 2))
                                canvas.blit(weight_text, text_rect)  # Blit the weight text onto the canvas
                                
                    

                    
                        



    # Player movement functions
    def move_player(self, direction): # 0 = right, 1 = up, 2 = left, 3 = down

        # The player can move "into" a wall, but they will receive the penalty then get moved back out of it to where they previously were.
        proposed_location = list(self.player_location)

        # Calculate proposed move on the map
        if(direction == 0):
            proposed_location[1] += 1
        elif(direction == 1):
            proposed_location[0] -= 1
        elif(direction == 2):
            proposed_location[1] -= 1
        elif(direction == 3):
            proposed_location[0] += 1

        # If there is a wall, overwrite the actual player position
        if(self.game_map[proposed_location[0]][proposed_location[1]] == 'wall'):
            reward = self.weights[proposed_location[0]][proposed_location[1]]
        # If the player finds the goal, they get the reward for it but can't stay (for training at least)
        elif(self.game_map[proposed_location[0]][proposed_location[1]] == 'goal'):
            self.player_location = (1, 1)
            reward = self.weights[proposed_location[0]][proposed_location[1]]
        else:
            # Move and assess reward
            self.player_location = tuple(proposed_location)
            reward = self.weights[proposed_location[0]][proposed_location[1]]
        
        # Will return reward once Q implemented
        return reward