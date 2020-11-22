# machine-learning-snake

start Player.py:
  - Input 0: Trains one model and renders some games (as defined in the variable games_to_render)
  - Input 1: Trains all available models, simulates games and outputs an comparisation for between the different models

requirements:
  - Snake.py has to be in the same directory
  - some dataset has to be in the same directory (saved_random_actions.npy or saved_expert_player.npy)
  - set the selected_dataset variable
    - for saved_random_actions.npy it is recommended to set the balancing variable to False
    - for saved_expert_player.npy it is recommended to set the balancing variable to True and the balancing_factor to 4
    
 ## creating your own dataset
 
 ### With randomly generated actions: start randomactions.py
  - overwrites saved_random_actions.npy if it exists!!
 ### Play snake yourself: start expertplayer.py
  - set the score_requirement variable (higher score requirement leads to higher performance after training the model)
  - set the games_i_want_to_play variable
  - if saved_expert_player.npy already exists, the data will be appended to that file

