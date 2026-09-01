# Blackjack Deep reinforcement learning model - Training and Demonstration
# Copyright (c) **Author**: `Adam Paszke <https://github.com/apaszke>`_
#                            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_
# Copyright (c) 2025-2026 henrymcneill
# Licensed under the BSD 3-Clause License - See the LICENSE document,
# more details on GitHub at https://github.com/mcneill-h/BlackjackDeepLearning

import time
import math
from random import*

import torch # for DeepRL
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib # for the graph
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


## OPTIONAL: VALUE TO CHANGE FOR DEVELOPPERS:
device = torch.device("cpu") # selects what processor trains the DeepRL (Deep reinforcement learning);
nb_episodes = 12000 # number of episodes (games) the DeepRL plays


## Set up matplotlib (creates graph):
is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from IPython import display

plt.ion()
win_rate_history = []
latest_wins = 0 # to compute the win rate


## DeepRL Set up:
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.001
EPS_DECAY = 100
LEARNING_RATE = 1e-4

number_input = 14 # number of input values
number_output = 2 # number of output values
steps_done = 0

namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DeepRL(nn.Module):

    def __init__(self, number_input, number_output):
        super(DeepRL, self).__init__()

        self.layer1 = nn.Linear(number_input, 128)  # number of neurons and layers
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, number_output)

    def forward(self, x):
        x = F.relu(self.layer1(x))  ## RELU non-linearity
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DeepRL(number_input, number_output).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr= LEARNING_RATE, amsgrad=True)


def DeepRL_selects_action(state):
    
    global steps_done
    steps_done += 1
    
    sample = random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    
    else:
        return torch.tensor([[randint(0, number_output-1)]], device=device, dtype=torch.long)


def update_graph(show_result=False):
    
    plt.figure(1)
    
    if show_result:
        plt.title('Win rate of the Deep Learning AI over time (Result)')
    
    else:
        plt.clf()
        plt.title('Win rate of the Deep Learning AI over time (Training...)')

    plt.xlabel('Number of games (1 digit = 20 games)')
    plt.ylabel('Win rate of the AI against the dealer (%)')

    # Shows the average of the last 20 games (in blue):
    win_rate_results = torch.tensor(win_rate_history, dtype=torch.float)
    plt.plot(win_rate_results.numpy())

    # Shows the average of the last 500 games (in orange):
    average_range = 25
    larger_average = []

    for i in range(len(win_rate_results)):
        w = min(i+1, average_range)
        # prevents the graph from crashing when not sufficient data (beginning of training)

        larger_average.append(win_rate_results[i-w+1:i+1].float().mean())

    larger_average = torch.tensor(larger_average)
    plt.plot(larger_average.numpy())


    plt.pause(0.001)  # pause a bit so that graph is updated
    
    if is_ipython:
        
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        
        else:
            display.display(plt.gcf())
            

def optimize_model(state, action, next_state, reward):
    
    state_action_value = policy_net(state).gather(1, action)

    # Compute target value using policy network:
    with torch.no_grad():
        
        if next_state is None:
            target_value = reward
        
        else:
            target_value = reward + GAMMA * policy_net(next_state).max(1).values

    # Compute loss and backprop:
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_value, target_value.unsqueeze(1))
    
    optimizer.zero_grad()
    
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()
    

def blackjack_game(DeepRL_sum, dealer_sum, action, DeepRL_ace , dealer_ace):
    # Blackjack game simulation; simplified version (DeepRL can only hold or hit)
    
    # DeepRL's turn to play:
    if str(action) == "tensor([[1]])": # if DeepRL hits!
        
        new_DeepRL_card, DeepRL_ace = draw_card(DeepRL_ace) #new card
        DeepRL_sum += new_DeepRL_card

        if print_game_details == True and new_DeepRL_card == 11:
            print("The DeepRL has drawn an ace!")
        
        if (DeepRL_sum <= 21): # game still in play
            return DeepRL_sum, dealer_sum , 0, False, DeepRL_ace , dealer_ace
        
        elif DeepRL_ace >= 1 : # checks for aces
            DeepRL_sum = DeepRL_sum - 10 # reduces the number of the ace from 11 to 1
            return DeepRL_sum, dealer_sum , 0, False, (DeepRL_ace-1) , dealer_ace
            
        else: # DeepRL looses
            return DeepRL_sum, dealer_sum , -1, True, DeepRL_ace , dealer_ace
    
    # Dealer's turn to play:
    while dealer_sum < 17: # the dealer draws until it reaches 17
        
        new_card, dealer_ace = draw_card(dealer_ace)
        dealer_sum += new_card
        
        if print_game_details == True:
            print("Dealer takes new card:",new_card)
            print("The sum of the dealer's cards is now", dealer_sum)
        
        if dealer_sum > 21 and dealer_ace > 0:
            
            if print_game_details == True:
                print("The dealer has busted but he has an ace!, so he will turn it into a 1")
                print("The sum of the dealer's cards is now", dealer_sum-10)
            
            dealer_sum = dealer_sum - 10
            dealer_ace = dealer_ace - 1
    
    # Final results:
    if dealer_sum > 21: 
        reward = 1
        
    elif DeepRL_sum > dealer_sum:
        reward = 1
        
    elif DeepRL_sum == dealer_sum:
        reward = 0
        
    else:                       
        reward = -1

    return DeepRL_sum, dealer_sum, reward, True, DeepRL_ace , dealer_ace


def draw_card (ace):
    random_card = randint(0, len(cards_in_deck)-1)
    drawn_card = cards_in_deck[random_card]
    cards_in_deck.remove(drawn_card)

    # if an ace is drawn:
    if drawn_card==1:
        
        return 11 , (ace + 1) 

    return drawn_card, ace


def reset_card_deck():
    cards_in_deck.clear()

    number_of_decks = 2
    for i in range(number_of_decks):
        for j in range(4):
            for card in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]:  # there are four "10" , as the jack, queen, and king are worth 10 points
                cards_in_deck.append(card)



## Training loop:
print_game_details = False # prints each of the DeepRL's game details; used for the DeepRL's demonstration
training_start_timer = time.time()

cards_in_deck = []
reset_card_deck()
for i_episode in range(nb_episodes):

    if len(cards_in_deck) <= 70: # reset card deck
        reset_card_deck()

    is_game_finished = False
    
    DeepRL_ace = 0 # number of aces the DeepRL has
    dealer_ace = 0  # number of aces the dealer has

    # DeepRL draws his cards:
    DeepRL_card_1, DeepRL_ace = draw_card(DeepRL_ace)
    DeepRL_card_2, DeepRL_ace = draw_card(DeepRL_ace)
    DeepRL_sum = DeepRL_card_1 + DeepRL_card_2
    
    # Dealer draws his card:
    dealer_sum, dealer_ace = draw_card(dealer_ace)
    
    state = (DeepRL_sum, dealer_sum, DeepRL_ace, dealer_ace,
             cards_in_deck.count(1), cards_in_deck.count(2), cards_in_deck.count(3), cards_in_deck.count(4), cards_in_deck.count(5),
             cards_in_deck.count(6), cards_in_deck.count(7),cards_in_deck.count(8), cards_in_deck.count(9), cards_in_deck.count(10))
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # converts the state values into tensors

    for t in count():
        
        action = DeepRL_selects_action(state)
        # If 0 then hold, if 1 then hit

        DeepRL_sum, dealer_sum, reward, is_game_finished, DeepRL_ace , dealer_ace = blackjack_game(DeepRL_sum, dealer_sum, action, DeepRL_ace , dealer_ace)
        # Blackjack round is played
        
        reward = torch.tensor([reward], device=device) # converts the reward value into tensors

        if is_game_finished:
            next_state_tensor = None ## when game ends, we do not optimize equation!
        
        else:  
            next_state_tensor = torch.tensor((DeepRL_sum, dealer_sum, DeepRL_ace, dealer_ace, cards_in_deck.count(1), cards_in_deck.count(2), cards_in_deck.count(3),
                                              cards_in_deck.count(4), cards_in_deck.count(5), cards_in_deck.count(6), cards_in_deck.count(7),cards_in_deck.count(8),
                                              cards_in_deck.count(9), cards_in_deck.count(10)), dtype=torch.float32, device=device).unsqueeze(0)
            # converts the next state into tensors 

        ## trains the model (actualises weights)
        optimize_model(state, action, next_state_tensor, reward)
        
        state = next_state_tensor

        # actualises graph + break:
        if is_game_finished:

            if i_episode%20 == 0:
                win_rate_history.append((latest_wins/20)*100) # average of last 20 games
                latest_wins = 0
                
                update_graph()
                
            elif reward == 1:
                latest_wins += 1
                
            break


## DeepRL's training phase is complete:
update_graph(show_result=True)
plt.ioff()

training_end_timer = time.time()
training_time = training_end_timer - training_start_timer

print('Complete')
print("It took", int(training_time), "seconds!")
print("There was a total of", nb_episodes, "games.")
print("It took on average", int(training_time)/nb_episodes, "seconds per games.")


## DeepRL model demonstration:
print_game_details = True
loop = True
reset_card_deck()

while loop == True:
    answer = input ("Want a demo? (yes/no) ")
    
    if answer != "yes":
        loop = False
    
    else:
        if len(cards_in_deck) <= 70:  # reset card deck
            reset_card_deck()

        is_game_finished = False
        
        DeepRL_ace = 0 # number of aces the DeepRL has
        dealer_ace = 0 # number of aces the dealer has

        # DeepRL draws his cards:
        DeepRL_card_1,DeepRL_ace = draw_card(DeepRL_ace)
        DeepRL_card_2,DeepRL_ace = draw_card(DeepRL_ace)
        DeepRL_sum = DeepRL_card_1 + DeepRL_card_2

        # Dealer draws his card:
        dealer_sum, dealer_ace = draw_card(dealer_ace)

        state = (DeepRL_sum, dealer_sum, DeepRL_ace , dealer_ace, cards_in_deck.count(1), cards_in_deck.count(2), cards_in_deck.count(3), cards_in_deck.count(4),
                 cards_in_deck.count(5), cards_in_deck.count(6), cards_in_deck.count(7),cards_in_deck.count(8), cards_in_deck.count(9), cards_in_deck.count(10))

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # converts the state values into tensors

        print("Cards drawn by the DeepRL:", DeepRL_card_1, "and", DeepRL_card_2)
        print("the DeepRL's sum is", DeepRL_sum)

        if DeepRL_ace == 1:
            print("The DeepRL has an ace!")

        elif DeepRL_ace == 2:
            print("The DeepRL has two aces!")

        print("Card drawn by dealer:", dealer_sum)

        if dealer_ace == 1:
            print("It is an ace!")

        for t in count():
        
            action = DeepRL_selects_action(state) # If 0 then hold, if 1 then hit
            
            DeepRL_sum, dealer_sum, reward, is_game_finished, DeepRL_ace , dealer_ace = blackjack_game(DeepRL_sum, dealer_sum, action, DeepRL_ace , dealer_ace)

            state = torch.tensor((DeepRL_sum, dealer_sum, DeepRL_ace , dealer_ace, cards_in_deck.count(1), cards_in_deck.count(2), cards_in_deck.count(3),
                                  cards_in_deck.count(4), cards_in_deck.count(5), cards_in_deck.count(6), cards_in_deck.count(7),cards_in_deck.count(8),
                                  cards_in_deck.count(9), cards_in_deck.count(10)), dtype=torch.float32, device=device).unsqueeze(0)
            # converts the state values into tensors

            if str(action)== "tensor([[1]])": # if DeepRL hits!
                print("the DeepRL HITS (draws a card)")

            else:
                print("the DeepRL HOLDS (does NOT draw a card)")

            print("DeepRL's sum:", DeepRL_sum)
            print("Dealer's sum:", dealer_sum)

            if is_game_finished:

                if reward == 1:
                    print("the DeepRL won!!!")

                elif reward == -1 :
                    print("the DeepRL lost")

                else:# only if it is a tie
                    print("it is a tie!")
                    
                break

plt.show()
