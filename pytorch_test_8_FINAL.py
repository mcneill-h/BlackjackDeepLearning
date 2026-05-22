## import the PyTorch library
import torch

## Import the time library
import time

## Calculate the start time
start = time.time()

import math
from random import*
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()



device = torch.device("cpu") ## you cannot use GPU because Pytorch does not support the Intel GPU of this Macbook


# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

testing_AI = False # says to the game loop to not send any messages to the console during the training time


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        ###check
        
        ## CONSIDER CHANGING !!!
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64) 
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x)) ## applied between layers to create non-linearity and make the AI leanr complex patterns 
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
#####################

# LR is the learning rate of the ``AdamW`` optimizer

GAMMA = 0.99 ## at "0" the agent considers only immediate rewards, "0.5" future rewards counts half as much as immediate, "0,99" almost as important

EPS_START = 0.9 ## how much the agent explores 
EPS_END = 0.001

EPS_DECAY = 80 ## the higher is the number, the slower is the decay of EPS
## Consider lowering!! put it high enough so that it trains slowly


LR = 1e-4 #3e-4 ## Learning rate, CONSIDER CHANGING



n_actions = 2 ##output number

n_observations = 4 ## number of input


policy_net = DQN(n_observations, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)



steps_done = 0

#### Q-network 
def select_action(state): ## Greedy policy!?
    
    global steps_done
    sample = random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    
    if sample > eps_threshold: ## Greedy policy?
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        ## random choice
        return torch.tensor([[randint(0, n_actions-1)]], device=device, dtype=torch.long)


episode_durations = []


###### FOR GRAPH AND VISUALISATION
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Win rate of the Deep Learning AI over time (Result)')
    else:
        plt.clf()
        plt.title('Win rate of the Deep Learning AI over time (Training...)')
    plt.xlabel('Number of games (1 digit = 20 games)')
    plt.ylabel('Win rate of the AI against the dealer (%)')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    

    
    window = 25
    means = []

    for i in range(len(durations_t)):
        w = min(i+1, window)
        means.append(durations_t[i-w+1:i+1].float().mean())

    means = torch.tensor(means)
    plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
            

################# Minor Training Loop
def optimize_model(state, action, next_state, reward): # GPT!!!!!
    # Compute Q(s, a)
    state_action_value = policy_net(state).gather(1, action)

    # Compute target value using policy network
    with torch.no_grad():
        if next_state is None:
            target_value = reward
        else:
            target_value = reward + GAMMA * policy_net(next_state).max(1).values

    # Compute loss and backpropagate
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_value, target_value.unsqueeze(1))
    
    optimizer.zero_grad()
    
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()
    
    


#####CREATE GAME
def game(player_sum, dealer_sum, action, ace_player , ace_dealer): 
    
    dealers_turn = False
    
    ##Player's turn to play
    if str(action)== "tensor([[1]])":
        
        new_player_card, ace_player = value(randint(1, 13),ace_player) #new card
        
        player_sum += new_player_card
        
        if testing_AI == True and new_player_card == 11:
            print("The player has drawn an ace!")
        
        if (player_sum <= 21): ##still in play  
            return (player_sum, dealer_sum) , 0, False, ace_player , ace_dealer
        
        elif ace_player >= 1 :# checks for aces
            player_sum = player_sum - 10 # reduces the number of the ace
            return (player_sum, dealer_sum) , 0, False, (ace_player-1) , ace_dealer #returns and takes away the power of the ace of the player
            
        else: ## player looses
            return (player_sum, dealer_sum) , -1, True, ace_player , ace_dealer
    
    
    while dealer_sum < 17: # makes the dealer draw until it reaches 17 for the dealer
        new_card, ace_dealer = value(randint(1,13),ace_dealer)
        dealer_sum += new_card
        
        if testing_AI == True:
            print("Dealer takes new card:",new_card)
            print("The sum of the dealer's cards is now", dealer_sum)
        
        if dealer_sum > 21 and ace_dealer > 0:
            
            if testing_AI == True:
                print("The dealer has busted but he has an ace!, so he will turn it into a 1")
                print("The sum of the dealer's cards is now", dealer_sum-10)
            dealer_sum = dealer_sum - 10
            ace_dealer = ace_dealer - 1 
        
        
    done = True
    
    if dealer_sum > 21:         # dealer busts
        reward = 1
        
    elif player_sum > dealer_sum:
        reward = 1
        
    elif player_sum == dealer_sum:
        reward = 0
        
    else:                       # dealer wins
        reward = -1
        

                    
    next_state = (player_sum, dealer_sum)
    
    return next_state, reward, done, ace_player , ace_dealer
    
    

def value (card, ace): ## sets up the value of the cards
    
    if (card == 11) or (card == 12) or (card == 13): #change face cards into "10"
        card = 10
    
    elif card==1: #determines if it is an ace
        
        return 11 , (ace + 1) #returns the ace as "11" (mke it a soft sum)

    return card, ace 






num_episodes = 4000
latest_wins = 0

##training loop 
for i_episode in range(num_episodes):
    
    done = False #reset
    
    ace_player = 0 # stores the number of aces the DQN has
    ace_dealer = 0  # stores the number of aces the player has
    # for this first version, we do not use the fact that aces could represent a 1 or 11
    
    # sum of the players cards
    player_card_1, ace_player = value(randint(1, 13),ace_player)
    
    player_card_2, ace_player = value(randint(1, 13),ace_player)
    
    player_sum = player_card_1 + player_card_2 ## sum of the cards
    
    # at the beginning, the dealer only has one card
    dealer_sum, ace_dealer = value(randint(1, 13),ace_dealer)
    
    state = (player_sum, dealer_sum, ace_player, ace_dealer)    
    
    #### converts the state into tensors
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    ##barely changes anything
    
    
    for t in count():##count?????????
        
        action = select_action(state) # If 0 then hold, if 1 then hit
        
        next_state, reward, done, ace_player , ace_dealer = game(player_sum, dealer_sum, action, ace_player , ace_dealer) ##not perfect 
        
        ##updates the player and dealer's sums
        player_sum, dealer_sum = next_state
        
        
        reward = torch.tensor([reward], device=device)
        

        #### converts the next state into tensors 
        if done:
            next_state_tensor = None ## when game ends, we do not optimize equation because we need the next state! 
        else:  
            next_state_tensor = torch.tensor((player_sum, dealer_sum, ace_player, ace_dealer), dtype=torch.float32, device=device).unsqueeze(0)
        

        ## trains the model (actualises weights)
        optimize_model(state, action, next_state_tensor, reward)
        
        state = next_state_tensor
        
    
        ##actualise le graphe + break
        if done:

            if i_episode%20 == 0: 
                episode_durations.append((latest_wins/20)*100)
                latest_wins = 0
                plot_durations()
            elif reward == 1:
                latest_wins += 1
               
                
            break

print('Complete')

plot_durations(show_result=True)

plt.ioff()


end = time.time()
length = end - start

# Show the results : this can be altered however you like
print("It took", int(length), "seconds!")
print("There was a total of", num_episodes, "games.")
print("It took on average", int(length)/num_episodes, "seconds per games.")


######## DEMO of the Model

testing_AI = True
loop = True 
while loop == True:
    plt.pause(0.1)
    answer = input ("Want a demo? (yes/no) ")
    
    if answer == "yes":
        done = False #reset
        
        ace_player = 0 # stores the number of aces the DQN has
        ace_dealer = 0  # stores the number of aces the player has
    
        player_card_1,ace_player = value(randint(1, 13),ace_player)
        player_card_2,ace_player = value(randint(1, 13),ace_player)
        
        print("Cards drawn by the DQN:",player_card_1,"and",player_card_2)
        
        #Just used to annonce the number of aces the player has
        if ace_player == 1:
            print("The player has an ace!")
        elif ace_player == 2:
            print("The player has two aces!")
        
        player_sum = player_card_1 + player_card_2 ## sum of the cards
        
        print("the DQN's sum is", player_sum)
        # at the beginning, the dealer only has one card
        dealer_sum,ace_dealer = value(randint(1, 13),ace_dealer)
        
        plt.pause(1)
        
        print("Card drawn by dealer:",dealer_sum)
        if ace_dealer ==1:
            print("It is an ace!")
        plt.pause(1) 
        

        
        #### converts the state into tensors
        state = torch.tensor((player_sum, dealer_sum, ace_player , ace_dealer), dtype=torch.float32, device=device).unsqueeze(0)
        ##barely changes anything

  
        
       
        for t in count():##count
        
            action = select_action(state) # If 0 then hold, if 1 then hit
            
            if str(action)== "tensor([[1]])":
                print("the DQN HITS (draws a card)")
            else:
                print("the DQN HOLDS (does NOT draw a card)")
                
            plt.pause(1)
            
            
            next_state, reward, done, ace_player , ace_dealer = game(player_sum, dealer_sum, action, ace_player , ace_dealer) ##not perfect 
            
            
            ##updates the player and dealer's sums
            player_sum, dealer_sum = next_state
            
            print("DQN's sum:",player_sum)
            print("Dealer's sum:",dealer_sum) 
            
            #### converts the state into tensors
            state = torch.tensor((player_sum, dealer_sum, ace_player , ace_dealer), dtype=torch.float32, device=device).unsqueeze(0)
        
            plt.pause(1)
            
            if done:
                if reward == 1:
                    print("the DQN won!!!")
                elif reward == -1 :
                    print("the DQN lost")
                else:# only if it is a tie
                    print("it is a tie!")
                    
                break
            
        
        
    else:
        loop = False


plt.show()
