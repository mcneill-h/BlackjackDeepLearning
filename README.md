# <p align="center"> Deep Learning model for Blackjack </p>
<p align="center">
<img width="585" height="400" alt="Screenshot 2026-06-24 at 11 31 08" src="https://github.com/user-attachments/assets/bc152118-2378-43ba-801d-cb782a1a7f43" />
</p>
<p align="center">
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">  <img src="https://img.shields.io/badge/License-BSD%203--Clause-blue?style=for-the-badge" alt="BSD 3-Clause License">  <img src="https://img.shields.io/badge/Release-v2.0.0-blue?style=for-the-badge" alt="Release">  <img src="https://img.shields.io/badge/Open_Source-Yes-success?style=for-the-badge" alt="Open Source">
</p>
<p align="center">
This project contains modified code derived from PyTorch tutorials -> intermediate sources -> reinforcement_q_learning.py
</p>
<p align="center">
  
  <https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py>
</p>

Copyright (c) 2017-2025 Adam Paszke <https://github.com/apaszke>, Mark Towers <https://github.com/pseudo-rnd-thoughts>

Copyright (c) 2025-2026 henrymcneill

The original program was designed for active physical motions, whereas minetargeted a stochastic card game. I deleted and rewrote most parts of the initial DQN program, only retaining the general structure and key PyTorch algorithms.

# What is this Deep Learning Model?
This project is a **Python** implementation of a **deep reinforcement learning model** using **PyTorch** to play Blackjack. Built by adapting an existing Deep Q-Learning algorithm, this project focuses on training an AI capable of handling the stochastic nature of Blackjack and learning optimal decision-making strategies through experience.

The model only has access of the number of aces it has, the sum of the dealer’s cards, and the sum of the its cards. The graph on the side represents the model’s win rate while training. The blue line represents the average win rate of the last 20 games, while the orange one represents the last 500 games. As we can observe, the model achieves a win rate of approximatively 38%-40%, getting at 42% at best. These are great results as the model's win rate is very close to the statistical 42% a player, who doesn't count cards, could have.

<p align="center">
<img width="253" height="188" alt="Screenshot 2026-05-26 at 12 45 32" src="https://github.com/user-attachments/assets/ac227f68-baa7-4158-ab5f-3b4f1c53036a" />  <img width="250" height="188" alt="Screenshot 2026-06-23 at 12 22 54" src="https://github.com/user-attachments/assets/0d630de5-a974-45fe-a58f-1f3e222e898f" />  <img width="250" height="188" alt="Screenshot 2026-06-23 at 12 23 40" src="https://github.com/user-attachments/assets/a59a8585-1441-41e5-b1fb-cd44a625a7ce" />
</p>
The model is set to play 4000 games in ~30 seconds, for an average of 0.007 seconds per game. As I don't have a GPU, I am using a CPU to compute.

The following step would be to not randomly generate numbers, but according to how a real deck of cards would (a card that was already played once can't appear a second time). Therefore, the model could learn how to count cards and find techniques to improve its win rate.


# How it works?
After copying the code and downloading the libraries, you can launch the model's training. Then, a real-time graph, similar to the examples above, will appear, showing the model's progression. After a moment, the training will stop. At the end, through the console, we can test the model by making it play individual games. Therefore, we can see what moves (hit or stand) the model would play for a given position.
<p align="center">
<img width="360" height="195" alt="Screenshot 2026-06-24 at 11 32 20" src="https://github.com/user-attachments/assets/1772d95f-6581-401e-ad1f-19a79c6f1a3d" /> <img width="255" height="195" alt="Screenshot 2026-06-24 at 11 50 40" src="https://github.com/user-attachments/assets/24ace831-4b44-4c05-8a22-2abd32625bf6" />
</p>


# Installation
Download the Python code that is in the repository. Then, we need to download the following libraries: math, random, time (v.1.0.0), matplotlib (v.3.10.1) and PyTorch (v.1.0.2). We would preferably want to use the same versions as listed above.


# License
Licensed under the BSD 3-Clause License - See the _LICENSE_ document
