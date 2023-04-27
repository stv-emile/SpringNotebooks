#create grid

#entrainer pour 100 Rounds - chaque round abouti a un gain de cercle ou une perte de cercle
#boucler a l'infini jusqu'au gain ou une perte de cercle
#drapeau de tour : cercle et carre a tour de role.
#mettre un agent aleatoirement dans l'un des espaces vides (choix parmi les case a etat 0)
#les cases a etat zero = espace des action, E0 = [0,1,2,3,4,5,6,7,8]
#position
#A chaque action choisi E perd une action 
#gain carre et perd cercle si 
#[111xxxxxx],[xxx111xxx],[xxxxxx111],[1xxx1xxx1],[1xx1xx1xx],[x1xx1xx1x],[xx1xx1xx1],[xx1x1x1xx]



"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import random
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

#tkinter is for version 3.0 and later
#tkinter is toolkit interface

UNIT = 100   # pixels
TIC_TAC_TOE_H = 3 #4  # grid height
TIC_TAC_TOE_W = 3 #4  # grid width
#SIZE = 30 #15
HALF_UNIT = UNIT/2
HALF_UNIT_MINUS10 = HALF_UNIT - 10
SIZE = int(3*UNIT/8)  #HALF_UNIT_MINUS10

EMPTY = 0
SQUARE = 1
CIRCLE = 2

WON_STATE_SPACE = np.array([[1,1,1, 0,0,0, 0,0,0],
                   [0,0,0, 1,1,1, 0,0,0],
                   [0,0,0, 0,0,0, 1,1,1],
                   [1,0,0, 0,1,0, 0,0,1],
                   [1,0,0, 1,0,0, 1,0,0],
                   [0,1,0, 0,1,0, 0,1,0],
                   [0,0,1, 0,0,1, 0,0,1],
                   [0,0,1, 0,1,0, 1,0,0]])


class Tic_tac_toe(tk.Tk, object):
    def __init__(self):
        super(Tic_tac_toe, self).__init__()
        self.title('maze')
        self.geometry('{0}x{1}'.format(TIC_TAC_TOE_W * UNIT, TIC_TAC_TOE_H * UNIT))
        
        self.action_space = list(range(TIC_TAC_TOE_W * TIC_TAC_TOE_H))
        self.state = np.zeros(9, dtype=int)
        
        self.n_actions = len(self.action_space)
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)
        
        self.agents = []
        self._build_tic_tac_toe()
        
    def cpx(self, p):
        return p-3*(p//3)
    def cpy(self, p):
        return p//3

    def _build_tic_tac_toe(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=TIC_TAC_TOE_H * UNIT,
                           width=TIC_TAC_TOE_W * UNIT)
        # create grids
        for c in range(TIC_TAC_TOE_W+1):
            x0, y0 = c*UNIT, 0
            x1, y1 = x0, y0 + TIC_TAC_TOE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(TIC_TAC_TOE_H+1):
            x0, y0 = 0, r*UNIT
            x1, y1 = x0 + TIC_TAC_TOE_W * UNIT, y0
            self.canvas.create_line(x0, y0, x1, y1)
            
        self.square = self.create_square(self.choose_random_action())
        self.circle = self.create_circle(self.choose_random_action())
        
        
        # pack all
        self.canvas.pack()
        
        return self.square
    
    
        
    def render(self):
        time.sleep(0.1)
        self.update()
        
    def check_state_exist(self) :
        s = str(self.canvas.coords(self.rect))
        if s not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.action_space),
                    index=self.q_table.columns,
                    name=s,
                )
            )
    
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 'droite':   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'gauche':   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 'bas':   # bas
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 'haut':   # haut
            if s[1] > UNIT:
                base_action[1] -= UNIT
        

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        if self.canvas.coords(self.rect) == self.canvas.coords(self.oval) :
            return 0
        else :
            self.check_state_exist()
            return 1
        
    def chooseAction(self) :
        epsilon = 0.5 #pour plus ou moins d'exploration
        s = str(self.canvas.coords(self.rect))
        if np.random.uniform() < epsilon :
            action = np.random.choice(self.action_space)
        else : 
            actionScores = self.q_table.loc[s, :]
            action = np.random.choice(actionScores[actionScores == np.max(actionScores)].index)
        return action
    
    def learn(self, etatActuel, action, etatSuivant) :
        gamma = 0.9 #coefficient d’affaiblissement d’influence d’un état à un autre.
        ta = 0.1 #Le taux d'apprentissage caractérisant la part d’une nouvelle étape dans le score d'une action
        self.check_state_exist() #l'agent se trouve actuellement dans etatSuivant
        if etatSuivant == self.canvas.coords(self.oval) : #etatSuivant est terminal(recompensé)
            q_cible = 1
        elif etatSuivant == self.canvas.coords(self.enfer_23): #etat suivant est enfer 23 (puni)
            q_cible = -1
        elif etatSuivant == self.canvas.coords(self.enfer_32): #etat suivant est enfer 32 (puni)
            q_cible = -1
        else :
            actionScores = self.q_table.loc[str(etatSuivant), :]
            q_cible = gamma*np.max(actionScores)
        q_actuel = self.q_table.loc[str(etatActuel), action]
        self.q_table.loc[str(etatActuel), action] += ta*(q_cible - q_actuel)
    
    def reset(self) :
        self.update()
        time.sleep(0.5)
        
        self.action_space = list(range(TIC_TAC_TOE_W * TIC_TAC_TOE_H))
        self.state = np.zeros(9, dtype=int)
        
        for agent in self.agents:
            self.canvas.delete(agent)
        
    #_____________________________________________________________
                
        #case_number est 0, 1, 2, 3 ...
    def create_square(self, case_number):
        point_depart = np.array([UNIT/8,UNIT/8])
        x0, y0 = point_depart[0]+self.cpx(case_number)*UNIT, point_depart[1]+self.cpy(case_number)*UNIT
        x1, y1 = x0+3*UNIT/4, y0+3*UNIT/4
        return self.canvas.create_rectangle(x0, y0, x1, y1, fill='blue')

    def create_circle(self, case_number):
        point_depart = np.array([UNIT/8,UNIT/8])
        x0, y0 = point_depart[0]+self.cpx(case_number)*UNIT, point_depart[1]+self.cpy(case_number)*UNIT
        x1, y1 = x0+3*UNIT/4, y0+3*UNIT/4
        return self.canvas.create_oval(x0, y0, x1, y1, fill='yellow')
    
    #quand Square joue, il place un carre dans le case au numero corresspondant a l'action
    def square_play(self, action):
        self.agents.append(self.create_square(action))
        self.state[action]=1
   
    #quand Square joue, il place un carre dans le case au numero corresspondant a l'action
    def circle_play(self, action):
        self.agents.append(self.create_circle(action))
        self.state[action]=2
        
    
    def choose_random_action(self):
        action = self.action_space.pop(random.randrange(len(self.action_space)))
        return action

    def choose_exact_action(self,action):
        try:
            self.action_space.remove(action)
            return action
        except:
            return
    
    def is_square_won(self):
        for won_state in WON_STATE_SPACE:
            if np.sum(self.state & won_state) == 3:
                return True
        return False

    def is_circle_won(self):
        for won_state in WON_STATE_SPACE:
            if np.sum((self.state & won_state)//2) == 3:
                return True
        return False

def update():
    
    for i in range(1) :
        env.reset()
        #env.check_state_exist()
        square_turn = True
        while True :
        #for j in range(3):
            env.render()
            
            if square_turn:
                #choose action
                action = env.choose_random_action()
                env.square_play(action)
                square_turn = False
            else:
                #choose action
                action = en.choose_random_action() 
                env.circle_play(action)
                square_turn = True
        
            if env.is_circle_won():
                break
            if env.is_square_won():
                break

    
    #print(env.q_table)
    print('game over')

env.destroy()
 
if __name__ == "__main__":
    env = Tic_tac_toe()
    update()
    env.mainloop()