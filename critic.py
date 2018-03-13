import numpy as np
import task as Task
import random
from collections import deque, namedtuple
from keras import layers,models,optimizers
from keras import backend as K

class Critic():
    # critic value model
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
    def build_model(self):
        # build a critic value network that maps State,Action pairs to Q-values
        # define input layers
        states = layers.Input(shape=(self.state_size,),name='states')
        actions = layers.Input(shape=(self.action_size,),name='actions')
        # add hidden layers for state pathway
        states_net = layers.Dense(units=32, activation='relu')(states)
        states_net = layers.Dense(units=64, activation='relu')(states_net)
        
        # add hidden layers for actions pathway
        actions_net = layers.Dense(units=32, activation='relu')(actions)
        actions_net = layers.Dense(units=64, activation='relu')(actions_net)
        
        # combine state and action pathways
        net = layers.Add()([states_net,actions_net])
        net = layers.Activation('relu')(net)
        
        # add final output layer to produce action values (Q-values)
        Q_values = layers.Dense(units=1, name='Q_values')(net)
        
        # create Keras model
        self.model = models.Model(inputs=[states,actions], outputs=Q_values)
        
        # define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # compute action gradients - derivatives of Q with respect to actions
        action_gradients = K.gradients(Q_values,actions)
        
        # define function to get action gradients
        self.get_action_gradients = K.function(inputs=[*self.model.input,K.learning_phase()],outputs=action_gradients)