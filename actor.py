import numpy as np
import task as Task
import random
from collections import deque, namedtuple
from keras import layers,models,optimizers
from keras import backend as K

class Actor():
    # actor policy model
    def __init__(self,state_size,action_size,action_high,action_low):
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        
        self.build_model()
        
    def build_model(self):
        # build actor/policy model that maps state to actions
        # define the input layer
        states = layers.Input(shape=(self.state_size,),name='states')
        
        # add hidden layers
        net = layers.Dense(units=32,activation='relu')(states)
        net = layers.Dense(units=64,activation='relu')(net)
        net = layers.Dense(units=32,activation='relu')(net)
        
        # add final output layer with sigmoid activation 
        raw_actions = layers.Dense(units=self.action_size,activation='sigmoid',name='raw_actions')(net)
        
        # scale [0,1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,name='actions')(raw_actions)
        
        # create Keras model
        self.model = models.Model(inputs=states,outputs=actions)
        
        # define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights,loss=loss)
        self.train_fn = K.function(inputs=[self.model.input,action_gradients, K.learning_phase()],outputs=[],updates=updates_op)