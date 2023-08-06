import tensorflow as tf
import numpy as np

def TfDynamics(i_state,c):
    # Define Constants
    m = 0.15; g = 9.81; L = 0.5; b = 0.1

    dx1 = tf.reshape(i_state[:,1],(-1,1))
    dx2 = tf.subtract( tf.add( m*g*L*tf.sin( tf.reshape(i_state[:,0],(-1,1)) ), c ),b* tf.reshape(i_state[:,1],(-1,1)) )/(m*L**2)
    dx = tf.concat([dx1,dx2],axis=1)
    return dx

def Dynamics(t,y,u): 
    # Define Constants
    m = 0.15; g = 9.81; L = 0.5; b = 0.1

    dy = [0]*2
    dy[0] = y[1]
    dy[1] = (m*g*L*np.sin(y[0]) + u - b*y[1])/(m*L**2)
    return dy