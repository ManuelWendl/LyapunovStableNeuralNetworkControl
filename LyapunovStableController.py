import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Custom loss and activation functions 
class LyapunovStabilityLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self,V,dV,dxh,alpha):
        '''
        Lyapunov Stability loss:
        ========================
        Computes the loss of exponential stability by calculating 
        dV^T dxh + alpha V
        '''
        loss = tf.nn.relu(tf.add(tf.keras.backend.batch_dot(dV,dxh),tf.multiply(alpha,V)))
        return loss

class ControlBehaviourLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self,c,V,lam1,lam2,dV,dxh,alpha):
        '''
        Control behaviour loss:
        =======================
        Computes the loss for the control behaviour. 
        Actuator energy loss is computed by: (1/5*G_C(x))^2
        State energy loss is computed by: 1-relu((V(t)-V(t+1))/V(t))
        Weighting factor applied to auctuator energy loss, to stabilise training.
        '''
        weighting = tf.nn.relu(tf.tanh(-lam2*tf.add(tf.keras.backend.batch_dot(dV,dxh),tf.multiply(alpha,V))))
        u = lam1 * tf.multiply(tf.pow(c/5.0,2),tf.stop_gradient(weighting))
        return 1/len(V)*tf.reduce_sum(u,axis = 0,keepdims=True), u

def mirrStep(x):
    '''
    Mirrored Step function:
    =======================
    Computes the mirrored step function: 
    - infinity < x < 0  -> 1
    0 <= x < infinity   -> 0
    '''
    return tf.keras.backend.cast_to_floatx(tf.keras.backend.less_equal(x,0.0))

def smoothRelu(x): 
    '''
    Smooth ReLU function:
    =====================
    Computes the smooth ReLU function (sReLU):
    x < 0       -> 0
    0 <= x <= 1 -> 1/2 x^2
    x > 1       -> x
    Continuously differntiable ReLU function. 
    '''
    part0 = tf.keras.backend.cast_to_floatx(tf.keras.backend.less(x,0.0))
    part1 = tf.keras.backend.cast_to_floatx(tf.keras.backend.less_equal(x,1.0))
    part2 = tf.keras.backend.cast_to_floatx(tf.keras.backend.greater(x,1.0))
    return (1.0-part0)*part1*tf.keras.backend.pow(x,2)/2+part2*(x-1/2)

def parabolicActivation(x): 
    '''
    Parabolic Activation function: 
    ==============================
    Parabolic activation: 
    y = 1/2 x^2
    '''
    return tf.multiply(tf.keras.backend.pow(x,2),.5)

def getGrid(refinement,gridtype):
    '''
    getGrid:
    ========
    Returns the domain grid. Can be distinguished between a exponential and a linear distribution of points. 
    Finer mesh resolution in the neighbourhood of the critical point. 
    '''
    # Define support vectors: 
    vec1,vec2,vec3,vec4 = [],[],[],[]
    if gridtype == 'exp':
        factor = 2
        for i in range(0,refinement):
            vec1.append(-np.pi/(factor**i))
            vec2.append(np.pi/(factor**(refinement-i-1)))
            vec3.append(-7/(factor**i))
            vec4.append(7/(factor**(refinement-i-1)))
    elif gridtype == 'lin':
        maxh = 2.0/(refinement+1) 
        a = maxh/refinement
        sum_a = 0
        for i in range(1,refinement+1):
            sum_a += a*i
            vec1.insert(0,-np.pi*sum_a)
            vec2.append(np.pi*sum_a)
            vec3.insert(0,-7*sum_a)
            vec4.append(7*sum_a)
        
    vec1.append(0)
    vec3.append(0)
    vecphi = vec1+vec2
    vecdphi = vec3+vec4  
    print(vecphi)
    print(vecdphi)
    # Meshgrid of support vectors
    [phi,dphi] = np.meshgrid(vecphi,vecdphi)          
    # Flatten grid
    phi = np.reshape(phi,(-1,1))
    dphi = np.reshape(dphi,(-1,1))
    D = np.concatenate((phi,dphi),axis=1)
    return D

# Lyapunov Stable Controller
class LyapunovStableController:
    '''
    LyapunovStableController:
    =========================
    Controller class. The class can be trained and used for prediction. 
    '''
    def __init__(self,num_states, num_target, dynamics, refinement, gridtype) -> None:
        self.num_states = num_states
        self.num_target = num_target
        self.dynamics = dynamics
        self.refinement = refinement
        self.gridtype = gridtype

        self.model_c, self.model_l, self.optimiser = self.initNNs(self.refinement,self.gridtype)

    def initNNs(self,refinement,gridtype):
        '''
        initNNs:
        ========
        Initialise the structure of the Neural Networks for the controller. 
        Function is called when initialising the Controller class. 
        '''
        norm_layer = tf.keras.layers.Normalization()
        norm_layer.adapt(getGrid(refinement=refinement, gridtype=gridtype))

        # Define input of network
        input_states = tf.keras.Input((self.num_states,))
        input_states_norm = norm_layer(input_states)
        input_targets = tf.keras.Input((self.num_target,))
        input_targets_norm = norm_layer(input_targets)

        # Define controller branch of network
        cinputs = tf.keras.layers.subtract([input_states_norm,input_targets_norm])
        c1 = tf.keras.layers.Dense(5, activation='relu')(cinputs)
        c2 = tf.keras.layers.Dense(10, activation='relu')(c1)
        c3 = tf.keras.layers.Dense(20, activation='relu')(c2)
        c4 = tf.keras.layers.Dense(10, activation='relu')(c3)
        c5 = tf.keras.layers.Dense(1, activation='tanh')(c4)
        cout = tf.keras.layers.Lambda(lambda x: x * 5)(c5)

        input_lyapunov = tf.keras.Input((self.num_states,))
        input_lyapunov_norm = norm_layer(input_lyapunov)

        # Define Lyapunov network
        l1 = tf.keras.layers.Dense(10, activation=parabolicActivation,bias_constraint=tf.keras.constraints.non_neg(),kernel_constraint=tf.keras.constraints.non_neg())(input_lyapunov_norm)
        l2 = tf.keras.layers.Dense(20, activation=smoothRelu,bias_constraint=tf.keras.constraints.non_neg(),kernel_constraint=tf.keras.constraints.non_neg())(l1)
        l3 = tf.keras.layers.Dense(10, activation=smoothRelu,bias_constraint=tf.keras.constraints.non_neg(),kernel_constraint=tf.keras.constraints.non_neg())(l2)
        l4 = tf.keras.layers.Dense(1, activation='tanh',bias_constraint=tf.keras.constraints.non_neg(),kernel_constraint=tf.keras.constraints.non_neg())(l3)
        lout = tf.keras.layers.Lambda(lambda x: x * 10)(l4)

        # Define models:
        model_c = tf.keras.Model(inputs = [input_states,input_targets], outputs = cout)
        model_l = tf.keras.Model(inputs = input_lyapunov, outputs = lout)

        optimiser = tf.keras.optimizers.Adam()
        model_c.compile(optimizer=optimiser)
        model_l.compile(optimizer=optimiser)

        # PLot structure of model:
        model_c.summary()
        model_l.summary()

        return model_c, model_l, optimiser
    

    def trainNNs(self, epochs, batchsize, batchsizeStep, batchsizeIncrement, lam1, lam2, alpha, lr, lrRampStep, lrRampFactor, HeunBool):
        '''
        trainNNs:
        =========
        trains Neural Networks, with given learning paramaters. Returns the loss history from training.
        '''
        lossHistory = np.array([[0],[0]]).reshape(2,1)

        self.optimiser.learning_rate = lr
        steps = (2*self.refinement+1)**2

        D = getGrid(refinement=self.refinement,gridtype=self.gridtype)

        # Initilaise loss function
        lyapunovstabilityloss = LyapunovStabilityLoss()
        controlbehaviourloss = ControlBehaviourLoss()
                
        # Custom Training Function
        for epoch in range(epochs):
            if epoch%lrRampStep == 0 and epoch != 0:
                self.optimiser.learning_rate = self.optimiser.learning_rate*lrRampFactor

            if epoch%batchsizeStep == 0 and epoch != 0:
                batchsize = batchsize + batchsizeIncrement

            np.random.seed(42)
            print('\nepoch Nr: ',epoch,' with lr= ', self.optimiser.learning_rate)
            perm = np.random.permutation(np.arange(0,np.size(D,axis=0)))
            D = D[perm,:]
            noise = np.zeros(shape=D.shape)
            addnoise = np.random.randn(np.size(D,0),np.size(D,1)) 
            noise[0:2:,:] = addnoise[0:2:,:]
            noise = noise[perm,:]
            D = np.add(D,noise*0.01)

            violatingStability1 = 0
            violatingStability2 = 0

            for i in range(0,steps-batchsize,batchsize):
                i_target = tf.reshape(tf.convert_to_tensor(np.zeros((batchsize,2))),(batchsize,2))

                i_state = tf.reshape(tf.convert_to_tensor(D[i:i+batchsize,:],dtype=tf.float32),(batchsize,2))

                i_state_l = tf.Variable(i_state,dtype=tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    c = self.model_c([i_state,i_target],training = True)
                    # First Step of Heun
                    dx = self.dynamics(i_state,c)

                    if HeunBool:
                        state = tf.add(i_state,tf.multiply(0.1,dx))
                        ch = self.model_c([state,i_target],training = True)
                        #  Second step of Heun
                        dxh = self.dynamics(state,ch)
                        DX = tf.multiply(0.5,tf.add(dx,dxh))
                    else: 
                        DX = dx

                    # Calculate Gradient of Lyapunov Function
                    with tf.GradientTape(persistent=True) as tape1:
                        l = self.model_l(i_state_l,training = True)   # Compute Lyapunov t=i
                        l0 = self.model_l(i_target)   # Compute Lyapunov t=infty
                        V = tf.add(smoothRelu(tf.subtract(l,l0)),0.5*tf.reshape(tf.reduce_sum(tf.pow(i_state_l,2),axis=1),(batchsize,1)))
                    dV = tape1.gradient(V,i_state_l)

                    # Calculate loss
                    Lyloss = lyapunovstabilityloss.call(V,dV,DX,alpha)
                    Closs, u = controlbehaviourloss.call(c,V,lam1,lam2,dV,DX,alpha)

                
                batchValid = mirrStep(tf.reduce_sum(tf.nn.relu(tf.add(tf.keras.backend.batch_dot(dV,DX),tf.multiply(alpha,V)))))
                violatingStability1 += tf.reduce_sum(tf.keras.backend.cast_to_floatx(tf.greater_equal(tf.add(tf.keras.backend.batch_dot(dV,DX),tf.multiply(alpha,V)),0)))
                violatingStability2 += tf.reduce_sum(tf.keras.backend.cast_to_floatx(tf.greater(tf.keras.backend.batch_dot(dV,DX),0)))
                
                # Update NN
                Lygrads = tape.gradient(Lyloss,self.model_c.trainable_variables+self.model_l.trainable_variables)
                self.optimiser.apply_gradients(zip(Lygrads,self.model_c.trainable_variables+self.model_l.trainable_variables))

                if tf.equal(batchValid,1.0):
                    Cgrads = tape.gradient(Closs,self.model_c.trainable_variables)
                    self.optimiser.apply_gradients(zip(Cgrads,self.model_c.trainable_variables))
                
                lossHistory = np.append(lossHistory,np.array([[tf.reduce_sum(Lyloss).numpy()],[tf.reduce_sum(u).numpy()]]).reshape(3,1),axis = 1)

                if i%(steps/(2*self.refinement+1)) == 0:
                    print('='*int((i+1)/(steps/(2*self.refinement+1))),' '*(int(steps/(2*self.refinement+1))-int((i+1)/(steps/(2*self.refinement+1)))),tf.reduce_sum(Lyloss),tf.reduce_sum(u),violatingStability1,violatingStability2,'              ',end ='\r')
            if violatingStability1 == 1 and lam1 == 0 and lam2 == 0:
                break
        return lossHistory
        


    def controlNN(self,state):
        '''
        controlNN:
        ==========
        returns the control output for a specific state. 
        '''
        state = tf.reshape(tf.convert_to_tensor(state),(1,2))
        target = tf.reshape(tf.convert_to_tensor(np.array([0,0])),(1,2))
        c = self.model_c([state,target])
        return c
    
    def lyapunovNN(self,state):
        '''
        lyapunovNN:
        ===========
        returns the Lyapunov function value for a specific state. 
        '''
        state = tf.reshape(tf.convert_to_tensor(state),(1,2))
        l = self.model_l([state])
        return l
    
    def VisualiseController(self):
        '''
        VisualuseController:
        ====================
        Visualises the controller and Lyapunov properties of the current networks. 
        '''
        [phi,dphi] = np.meshgrid(np.linspace(-np.pi,np.pi,51),np.linspace(-7,7,51))

        Lyapunov = np.zeros(np.shape(phi))
        LyapunovNN = np.zeros(np.shape(phi))
        dLyapunov = np.zeros(np.shape(phi))
        Controller = np.zeros(np.shape(phi))
        DXH = np.zeros((len(phi),len(phi),2))
        DV = np.zeros((len(phi),len(phi),2))

        for i in range(51):
            for j in range(51):
                i_state = tf.reshape(tf.convert_to_tensor(np.array([phi[i,j],dphi[i,j]]),dtype=tf.float32),(1,2))
                i_target = tf.reshape(tf.convert_to_tensor(np.array([0,0]),dtype=tf.float32),(1,2))

                i_state_l = tf.Variable(i_state,dtype=tf.float32)

                c = self.model_c([i_state,i_target])

                with tf.GradientTape(persistent=True) as tape:
                    l = self.model_l(i_state_l) # Compute Lyapunov t=i
                    l0 = self.model_l(i_target) # Compute Lyapunov t=i

                    V = tf.add(smoothRelu(tf.subtract(l,l0)),0.5*tf.reduce_sum(tf.pow(i_state_l,2)))
                dV = tape.gradient(V,i_state_l)

                # First Step of Heun
                dx = self.dynamics(i_state,c)
                state = tf.add(i_state,tf.multiply(0.1,dx))

                ch = self.model_c([state,i_target])
                #  Second step of Heun
                dxh = self.dynamics(state,ch)

                DX = tf.multiply(0.5,tf.add(dx,dxh))

                
                Lyapunov[i,j] = V
                LyapunovNN[i,j] = l
                dLyapunov[i,j] = tf.matmul(dV,tf.transpose(DX))
                Controller[i,j] = c
                DXH[i,j,:] = DX
                DV[i,j,:] = dV

        fig = plt.figure(figsize=(18,5))
        ax1 = fig.add_subplot(141,projection='3d')
        ax1.plot_surface(phi,dphi,Lyapunov,cmap=cm.coolwarm)
        ax1.set_xlabel('$\phi$')
        ax1.set_ylabel('$\dot \phi$')
        plt.title('$V(x)$')

        ax2 = fig.add_subplot(142,projection='3d')
        ax2.plot_surface(phi,dphi,Controller,cmap=cm.coolwarm)
        ax2.set_xlabel('$\phi$')
        ax2.set_ylabel('$\dot \phi$')
        plt.title('$G_C(x)$')

        ax3= fig.add_subplot(143,projection='3d')
        ax3.plot_surface(phi,dphi,dLyapunov,cmap=cm.coolwarm)
        ax3.set_xlabel('$\phi$')
        ax3.set_ylabel('$\dot \phi$')
        plt.title(''r'$\nabla V^T f(x,u(x))$')        

        ax4= fig.add_subplot(144,projection='3d')
        ax4.plot_surface(phi,dphi,LyapunovNN,cmap=cm.coolwarm)
        ax4.set_xlabel('$\phi$')
        ax4.set_ylabel('$\dot \phi$')
        plt.title('$G_V(x)$')        
        plt.show()

        plt.figure(figsize=(18,5))
        plt.subplot(1,4,1)
        plt.imshow(Lyapunov,cmap='coolwarm',extent=[phi[0,0],phi[0,-1],dphi[0,0],dphi[-1,0]],origin='lower',alpha=0.6)
        plt.colorbar()
        plt.streamplot(phi,dphi,DXH[:,:,0],DXH[:,:,1],color='gray')
        plt.xlabel('$\phi$')
        plt.ylabel('$\dot \phi$')
        plt.title('$V(x)$')
        plt.subplot(1,4,2)
        plt.imshow(Controller,cmap='coolwarm',extent=[phi[0,0],phi[0,-1],dphi[0,0],dphi[-1,0]],origin='lower',alpha=0.6)
        plt.colorbar()
        plt.streamplot(phi,dphi,DXH[:,:,0],DXH[:,:,1],color='gray')
        plt.xlabel('$\phi$')
        plt.ylabel('$\dot \phi$')
        plt.title('$G_C(x)$')
        plt.subplot(1,4,3)
        plt.imshow(dLyapunov,cmap='coolwarm',extent=[phi[0,0],phi[0,-1],dphi[0,0],dphi[-1,0]],origin='lower',alpha=0.6)
        plt.colorbar()
        plt.contour(phi,dphi,dLyapunov,levels=[0])
        plt.streamplot(phi,dphi,DXH[:,:,0],DXH[:,:,1],color='gray')
        plt.xlabel('$\phi$')
        plt.ylabel('$\dot \phi$')
        plt.title(''r'$\nabla V^T f(x,u(x))$')
        plt.subplot(1,4,4)
        plt.imshow(LyapunovNN,cmap='coolwarm',extent=[phi[0,0],phi[0,-1],dphi[0,0],dphi[-1,0]],origin='lower',alpha=0.6)
        plt.colorbar()
        plt.streamplot(phi,dphi,DXH[:,:,0],DXH[:,:,1],color='gray')
        plt.xlabel('$\phi$')
        plt.ylabel('$\dot \phi$')
        plt.title('$G_V(x)$')
        plt.show()