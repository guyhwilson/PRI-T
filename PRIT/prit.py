import numpy as np
import scipy, numba
import scipy.special
from numba import jit
import time 
from joblib import Parallel, delayed



class HMMRecalibration(object):
    """
    PRI-T HMM object. Initialize by defining the state transition matrix, observation models, and 
    state prior probabilities. 
    """
    
    def __init__(self, stateTransitions, targLocs, pStateStart, vmKappa, adjustKappa = None, getClickProb = None):
        '''Inputs are:

            stateTransitions (2D array) - transition probabilities; nStates x nStates
            targLocs (2D array)         - nStates x 2 array containing corresponding target locations for each state
            pStateStart (vector)        - starting probabilities for each state 
            vmKappa (float)             - precision parameter for Von Mises observation model
            adjustKappa (method)        - fxn for weighting kappa values based on distance; defaults to None
            getClickProb (method)       - a function (cursor_state) --> P(cursor_state | target) for all targets in targLocs;
                                          (default: None, which yields a "vanilla" PRI-T HMM without click model) '''
    
        self.stateTransitions = stateTransitions
        self.targLocs         = targLocs
        self.pStateStart      = pStateStart
        self.vmKappa          = vmKappa
    
        if adjustKappa is None:
            self.adjustKappa = lambda x: np.ones(x.shape)
        else:
            self.adjustKappa = adjustKappa
            
        self.getClickProb = getClickProb
        
        
        
    def _compute_dists_and_angles(self, cursorVel, cursorPos):
        '''Given cursor states across time, compute expected angle and distance with respect to 
           all possible target states. Inputs are:

                cursorVel (2D array)     - time x 2 array containing decoder outputs at each timepoint
                cursorPos (2D array)     - time x 2 array of cursor positions  '''
        
        observedAngle    = np.arctan2(cursorVel[:, 1], cursorVel[:, 0])
        tDists           = np.linalg.norm(self.targLocs - cursorPos[:, np.newaxis, :], axis = 2)
        normPosErr       = (self.targLocs[:, np.newaxis] - cursorPos) / tDists.T[:, :, np.newaxis]
        expectedAngle    = np.arctan2(normPosErr[:, :, 1], normPosErr[:, :, 0])
        
        return tDists, expectedAngle, observedAngle

            
    def get_posterior_prob(self, cursorVel, cursorPos, clickState = None):
        '''Given cursor position and velocity, returns P(targ | cursor info) for all possible targets. Inputs are:
           
                cursorVel (2D array)   - time x 2 array containing decoder outputs at each timepoint
                cursorPos (2D array)      - time x 2 array of cursor positions  
                clickState (1D array)     - binary indicator of whether or not user clicked'''
        
         # 1. compute distance from the cursor to each target, and expected angle for that target
        tDists, expectedAngle, observedAngle = self._compute_dists_and_angles(cursorVel, cursorPos)
        vmKappa_adjusted                     = self.vmKappa * self.adjustKappa(tDists)
        
        # 2. compute VM probability densities
        obsProbLog = (vmKappa_adjusted * np.cos(observedAngle - expectedAngle).T) - np.log(2*np.pi* scipy.special.i0(vmKappa_adjusted))
        
        # 3. Optionally compute bernoulli click (log) probabilities and add to posterior 
        if clickState is not None:
            if self.getClickProb is None:
                raise ValueError('HMM doesnt have click observation model.')
            else:
                probClick    = self.getClickProb(tDists)
                obsProbLog  += np.log(clickState[:, None] * probClick + ((1 - clickState)[:, None] * (1 - probClick)))
        
        return obsProbLog

        
        
    def viterbi_search(self, cursorVel, cursorPos, clickSignal = None, verbose = False):
        '''Run viterbi algorithm to find most likely sequence of target states given the cursor position and decoder outputs. Inputs are:

            cursorVel (2D array)     - time x 2 array containing decoder outputs at each timepoint
            cursorPos (2D array)        - time x 2 array of cursor positions  '''
        
        numStates    = len(self.stateTransitions)
        L            = cursorVel.shape[0]
        currentState = np.zeros((L, ))

        # work in log space to avoid numerical issues
        logTR = np.log(self.stateTransitions)
        v     = np.log(self.pStateStart)

        # Precompute p(obs | latent) for all timesteps and states: 
        vmProbLog = self.get_posterior_prob(cursorVel, cursorPos, clickSignal)
        
        # loop through the model;  von mises emissions probabilities
        pTR, v = forward_pass(v.flatten(), logTR, vmProbLog)  # most time intensive part - use numba to get ~2x speedup
       
        # decide which of the final states is most probable
        finalState = np.argmax(v)
        logP       = v[finalState]

        # Now back trace through the model
        currentState[L - 1] = finalState
        for count in reversed(range(0, L - 1)):
            currentState[count] = pTR[int(currentState[count + 1]), count + 1]
            if currentState[count] == 0 & verbose == True:
                print('stats:hmmviterbi:ZeroTransitionProbability', currentState[ count + 1 ])

        return currentState, logP
    
        
    def decode(self, cursorVel, cursorPos, clickSignal = None, verbose = False):
        '''Run forward-backward algorithm to find marginal probabilities of hidden states at each timestep (given observed data). 
        Inputs are:

                cursorVel (2D array)   - time x 2 array containing decoder outputs at each timepoint
                cursorPos (2D array)   - time x 2 array of cursor positions  
                clickSignal (1D array) - time x 1 indicator of whether or not user clicked
        
        Returns:
            
            pStates (2D array) - time x nStates of occupation probabilities
            pSeq (float)       - log probability of observed data
        '''
        
        numStates = len(self.stateTransitions)
        L         = cursorVel.shape[0] + 1  # add extra symbols to start to make algorithm cleaner at f0 and b0

        # introduce scaling factors for stability
        fs      = np.zeros((numStates,L))
        fs[:,0] = self.pStateStart.squeeze()
        s       = np.zeros((L,))
        s[0]    = 1

        # Precompute some values for speedup: 
        vmProb      = np.exp(self.get_posterior_prob(cursorVel, cursorPos, clickSignal))
        T_transpose = self.stateTransitions.T
        
        for count in range(1, L):
            fs[:,count]   = vmProb[count - 1, :] * (T_transpose.dot(fs[:, count-1]))
            s[count]      =  np.sum(fs[:,count])
            fs[:,count]  /=  s[count]

        bs = np.ones((numStates,L))
        for count in reversed(range(0, L - 1)):
            probWeightBS = bs[:,count + 1] * vmProb[count, :]
            tmp          = self.stateTransitions.dot(probWeightBS)
            bs[:,count]  = tmp * (1/s[count+1])

        pSeq    = np.sum(np.log(s))
        pStates = fs * bs
        pStates = pStates[:, 1:].T # get rid of the column that we stuck in to deal with the f0 and b0 

        return pStates, pSeq
            
        
    def predict(self, cursorPos, cursorVel, clickSignal = None, parallel = False):
        '''Run PRI-T prediction on retrospective data and return Viterbi sequence and 
           occupation probabilities. Inputs are:
           
               cursorPos (list of 2D arrays)   - entries are time x 2 of cursor positions
               cursorVel (list of 2D arrays)   - entries are time x 2 of cursor velocities 
               clickSignal (list of 1D arrays) - entries are sequences of click states, or None (default)
               
          Returns:
              
               viterbi_seq (2D array)      - time x 1 of target states (viterbi sequence)
               occupation_probs (2D array) - time x nStates of occupation probabilities 
        '''
        
        if clickSignal is None:
            clickSignal = [None] * len(cursorPos)
        
        if parallel:
            chunk_inferences = Parallel(n_jobs=-1, verbose = 0)(delayed(self._predict_individual)(*arg) for arg in zip(cursorPos, cursorVel, clickSignal))

        else:
            chunk_inferences = list()
            for (block_pos, block_vel, block_click) in zip(cursorPos, cursorVel, clickSignal):
                targs, pTargs = self._predict_individual(block_pos, block_vel, block_click)
                chunk_inferences.append([targs, pTargs])

        viterbi_seq      = np.concatenate([x[0] for x in chunk_inferences], axis = 0)
        occupation_probs = np.concatenate([x[1] for x in chunk_inferences], axis = 0)
                
        return viterbi_seq, occupation_probs
        
        
    def _predict_individual(self, cursorPos, cursorVel, clickSignal):
        '''Helper function called by predict() on individual timestretches. Inputs are:
        
            cursorPos (2D array) - time x 2 of cursor positions
            cursorVel (2D array) - time x 2 of cursor velocities 
            clickSignal (1D array or None) - sequence of click states 
            
           Returns:
               
               targs (2D array)  - time x 1 of target states (viterbi sequence)
               pTargs (2D array) - time x nStates of occupation probabilities '''
             
        targs, vp    = self.viterbi_search(cursorVel, cursorPos, clickSignal)
        pTargs, _    = self.decode(cursorVel, cursorPos, clickSignal)
        
        return targs, pTargs
        
        

    def recalibrate(self, decoder, neural, cursorPos, cursorVel = None, clickSignal = None, 
                    probThreshold = 'probWeighted', parallel = False):
        '''Retrain sklearn-style velocity decoder with inferred targets. Inputs are:

            decoder (sklearn object)      - decoder to use
            neural (list of 2D arrays)    - entries are time x n_channels arrays of neural activity 
            cursorPos (list of 2D arrays) - entries are time x 2 arrays of cursor positions
            cursorVel (list of 2D arrays, or None)   - entries are time x 2 arrays of cursor velocities
            clickSignal (list of 1D arrays, or None) - entries are sequences of click states
            
            probThreshold (float or str)  - threshold for subselecting high certainty regions (only where best
                                           guess > probThreshold); can also use mode "probWeighted" to weight
                                           by square of HMM's most likely state probability. Default: WLS weighting
            parallel (Bool)               - toggle processing data in parallel  
            
            Returns:
                
                decoder (sklearn object) - recalibrated decoder '''

        assert isinstance(probThreshold, float) or probThreshold == 'probWeighted', "Invalid probThreshold value. Check input."
        
        if cursorVel is None:
            cursorVel = [decoder.predict(block_neural) for block_neural in neural]

        neural_flattened       = np.concatenate(neural)
        cursorPos_flattened    = np.concatenate(cursorPos)
        
        # get concatenated Viterbi sequence across timestretches as well as occupation probs
        targStates, pTargState = self.predict(cursorPos, cursorVel, clickSignal, parallel = parallel)

        maxProb         = np.max(pTargState, axis = 0)              
        inferredTargLoc = self.targLocs[targStates.astype('int').flatten(), :]  # predicted target locations at each timepoint
        inferredPosErr  = inferredTargLoc - cursorPos_flattened  # inferred point-at-target signals
            
        # use high certainty time periods for recalibration:
        if isinstance(probThreshold, float):
            highProbIdx = np.where(maxProb > probThreshold)[0]  
            while len(highProbIdx) == 0: # if no valid data points, decrement threshold 
                probThreshold -= 0.1
                highProbIdx    = np.where(maxProb > probThreshold)[0]
                print('ProbThreshold too high. Lowering by 0.1')    
            decoder.fit(neural_flattened[highProbIdx, :], inferredPosErr[highProbIdx, :])

        # use all timepoints in weighted least squares, weighting is square of maxProb 
        elif probThreshold == 'probWeighted':
            decoder.fit(neural_flattened, inferredPosErr, maxProb**2)
                
        else:
            raise ValueError('<probThreshold> argument not recognized.')
            

        return decoder
    
    

@jit(nopython = True)
def forward_pass(v, logTR, vmProbLog):
    numStates = logTR.shape[0]
    L         = vmProbLog.shape[0]
    
    tmpV   = np.zeros((numStates, numStates))
    pTR    = np.zeros((numStates, L))
    vOld   = v
    maxVal = np.zeros((numStates,))
    maxIdx = np.zeros((numStates,))
    
    for count in range(L):
        for i in range(numStates):
            tmpV[i, :] = vOld + logTR[i, :]
            maxIdx[i]  = np.argmax(tmpV[i, :])
            maxVal[i]  = tmpV[i, int(maxIdx[i])]
        
        pTR[:,count] = maxIdx
        v            = vmProbLog[count, :] + maxVal
        vOld         = v
    
    return pTR, v
        
        
@jit(nopython = True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result