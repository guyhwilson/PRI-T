import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def generateTargetGrid(gridSize, is_simulated = False, x_bounds = None, y_bounds = None):
    '''
    Discretize screen into equally spaced tiles. Inputs are:
    
        gridSize (int)         - number of rows/columns to discretize screen into
        is_simulated (Bool)    - whether or not using simulator data 
        x_bounds (float tuple) - float tuple containing x-axis boundaries for the screen
        y_bounds (float tuple) - float tuple containing y-axis boundaries for the screen
        
    Returns:
    
        targLocs (2D float array) - gridSize**2 x 2 of target positions
    '''
    
    assert np.logical_and(x_bounds == None, y_bounds == None), "<x_bounds> and <y_bounds> must be both None or tuples"
    assert np.logical_xor(is_simulated, x_bounds != None), "<is_simulated> and <bound> parameters cannot both be toggled"
    
    if is_simulated:
        # if using simulator we know the screen's coordinates are enclosed in 1 x 1 box
        x_bounds = [-0.5, 0.5]
        y_bounds = [-0.5, 0.5]
    
    X_loc,Y_loc = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], gridSize), np.linspace(y_bounds[0], y_bounds[1], gridSize))
    targLocs    = np.vstack([np.ravel(X_loc), np.ravel(Y_loc[:])]).T
    
    return targLocs


def generateTransitionMatrix(gridSize, stayProb):
    '''Generate unstructed transition matrix for target states. Inputs are:
    
        gridSize (int)   - number of rows/columns to discretize screen into
        stayProb (float) - value in [0, 1); probability of target staying put at a given timestep 
    
      Returns:
      
          stateTrans (2D float)  - nStates x nStates transition matrix
          pStateStart (1D array) - length nStates of prior probabilities 
    '''
    
    nStates     = gridSize**2
    stateTrans  = np.eye(nStates)*stayProb # Define the state transition matrix

    for x in range(nStates):
        idx                = np.setdiff1d(np.arange(nStates), x)
        stateTrans[x, idx] = (1-stayProb)/(nStates-1)

    pStateStart = np.zeros((nStates,1)) + 1/nStates
    
    return stateTrans, pStateStart


