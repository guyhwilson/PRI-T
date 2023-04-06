# Probabilistic Retrospective Inference of Targets (PRI-T)

This repo contains code for running PRI-T, an unsupervised cursor BCI retraining procedure described in the paper ["Long-term unsupervised recalibration of cursor BCIs"](https://www.biorxiv.org/content/10.1101/2023.02.03.527022v1).

<img src="/resources/PRIT_example.gif" width=50% height=50%>

## Environment Setup

From this topmost level of the repo, use pip to install dependencies and the PRI-T package.

```
pip install .
```

Navigate over to `examples/` to check out some demo Jupyter notebooks. You should be able to run these all if the repo was installed correctly.


## Quick start 


To run PRI-T, first generate a screen discretization, state transition matrix, and observation model:

```
from PRIT.prit_utils import generateTargetGrid, generateTransitionMatrix
from PRIT.prit import HMMRecalibration

gridSize  = 20      # number of rows/columns when discretizing screen
stayProb  = 0.999  # probability that target just stays where it is at any given timestep
vmKappa   = 2       # precision parameter for the von mises distribution.
logistic_inflection = 0.  # controls von mises variance fine-tuning
logistic_exp = 32.        # controls von mises variance fine-tuning

#------------------

# finetunes the observation model's variance; set to None to ignore
adjustKappa = lambda x: 1 / (1 + np.exp(-1 * (x - logistic_inflection) * logistic_exp)) 

# this assumes screen is resized to be 1 x 1; can pass arguments to generateTargetGrid() to modify
nStates                 = gridSize**2
targLocs                = generateTargetGrid(gridSize = gridSize, is_simulated=True)
stateTrans, pStateStart = generateTransitionMatrix(gridSize = gridSize, stayProb = stayProb)

```

Next build a PRI-T HMM:

```
# create a PRI-T HMM object
hmm = HMMRecalibration(stateTrans, targLocs, pStateStart, vmKappa, adjustKappa = adjustKappa)
```

You can now infer target locations from cursor behavior. Supposing you have `time x 2` numpy arrays `cursorPos` and `cursorVel`, simply run:

```
targStates, pTargState = hmm.predict([cursorPos], [cursorVel])
inferredTargLoc        = hmm.targLocs[targStates.astype('int').flatten(),:]
```
`inferredTargLoc` holds the most likely sequence of targets across time ("Viterbi" sequence). `pTargState` is a `time x nStates` matrix containing the probabilities of the target being in each location across all timesteps. For a more in-depth overview, check out the example notebooks mentioned earlier. Extra details include click integration and speed optimization.


## Recommendations

Recommended defaults:

- `gridSize`: values >= 20 are fine
- `stayProb`: 0.99 - 0.9999 generally good
- `vmKappa` : 2 - 8 usually good; this is maybe the most important to tune
- `logistic_inflection` : maybe ~10% of screen width is fine
- `logistic_exponent` : slope of logistic curve (20-40 seems okay...?? least confident about this one)

