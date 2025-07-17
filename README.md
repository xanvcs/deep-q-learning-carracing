# Deep Q-Learning for Autonomous Racing

## Overview
Implementation of a Deep Q-Learning agent that learns to navigate the CarRacing-v3 environment using CNNs for visual processing and achieving 900+ rewards.

## Key Features
- CNN-based visual processing (96x96 RGB → grayscale)
- Experience replay buffer with 100k capacity
- Epsilon-greedy exploration strategy
- Adaptive Moment Estimation (Adam) Optimizer that dynamically adjusted the learning rate based on its own performance.

## Results
- **Peak validation reward**: 940.50
- **Average validation reward**: 855.74 over 100 episodes
- **Improvement during training**: 1000 point increase from baseline (-54 to 946.30)
- **Training time**: 16.6 hours over 2000 episodes using 7900 XTX

## Architecture
- 3 Convolutional layers (32→64→64 filters)
- 2 Fully connected layers (512→5 neurons)
- Discrete action space (5 actions)

## Files
- `writeup.pdf` - Detailed technical report and analysis
- `test_model.ipynb` - Jupyter notebook to run and test the trained model
- `dqn_agent.py` - Main DQN implementation
- `model.ipynb` - Jupyter notebook to train model
- `final_model.pth` - Trained model file.

## Quick Start
Open and run `test_model.ipynb` to see the trained agent in action.

## Full Documentation
See `writeup.pdf` for complete implementation details, methodology, and analysis.