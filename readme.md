# CNN Denoise usage

## Requirements
Python3 in order to create virtual environment and import libraries from requirements.txt

## Execution
Add noise to your image Dataset
```bash
python3 add-noise.py
```
Train the network
```bash
python3 ai_agent.py
```
MSE is shown in terminal for each Epoch and images are saved in 'predicted' folder.

To evaluate using trained network:
```bash
python3 evaluate.py
```
