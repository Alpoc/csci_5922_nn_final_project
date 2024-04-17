This program is an suite of scripts for creating a supervised training model for the game BeamNG.drive.
Most of the functionality is hidden behind variables in main.

### To Train
- ensure that the file location in config.py is correct.
- conda activate tf
- python3 train.py

- nice little article for getting TF GPU working
  - https://medium.com/nerd-for-tech/installing-tensorflow-with-gpu-acceleration-on-linux-f3f55dd15a9


### Recording game sessions
To record game sessions and run inference you'll need to ensure that the screen is in the
correct location. Both run_inference file and  game_session_recorder expect the screen to be on the
second screen in the top left corner. With run_inference pressing "f8" will display an image of
what the camera is seeing. An equivalent in linux is still needed.


### My conda setup. above it outdated
- `conda create --name tf`
- `conda install tensorflow-gpu -c conda-forge -y`
- `python3 -m pip install --upgrade Pillow`
- `pip intstall -r requirements.txt`

### BeamNG.drive setup
- set second monitor to 720p
- launch BeamNG.drive and move to second monitor. In windowed mode
- Load hirochi speedway on the figure 8 track. 
  - backup to the center of the track.
- Hit the "6" key to change camera mode to "Chase"
- "f9" to start recording. "f9" or "esc" to end

- kill -9 $(pgrep -f "run_inference")

### Training
- config contains the some settings that need to be set such as training data location
- `conda activate tf`
- `python3 train.py`

### Gathering Training data
- See BeamNG.drive setup
- If on windows press
  - 'f9' to start recording
  - 'f9' or 'esc' to end recording
  
- `python3 game_session_recorder.py`

### Running the model on the game
- `f9` to start the program. You'll need to kill it manually
- `f8` to preview frame (Does not work on Linux) you have to manually change to true in code. 
- `python3 run_inference.py` or `python3 run_inference_linux.py`
