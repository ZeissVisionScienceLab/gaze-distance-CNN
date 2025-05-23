# Replay mode for depth extraction

This project is a replay mode of the experiment, where the scene depth data corresponding to the recorded eye tracking data is extracted after the experiment. The extraction and saving of each frame's depth data can not be done in parallel to the experiment, due to performance requirements. Running the code requires the recorded eye tracking data from the experiment, an exemplary file can be found in [resources/etdata](<resources/etdata>), which was recorded for the given training scene. 
