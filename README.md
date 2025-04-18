# CAS_737_Project
#### Evaluating Video-based Pose Estimation using MediaPipe and MoveNet

| Joint Name      | MoveNet Index | MediaPipe Index |
|-----------------|---------------|-----------------|
| Nose            | 0             | 0               |
| Left Eye        | 1             | 2               |
| Right Eye       | 2             | 5               |
| Left Ear        | 3             | 7               |
| Right Ear       | 4             | 8               |
| Left Shoulder   | 5             | 11              |
| Right Shoulder  | 6             | 12              |
| Left Elbow      | 7             | 13              |
| Right Elbow     | 8             | 14              |
| Left Wrist      | 9             | 15              |
| Right Wrist     | 10            | 16              |
| Left Hip        | 11            | 23              |
| Right Hip       | 12            | 24              |
| Left Knee       | 13            | 25              |
| Right Knee      | 14            | 26              |
| Left Ankle      | 15            | 27              |
| Right Ankle     | 16            | 28              |



#### Inference Time Comparison (seconds)

| Scenarios                     | MediaPipe | MoveNet |
| ----------------------------- | --------- | ------- |
| squat_angle_good_lighting     | 21.48     | 13.72   |
| squat_side_good_lighting      | 22.59     | 15.00   |
| squat_front_good_lighting     | 19.77     | 10.53   |
| squat_occulsion_good_lighting | 18.99     | 10.81   |
| squat_front_low_exposure      | 23.94     | 10.51   |
| squat_front_dim_lighting      | 22.12     | 11.24   |

*Note: MoveNet model is movenet_lightning_f16, MediaPipe model is Version: 0.10.21*



#### Aggregate Pose Estimation Error: Euclidean Distance Across Frames (MoveNet vs MediaPipe vs Ground Truth) (pixels)

| Scenarios                     | MediaPipe | MoveNet  |
| :---------------------------- | :-------- | :------- |
| squat_angle_good_lighting     | 35.6915   | 83.9724  |
| squat_side_good_lighting      | 48.7997   | 70.3637  |
| squat_front_good_lighting     | 23.8966   | 59.1432  |
| squat_occulsion_good_lighting | 47.9748   | 120.8132 |
| squat_front_low_exposure      | 31.3774   | 180.4030 |
| squat_front_dim_lighting      | 35.6595   | 80.4206  |
|                               |           |          |



#### Pose Estimation Consistency: Variance of Errors Across Frames (MoveNet vs MediaPipe vs Ground Truth) (pixels)

| Scenarios                     | MediaPipe | MoveNet |
| :---------------------------- | :-------- | :------ |
| squat_angle_good_lighting     | 0.0015    | 0.0569  |
| squat_side_good_lighting      | 0.0081    | 0.0348  |
| squat_front_good_lighting     | 0.0012    | 0.0732  |
| squat_occulsion_good_lighting | 0.1604    | 0.9898  |
| squat_front_low_exposure      | 0.0015    | 1.9031  |
| squat_front_dim_lighting      | 0.0216    | 0.0534  |



#### Scenario-Based Variance in Pose Estimation Errors (MoveNet vs MediaPipe vs Ground Truth) (pixels)

| scenarios                                                    | MediaPipe | MoveNet   |
| :----------------------------------------------------------- | :-------- | :-------- |
| angles scenarios (squat_side_good_lighting, squat_angle_good_lighting, squat_front_good_lighting) | 103.46    | 103.08    |
| lightning_scenarios(squat_front_good_lighting, squat_front_low_exposure, squat_front_dim_lighting) | 23.629396 | 2794.7941 |
| difficult_scenarios(squat_side_good_lighting, squat_occulsion_good_lighting, squat_front_low_exposure) | 64.410076 | 2022.7493 |
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
