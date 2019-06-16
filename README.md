# Machine Perceprion 2019
## Project 2: 3D Human Pose Estimation

### Authors:
	- Mounir Amrani       mamrani@student.ethz.ch
	- Srikanth Sarma      sgurram@student.ethz.ch

Requirements:
	- Data must be in the current folder or the path `/cluster/project/infk/hilliges/lectures/mp19/project2/` at leonhard cluster should be accesible
	- Data should be organized in two folders namely `images` and `annot`.
	- `images` should contain the images while `annot` should contain `train.h5` (with 2d and 3d pose) and image file list (`train_images.txt` and `valid_images.txt`)
	- All python dependencies must be installed using `setup.py`

**Training:**

To train model x (hourglass3D, hourglass2D or simple_baseline), run the command:
	`python3 train_x.py`

Training parameters can be changes using command line arguments. For (more details on available arguments, run the command:
	`python3 train_x.py --help`

**Testing:**

Once the necessary models have been trained and the model checkpoints have been saved at respective locations, we can perform predictions on test data.
To use model (hourglass3D, hourglass2D or simple_baseline) for performing prediction on valid images, run the command:
	`python3 test_x.py`

In order to compute the average of multiple predictions, create a folder `./to_merge` and add the prediction files (.csv.gz files) to the folder and run the command;
	`python3 merge_predictions.py`