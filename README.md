# Machine Perceprion 2019
## Project 2: 3D Human Pose Estimation

### Authors:
	- Mounir Amrani       mamrani@student.ethz.ch
	- Srikanth Sarma      sgurram@student.ethz.ch

Requirements:

	- Data must be in the current folder or the path `/cluster/project/infk/hilliges/lectures/mp19/project2/` at leonhard cluster should be accesible.

	- Data should be organized in two folders namely `images` and `annot`.

	- `images` should contain the images while `annot` should contain `train.h5` (with 2d and 3d pose) and image file list (`train_images.txt` and `valid_images.txt`)

	- All python dependencies must be installed using `setup.py`

**Training 2D hourglass:**
To train our best performing 2D hourglass model, simply run:
```
python3 train_hourglass2D.py
```
The model will train for 2 epochs, with `batch_size=4`, `learning_rate=2.5e-4`, `nb_stacks=8` and produce at the end of every epoch a csv file with 2D predictions on the test set in a folder of the form `./log_HG2D/YYYY.MM.DD-HH:MM:SS/predictions` where `YYYY.MM.DD-HH:MM:SS` is the year, month, day, hour, minutes and seconds when the code was run.
Each epoch takes around 5.5h to finish on the GTX 1080 of the Leonhard cluster (should expect it to be faster on the GTX 1080 Ti).
Testing takes around 10 min on the GTX 1080

**Training the Feed-Forward Neural Network (Simple Baseline):**
To train the simple baseline model, simply run:
```
python3 train_simple_baseline.py
```
The model will train for 200 epochs, with `batch_size=64`, `learning_rate=0.001`,  and save the model at `./log_SB/YYYY.MM.DD-HH:MM:SS/predictions` where `YYYY.MM.DD-HH:MM:SS` is the year, month, day, hour, minutes and seconds when the code was run.

The whole training process should take around 2h.

Once the training finished, you can transform the 2D poses predicted by the last experiment run of the 2D hourglass model by simply running:

```
python3 test_simple_baseline.py
```
The model will transform the 2D prediction in each csv file in the `predictions` folder of the log folder of the last 2D hourglass experiment available.

**Training 3D hourglass:**
To train our best performing 3D hourglass model, simply run:
```
python3 train_hourglass3D.py
```
The model will train for 2 epochs, with `batch_size=4`, `learning_rate=2.5e-4`, and produces at the end of every epoch a csv.gz file with 3D predictions on the test set in a folder of the form `./log_HG3D/YYYY.MM.DD-HH:MM:SS/predictions` where `YYYY.MM.DD-HH:MM:SS` is the year, month, day, hour, minutes and seconds when the code was run.
Each epoch takes around 7.5h to finish on the GTX 1080 Ti of the Leonhard cluster.
Testing takes around 35 min on the GTX 1080 Ti

To continue training for 2 more epochs, simply run:
```
python3 train_hourglass3D.py --continue_training
```

**Other Options**
Training parameters can be changes using command line arguments. For more details on available arguments, run one of the following commands command:
```
python3 train_hourglass2D.py --help
python3 train_hourglass2D.py --help
python3 train_simple_baseline.py --help
python3 test_simple_baseline.py --help
```
Make sure to test the simple baseline with the same parameters as the ones used for training

**Testing after training finishes**
You can use one of the following commands to predict the 2D/3D poses on the latest checkpoint of each model (please make sure 2D predictions are available before testing using the simple baseline as described above):
```
python3 test_hourglass2D.py
python3 test_hourglass2D.py
python3 test_simple_baseline.py
```

**Averaging:**

In order to compute the average of multiple predictions, create a folder `./to_merge` and add the prediction files (.csv.gz files) to the folder and run the command;
	`python3 merge_predictions.py`