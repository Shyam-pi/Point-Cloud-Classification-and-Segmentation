# Point-Cloud-Classification-and-Segmentation

**Note**:  I completed this project as a part of the course CMSC848F - 3D Vision, Fall 2023, at the University of Maryland. The original tasks and description can be found at https://github.com/848f-3DVision/assignment4 

**Goals**: In this project we implement a PointNet based architecture for classification and segmentation with point clouds, sections 1 and 2 focus on implementing, training and testing models. Section 3 is about quantitatively analyzing the model's robustness.

**Results**: All the results and inferences from this project can be found in this webpage : https://shyam-pi.github.io/Point-Cloud-Classification-and-Segmentation/ 

## Data Preparation
Download zip file (~2GB) from https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing. Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

**General Note**: `models.py` is where all the model structures have been defined. `train.py` loads data, trains models, logs trajectories and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain script to evaluate model accuracy and visualize segmentation result. Feel free to modify any file as needed.

The following PointNet architecture (without the feature transform) is used as the base network for both these models:

![image](https://github.com/Shyam-pi/Point-Cloud-Classification-and-Segmentation/assets/57116285/94218436-a3c1-436a-9032-732221301ed4)


## 1. Classification Model
Implement the classification model in `models.py`.

- Input: points clouds from across 3 classes (chairs, vases and lamps objects)

- Output: probabibility distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Complete model initialization and prediction in `train.py` and `eval_cls.py`. Run `python train.py --task cls` to train the model, and `python eva_cls.py` for evaluation. Check out the arguments and feel free to modify them as you want.

On the webpage, one can find the following: 

- Test accuracy.

- Visualization of a few random test point clouds and their corresponding predicted classes.
 
## 2. Segmentation Model 
Implement the segmentation model in `models.py`.  

- Input: points of chair objects (6 semantic segmentation classes) 

- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Complete model initialization and prediction in `train.py` and `eval_seg.py`. Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gif's, one for ground truth and the other for model prediction. Check out the arguments and feel free to modify them as you want. In particular, you may want to specify `--i` and `--load_checkpoint` arguments in `eval_seg.py` to use your desired model checkpoint on a particular object.

On the webpage, one can find the following: 

- Test accuracy.

- Visualization of the segmentation results of at least 5 objects (including 2 bad predictions) with corresponding ground truth, and the prediction accuracy for each object, along with some interpretation for the results.
  
## 3. Robustness Analysis
Conducted 2 experiments to analyze the robustness of the learned model:
1. Rotate the input point clouds by certain degrees and report how much the accuracy falls
2. Input a different number of points points per object (modify `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py`)

On the webpage, for each experiment, one can find:

- Procedure for each of the experiments.
- Test accuracy and visualization on a few samples in each experiment.
- Some interpretation in a few sentences.


