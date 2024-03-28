import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg, rotation_matrix_z
import random


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    parser.add_argument('--rotate_pc', type=bool, default=False, help='Add noise to dataset or not')
    parser.add_argument('--std', type=float, default=0.0, help='Define STD')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Constants
    ROTATION_ANGLE = 45

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(N=args.num_points).to(args.device)
    model.cuda()

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    test_dataloader = get_data_loader(args=args, train=False)
    total_samples = 0
    correct_samples = 0

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    preds_labels = []

    #Add rotation to the dataset
    if args.rotate_pc:
        rot_matrix = rotation_matrix_z(ROTATION_ANGLE)
        # noise=(torch.randn(test_dataloader.dataset.data.size()) * std_dev + mean).to(args.device)
        test_dataloader.dataset.data = torch.matmul(test_dataloader.dataset.data.to(args.device).to(torch.float), rot_matrix.to(args.device))

    net_accuracy=[]
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds[:,ind].to(args.device)
            labels = labels[:,ind].to(args.device).to(torch.long)

            # Forward pass
            predictions = model(point_clouds)
            pred_label = torch.argmax(predictions,-1, keepdim=False)
            correct_samples += pred_label.eq(labels.data).cpu().sum().item()
            total_samples += labels.view([-1,1]).size()[0]
            preds_labels.append(pred_label)

        accuracy = correct_samples / total_samples
        print(f"Accuracy: {accuracy:.4f}")


    preds_labels = torch.cat(preds_labels).detach().cpu()

    # Visualizing the predictions for 5 objects below and above thresholds

    # above_thresh = 0
    # below_thresh = 0

    # for i in range(601):
    #     verts = test_dataloader.dataset.data[i,ind].detach().cpu()
    #     gt_cls = test_dataloader.dataset.label[i,ind].to(torch.long).detach().cpu()
    #     pred_cls = preds_labels[i].detach().cpu().data
    #     # print(gt_cls.shape,pred_cls.shape,verts.shape)

    #     correct_samples += pred_cls.eq(gt_cls.data).cpu().sum().item()
    #     total_samples += gt_cls.view([-1,1]).size()[0]
    #     accuracy = correct_samples / total_samples
    #     print(f"Accuracy: {accuracy:.4f}")
    #     # print(gt_cls, pred_cls)
    #     if accuracy < 0.6 and below_thresh < 5:
    #         path = f"output/seg/below_thresh_seg_gt_idx_{i}.gif"
    #         viz_seg(verts, gt_cls,path, args.device)
    #         path=f"output/seg/below_thresh_seg_pred_idx_{i}_acc_{accuracy:.3f}.gif"
    #         viz_seg(verts,pred_cls,path,args.device)
    #         below_thresh += 1

    #     if accuracy >= 0.9 and above_thresh < 5:
    #         path = f"output/seg/above_thresh_seg_gt_idx_{i}.gif"
    #         viz_seg(verts, gt_cls,path, args.device)
    #         path=f"output/seg/above_thresh_seg_pred_idx_{i}_acc_{accuracy:.3f}.gif"
    #         viz_seg(verts,pred_cls,path,args.device)
    #         above_thresh += 1

    #     if above_thresh >= 5 and below_thresh >= 5:
    #         break

    #     total_samples = 0
    #     correct_samples = 0

# Visualizing the predictions for Two objects for Noise and Num_points
total_samples = 0
correct_samples = 0

vals = (50,100)

for i in vals:
    verts = test_dataloader.dataset.data[i,ind].detach().cpu()
    gt_cls = test_dataloader.dataset.label[i,ind].to(torch.long).detach().cpu()
    pred_cls = preds_labels[i].detach().cpu().data

    correct_samples += pred_cls.eq(gt_cls.data).cpu().sum().item()
    total_samples += gt_cls.view([-1,1]).size()[0]
    accuracy = correct_samples / total_samples
    print("\n",f"Accuracy: {accuracy:.4f}")

    if args.rotate_pc:
        path = f"output/seg/seg_gt_idx_{i}_numpts_{args.num_points}_rotated_by_{ROTATION_ANGLE}_degrees.gif"
        viz_seg(verts, gt_cls,path, args.device)

        path=f"output/seg/seg_pred_idx_{i}_numpts_{args.num_points}_rotated_by_{ROTATION_ANGLE}_degrees_acc_{accuracy}.gif"
        viz_seg(verts,pred_cls,path,args.device)
        print(i)

    else:
        path = f"output/seg/seg_gt_idx_{i}_numpts_{args.num_points}.gif"
        viz_seg(verts, gt_cls,path, args.device)

        path=f"output/seg/seg_pred_idx_{i}_numpts_{args.num_points}_acc_{accuracy}.gif"
        viz_seg(verts,pred_cls,path,args.device)
        print(i)
