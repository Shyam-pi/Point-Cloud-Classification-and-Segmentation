import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls, rotation_matrix_z
from data_loader import get_data_loader
import random

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--rotate_pc', type=bool, default=False, help='Add noise to dataset or not')
    parser.add_argument('--std', type=float, default=0.0, help='Define STD')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')


    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Constants
    ROTATION_ANGLE = 45

    # ------ TO DO: Initialize Model for Classification Task ------
    model = model = cls_model(N=args.num_points)
    model.cuda()

    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))
    test_dataloader = get_data_loader(args=args, train=False)

    #Add rotation to the dataset
    if args.rotate_pc:
        rot_matrix = rotation_matrix_z(ROTATION_ANGLE)
        # noise=(torch.randn(test_dataloader.dataset.data.size()) * std_dev + mean).to(args.device)
        test_dataloader.dataset.data = torch.matmul(test_dataloader.dataset.data.to(args.device).to(torch.float), rot_matrix.to(args.device))

    #predict the labels
    total_samples = 0
    correct_samples = 0
    ind = np.random.choice(10000,args.num_points, replace=False)
    preds_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            point_clouds, labels = batch
            point_clouds = point_clouds[:,ind].to(args.device)
            labels = labels.to(args.device).to(torch.long)

            # Forward pass
            predictions = model(point_clouds)
            pred_label = torch.argmax(predictions, 1)
            correct_samples += pred_label.eq(labels.data).sum().cpu().item()
            total_samples += labels.size()[0]
            preds_labels.append(pred_label)
            # total_samples+=1
        accuracy = correct_samples / total_samples
        print(f"Accuracy: {accuracy:.4f}")
    preds_labels = torch.cat(preds_labels).detach().cpu()
    fail_inds = torch.argwhere(preds_labels != test_dataloader.dataset.label).flatten().numpy()
    success_inds = torch.argwhere(preds_labels == test_dataloader.dataset.label).flatten().numpy()

    # #visualize the wrong and correct predictions

    # random_values = np.random.choice(fail_inds, 15, replace=False)
    # # print(fail_inds)
    # for i in random_values:
    #     verts = test_dataloader.dataset.data[i, ind]
    #     gt_cls = test_dataloader.dataset.label[i].to(torch.long).detach().cpu().data
    #     pred_cls = preds_labels[i].detach().cpu().data
    #     # print(gt_cls, pred_cls)
    #     path = f"output/cls/wrong_idx_{i}_with_gt_{gt_cls}_pred_{pred_cls}.gif"
    #     viz_cls(verts, path, args.device)

    # random_values = np.random.choice(success_inds, 15, replace=False)
    # # print(random_values)
    # for i in random_values:
    #     verts = test_dataloader.dataset.data[i, ind]
    #     gt_cls = test_dataloader.dataset.label[i].to(torch.long).detach().cpu().data
    #     pred_cls = preds_labels[i].detach().cpu().data
    #     # print(gt_cls, pred_cls)
    #     path = f"output/cls/success_idx_ {i}_gt_{gt_cls}_pred_{pred_cls}.gif"
    #     viz_cls(verts, path, args.device)

# visualize Predictions for Rotation and Variation of number of points
vals = (50,100)

if args.rotate_pc:
    for i in vals:
        verts = test_dataloader.dataset.data[i, ind]
        # print(verts.shape)
        gt_cls = test_dataloader.dataset.label[i].to(torch.long).detach().cpu().data
        pred_cls = preds_labels[i].detach().cpu().data
        # print(gt_cls, pred_cls)
        # path = f"output/cls/success_idx_ {i}_gt_{gt_cls}_pred_{pred_cls}_{args.std}.gif"
        path = f"output/cls/idx_ {i}_gt_{gt_cls}_pred_{pred_cls}_num_pts_{args.num_points}_rotated_by_{ROTATION_ANGLE}_degrees.gif"
        viz_cls(verts, path, args.device)

else:
    for i in vals:
        verts = test_dataloader.dataset.data[i, ind]
        # print(verts.shape)
        gt_cls = test_dataloader.dataset.label[i].to(torch.long).detach().cpu().data
        pred_cls = preds_labels[i].detach().cpu().data
        # print(gt_cls, pred_cls)
        # path = f"output/cls/success_idx_ {i}_gt_{gt_cls}_pred_{pred_cls}_{args.std}.gif"
        path = f"output/cls/idx_ {i}_gt_{gt_cls}_pred_{pred_cls}_num_pts_{args.num_points}.gif"
        viz_cls(verts, path, args.device)