'''
使用训练好的模型对图片进行预测，将input  predict mask  图片绘制出来，放在同一张图上对比
'''
import os
from PIL import Image
from datetime import datetime
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
import joblib
import torchmetrics
# 绘图
import matplotlib.pyplot as plt
from medpy import metric
import torch
import cv2

# 自己写的函数引用
from train import create_model
from my_dataset import DriveDataset
import transforms as T


from src import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")


def getDataset():
    class SegmentationPresetTrain:
        def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            min_size = int(0.5 * base_size)
            max_size = int(1.2 * base_size)

            trans = [T.RandomResize(min_size, max_size)]
            if hflip_prob > 0:
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            if vflip_prob > 0:
                trans.append(T.RandomVerticalFlip(vflip_prob))
            trans.extend([
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.transforms = T.Compose(trans)

        def __call__(self, img, target):
            return self.transforms(img, target)

    class SegmentationPresetEval:
        def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

        def __call__(self, img, target):
            return self.transforms(img, target)

    def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        base_size = 565
        crop_size = 480

        if train:
            return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
        else:
            return SegmentationPresetEval(mean=mean, std=std)

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    ds = DriveDataset(r'E:\服务器代码\DRIVE\unet',
                                  train=False,
                                  transforms=get_transform(train=False, mean=mean, std=std))




    test_dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, drop_last=True)  #

    return test_dataloader


def calculate_metric_percase(pred, gt):
    pred = list(pred.values())[0]
    gt[gt > 1] = 0
    # pred[pred > 0.5] = 1
    # pred[pred < 0.5] = 0
    pred = torch.max(pred, 1)[1]
    # gt[gt > 0.5] = 1
    # gt[gt < 0.5] = 0
    pred_ = torch.squeeze(pred).cpu().numpy()
    gt_ = torch.squeeze(gt).cpu().numpy()

    seg_inv, gt_inv = np.logical_not(pred_), np.logical_not(gt_)

    true_pos = float(np.logical_and(pred_, gt_).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pred_, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt_).sum()

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    # 它的最大值是1，最小值是0，值越大意味着模型越好。
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    # IoU正例 IoU = TP / (TP + FP + FN)
    iou = true_pos / (true_pos + false_neg + false_pos + 1e-6)
    # IoU反例 IoU_= TN/(TN + FN + FP)
    iou_ = true_neg / (true_neg + false_neg + false_pos + 1e-6)
    # MIoU = (IoU正例p + IoU反例n) / 2 = [ TP/(TP + FP + FN) +TN/(TN + FN + FP) ]/2
    miou = (iou + iou_) / 2

    dice = metric.binary.dc(pred_, gt_)
    # jc = metric.binary.jc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # assd = metric.binary.assd(pred, gt)
    precision = metric.binary.precision(pred_, gt_)
    recall = metric.binary.recall(pred_, gt_)
    sensitivity = metric.binary.sensitivity(pred_, gt_)
    specificity = metric.binary.specificity(pred_, gt_)
    # true_positive_rate = metric.binary.true_positive_rate(pred, gt)
    # true_negative_rate = metric.binary.true_negative_rate(pred, gt)
    # positive_predictive_value = metric.binary.positive_predictive_value(pred, gt)
    # ravd = metric.binary.ravd(pred, gt)

    return dice, precision, recall, sensitivity, specificity, accuracy, f1, iou, miou



def predict(model,test_dataloader,save_predict:bool):

    weights_path = "./save_weights/best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()  # **********

    #plt.ion() #开启动态模式
    with torch.no_grad():
        # 验证指标
        dice_total = 0
        precision_total = 0
        recall_total = 0
        sensitivity_total = 0
        specificity_total = 0
        accuracy_total = 0
        f1_total = 0
        iou_total = 0
        miou_total = 0
        num = len(test_dataloader)  # 验证集图片的总数
        for i, (pic, mask) in enumerate(test_dataloader):
            i += 1
            pic = pic.to(device)
            mask = mask.to(device)
            predict = model(pic)

            array_predict = torch.squeeze(list(predict.values())[0]).cpu().numpy()

            # 使用y和predict来计算指标

            dice, precision, recall, sensitivity, specificity, accuracy, f1, iou, miou = calculate_metric_percase(predict, mask)

            data_root = os.path.join("./", "DRIVE", "test")

            img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]

            roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_test_mask.gif") for i in img_names]
            roi_mask_path = roi_mask[i-1]

            # load roi mask
            roi_img = Image.open(roi_mask_path).convert('L')
            roi_img = np.array(roi_img)

            prediction = predict['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255
            # 将不敢兴趣的区域像素设置成0(黑色)
            prediction[roi_img == 0] = 0
            mask = Image.fromarray(prediction)
            mask.save("test_result{}.tif".format(i))

            dice_total += dice
            precision_total += precision
            recall_total += recall
            sensitivity_total += sensitivity
            specificity_total += specificity
            accuracy_total += accuracy
            f1_total += f1
            iou_total += iou
            miou_total += miou

            print('\n[{}/{}]:  dice={:.4f},  precision={:.4f},  recall={:.4f},sensitivity={:.4f}, specificity={:.4f},'
                  ' accuracy={:.4f},f1={:.4f}, iou={:.4f}, miou={:.4f}'.format(i,num,dice,precision,recall, sensitivity,
                                                                               specificity, accuracy,f1, iou,miou))     # dice和precision.recall
            logging.info('[{}/{}]: dice={:.4f}, precision={:.4f}, recall={:.4f}, sensitivity={:.4f}, specificity={:.4f}, '
                         'accuracy={:.4f},f1={:.4f}, iou={:.4f}, miou={:.4f}'.format(i,num,dice,precision,recall, sensitivity,
                                                                                     specificity, accuracy,f1, iou,miou))

            # # 保存图片
            # if save_predict is True:
            #
            #     array_predict[array_predict >= 0.5] = 1
            #     array_predict[array_predict <= 0.5] = 0
            #
            #     # plt.imsave(predict_plot + '/' + name, array_predict*255, cmap='gray')  # 保存灰度图
            #     plt.imsave('{}.png'.format(i), array_predict*255, cmap='gray')  # 保存图
            #     # plt.show()

        aver_dice = dice_total/num
        aver_precision = precision_total/num
        aver_recall = recall_total/num
        aver_sensitivity = sensitivity_total/num
        aver_specificity = specificity_total/num
        aver_accuracy = accuracy_total/num
        aver_f1 = f1_total/num
        aver_miou = miou_total/num
        print("aver_dice:{:.4f}, aver_precision:{:.4f}, aver_recall:{:.4f}, aver_sensitivity={:.4f}, "
              "aver_specificity={:.4f}, aver_accuracy={:.4f},aver_f1={:.4f},aver_miou={:.4f}".format(aver_dice,aver_precision,aver_recall,
                                                                     aver_sensitivity,aver_specificity,aver_accuracy,aver_f1,aver_miou))

if __name__ == "__main__":


    print('===================>')
    model = UNet(in_channels=3, num_classes=2, base_c=32).to(device)  # 选择使用的模型
    test_dataloader = getDataset()
    # 测试函数
    print("!---------Start predict!---------!")
    predict(model, test_dataloader, save_predict=False)