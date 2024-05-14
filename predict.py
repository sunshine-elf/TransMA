import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from medpy import metric
from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def calculate_metric_percase(pred, gt):

    pred[pred > 0.5] = 1
    pred[pred < 0.5] = 0
    # gt[gt > 0.5] = 1
    # gt[gt < 0.5] = 0
    pred_ = torch.squeeze(pred).cpu().numpy()  # (512,512)  .cpu()
    gt_ = torch.squeeze(gt).cpu().numpy()  # (512,512)
    # acc = metrics.ac
    dice = metric.binary.dc(pred_, gt_)
    # from sklearn import metrics jc = metric.binary.jc(pred, gt)
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

    return dice, precision, recall, sensitivity, specificity

def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    img_path = "./DRIVE/test/images/10_test.tif"
    roi_mask_path = "./DRIVE/test/mask/10_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
