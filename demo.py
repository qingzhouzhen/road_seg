from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from torchvision import transforms
from FCN import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss = 0
    fcn_model = torch.load("/data/project/road_seg/checkpoints/fcn_model_995.pt")
    fcn_model.eval()
    ori_path = "/data/data/road/"
    mask_path = "/data/data/road_mask/"
    origin_img_list = os.listdir(ori_path)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    with torch.no_grad():

        for name in origin_img_list:
            abs_img_path = os.path.join(ori_path, name)
            imgA = cv2.imread(abs_img_path)
            imgA = cv2.resize(imgA, (640, 640))
            road = transform(imgA)
            road = road.unsqueeze(0)
            abs_mask_path = os.path.join(mask_path, name)
            imgB = cv2.imread(abs_mask_path, 0)
            imgB = cv2.resize(imgB, (640, 640))
            imgB = imgB / 255
            imgB = imgB.astype('uint8')
            imgB = onehot(imgB, 2)
            imgB = imgB.transpose(2, 0, 1)
            road_msk = torch.FloatTensor(imgB)
            road_msk = road_msk.unsqueeze(0)
            road = road.to(device)
            road_msk = road_msk.to(device)

            # optimizer.zero_grad()
            output = fcn_model(road)

            print(output.shape)

            output = torch.sigmoid(output)

            output_np = output.cpu().detach().numpy().copy()
            output_np = np.argmin(output_np, axis=1)
            road_msk_np = road_msk.cpu().detach().numpy().copy()
            road_msk_np = np.argmin(road_msk_np, axis=1)

            img_path = os.path.join(abs_img_path)
            img = cv2.imread(img_path)
            img_shape = img.shape
            output_shape = output_np.shape
            width_ratio = img_shape[1] / output_shape[2]
            height_ratio = img_shape[0] / output_shape[1]

            # plt.subplot(1, 2, 1)
            # plt.imshow(np.squeeze(road_msk_np[0, ...]), 'gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
            # plt.pause(5)

            output_shape = output_np.shape
            for i in range(output_shape[1]):
                for j in range(output_shape[2]):
                    if output_np[0, i, j] == 0:
                        # print("draw circle: ", i, j)
                        x = int(j * width_ratio)
                        y = int(i * height_ratio)
                        img[y, x, 0] = 0
                        img[y, x, 1] = 0
                        img[y, x, 2] = 255
                        cv2.circle(img, (x,y), 1, (0, 0, 255), 0)
            cv2.imshow("test", img)
            cv2.waitKey(0)