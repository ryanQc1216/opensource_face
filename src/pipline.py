import cv2, os
import sys
import numpy as np
import pickle
import importlib
from math import floor
from opensource_module.PIPNet.FaceBoxesV2.faceboxes_detector import *
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
from opensource_module.PIPNet.lib.networks import *
from opensource_module.PIPNet.lib.functions import *
from opensource_module.L2CSNet.model import L2CS
from opensource_module.L2CSNet.utils import draw_gaze
from opensource_module.single_image_prediction.ce_clean import resnet34 as AgeNet
import os
from torch.autograd import Variable
import copy


class FacePipeline:
    def __init__(self):
        # fix model init path
        root_PIPNET = './opensource_module/PIPNet'
        root_L2CSNET = './opensource_module/L2CS-Net'

        pipnet_config = os.path.join(root_PIPNET, 'experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py')
        pipnet_weight_path = './opensource_checkpoints/pip_32_16_60_r18_l2_l1_10_1_nb10.pth'
        facebox_weight_path = os.path.join(root_PIPNET, 'FaceBoxesV2/weights/FaceBoxesV2.pth')
        lcsnet_weight_path = './opensource_checkpoints/L2CSNet_gaze360.pkl'
        facenet_weight_path = './opensource_checkpoints/Seesaw_shuffleFaceNet.pt'
        agenet_weight_path = './opensource_checkpoints/afad-ce__seed1.pt'

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        '''
        init face boxes
        '''
        self.face_detector = FaceBoxesDetector('FaceBoxes', facebox_weight_path, self.device != 'cpu', None)
        self.det_thresh = 0.6
        self.det_box_scale = 1.2

        '''
        init pip-net
        '''
        experiment_name = pipnet_config.split('/')[-1][:-3]
        data_name = pipnet_config.split('/')[-2]
        config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
        my_config = importlib.import_module(config_path, package='PIPNet')
        Config = getattr(my_config, 'Config')
        cfg = Config()
        cfg.experiment_name = experiment_name
        cfg.data_name = data_name
        meanface_indices, self.reverse_index1, self.reverse_index2, self.max_len = get_meanface(
            os.path.join(root_PIPNET, 'data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
        assert cfg.backbone == 'resnet18'
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        self.model_pipnet = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size,
                                         net_stride=cfg.net_stride)
        self.input_size_pipnet = cfg.input_size
        self.net_stride_pipnet = cfg.net_stride
        self.num_nb_pipnet = cfg.num_nb
        self.num_lms_pipnet = cfg.num_lms

        self.model_pipnet = self.model_pipnet.to(self.device)
        state_dict = torch.load(pipnet_weight_path, map_location=self.device)
        self.model_pipnet.load_state_dict(state_dict)
        self.model_pipnet.eval()
        normalize_pipnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self.preprocess_pipnet = transforms.Compose(
            [transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize_pipnet])

        '''
        LCS-NET
        '''
        self.preprocess_lcsnet = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.model_lcsnet = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_bins=90)
        state_dictl_lcsnet = torch.load(lcsnet_weight_path)
        self.model_lcsnet.load_state_dict(state_dictl_lcsnet)
        self.model_lcsnet.to(self.device)
        self.model_lcsnet.eval()
        self.softmax_lcsnet = nn.Softmax(dim=1)

        '''
        init face-net
        '''
        self.model_facenet = torch.jit.load(facenet_weight_path)
        self.preprocess_facenet = Compose([transforms.Resize((112, 112)),
                                           ToTensor(), Normalize(mean=[0.5] * 3, std=[0.5] * 3, inplace=True)])
        self.pre_compute_from_config('./images/user_files')

        '''
        init age net
        '''
        self.add_class_agenet = 15
        self.model_agenet = AgeNet(num_classes=26, grayscale=False)
        state_dict_agenet = torch.load(agenet_weight_path, map_location=torch.device('cpu'))
        self.model_agenet.load_state_dict(state_dict_agenet)
        self.model_agenet.to(self.device)
        self.model_agenet.eval()
        self.preprocess_agenet = transforms.Compose([transforms.Resize((128, 128)),
                                                     transforms.CenterCrop((120, 120)),
                                                     transforms.ToTensor()])
        pass

    def bbox_xyxywh(self, detection, image_width, image_height, for_pipnet=False):
        det_xmin = detection[2]
        det_ymin = detection[3]
        det_width = detection[4]
        det_height = detection[5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1
        if for_pipnet:
            det_xmin -= int(det_width * (self.det_box_scale - 1) / 2)
            # remove a part of top area for alignment, see paper for details
            det_ymin += int(det_height * (self.det_box_scale - 1) / 2)
            det_xmax += int(det_width * (self.det_box_scale - 1) / 2)
            det_ymax += int(det_height * (self.det_box_scale - 1) / 2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width - 1)
            det_ymax = min(det_ymax, image_height - 1)
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1

        return det_xmin, det_ymin, det_xmax, det_ymax, det_width, det_height

    def process_pipnet(self, image, detection):
        image_height, image_width, _ = image.shape
        det_xmin, det_ymin, det_xmax, det_ymax, det_width, det_height = \
            self.bbox_xyxywh(detection, image_width, image_height, for_pipnet=True)

        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (self.input_size_pipnet, self.input_size_pipnet))
        inputs = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
        inputs = self.preprocess_pipnet(inputs).unsqueeze(0)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(self.model_pipnet,
                                                                                                     inputs,
                                                                                                     self.preprocess_pipnet,
                                                                                                     self.input_size_pipnet,
                                                                                                     self.net_stride_pipnet,
                                                                                                     self.num_nb_pipnet)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.num_lms_pipnet, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.num_lms_pipnet, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        return lms_pred_merge, [det_xmin, det_ymin, det_width, det_height]

    def process_lcsnet(self, image, detection):
        image_height, image_width, _ = image.shape
        det_xmin, det_ymin, det_xmax, det_ymax, det_width, det_height = \
            self.bbox_xyxywh(detection, image_width, image_height, for_pipnet=False)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]

        img_lscnet = cv2.cvtColor(det_crop, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_lscnet)
        img_lscnet = self.preprocess_lcsnet(im_pil)
        img_lscnet = Variable(img_lscnet).to(self.device)
        img_lscnet = img_lscnet.unsqueeze(0)

        # gaze prediction
        with torch.no_grad():
            gaze_pitch, gaze_yaw = self.model_lcsnet(img_lscnet)

        idx_tensor = [idx for idx in range(90)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        pitch_predicted = self.softmax_lcsnet(gaze_pitch)
        yaw_predicted = self.softmax_lcsnet(gaze_yaw)
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

        pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi / 180.0
        yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi / 180.

        return pitch_predicted, yaw_predicted, [det_xmin, det_ymin, det_width, det_height]

    def process_facenet(self, image, detection=None):
        if detection is None:
            detections, _ = self.face_detector.detect(image, self.det_thresh, 1)
            if len(detections) == 0:
                raise "Not found face"
            detection = detections[0]
        image_height, image_width, _ = image.shape
        det_xmin, det_ymin, det_xmax, det_ymax, det_width, det_height = \
            self.bbox_xyxywh(detection, image_width, image_height, for_pipnet=False)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]

        image_mat = cv2.cvtColor(det_crop, cv2.COLOR_RGB2BGR)
        inputs = Image.fromarray(image_mat[:, :, ::-1].astype('uint8'), 'RGB')
        image_tesor = self.preprocess_facenet(inputs).unsqueeze(0)
        face_embedding = self.model_facenet(image_tesor)
        return face_embedding, [det_xmin, det_ymin, det_width, det_height]

    def select_person(self, face_embedding):
        max_prob = -1
        person_name = None
        for person in self.face_embedding_dict:
            prob = self.facenet_calc_similarity(self.face_embedding_dict[person], face_embedding)
            if prob > max_prob:
                person_name = person
                max_prob = prob
        return person_name, max_prob

    def facenet_calc_similarity(self, feature1, feature2):
        dot = np.sum(np.multiply(feature1, feature2), axis=1)
        norm = np.linalg.norm(feature1, axis=1) * np.linalg.norm(feature2, axis=1)
        dist = dot / norm
        return dist

    def pre_compute_from_config(self, user_files):
        filelist = os.listdir(user_files)
        self.face_embedding_dict = dict()
        for filename in filelist:
            user_name = filename.split('.')[-2]
            image = cv2.imread(os.path.join(user_files, filename))

            face_embedding, _ = self.process_facenet(image, detection=None)
            self.face_embedding_dict[user_name] = face_embedding.detach().cpu().numpy()
        '''
        only for test
        '''
        print(self.facenet_calc_similarity(self.face_embedding_dict['Person_1'], self.face_embedding_dict['Person_2']))
        # print(self.facenet_calc_similarity(self.face_embedding_dict['Test_1'], self.face_embedding_dict['Person_2']))
        # print(self.facenet_calc_similarity(self.face_embedding_dict['Person_1'], self.face_embedding_dict['Test_1']))

    def process_agenet(self, image, detection):

        image_height, image_width, _ = image.shape
        det_xmin, det_ymin, det_xmax, det_ymax, det_width, det_height = \
            self.bbox_xyxywh(detection, image_width, image_height, for_pipnet=False)
        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]

        image_mat = cv2.cvtColor(det_crop, cv2.COLOR_RGB2BGR)
        inputs = Image.fromarray(image_mat[:, :, ::-1].astype('uint8'), 'RGB')
        image_tesor = self.preprocess_agenet(inputs).unsqueeze(0).to(self.device)
        logits, age_probs = self.model_agenet(image_tesor)
        age = torch.argmax(age_probs, 1).item() + self.add_class_agenet
        age_prob = torch.max(age_probs).item()
        return age, age_prob, [det_xmin, det_ymin, det_width, det_height]

    def draw_information(self,
                         image, lms_pred_merge, pitch_predicted, yaw_predicted, person_name, person_prob, age, age_prob,
                         xywh_pipnet, xywh_lcsnet, xywh_facenet,
                         ):
        det_xmin_pipnet, det_ymin_pipnet, det_width_pipnet, det_height_pipnet = xywh_pipnet
        det_xmin_lcsnet, det_ymin_lcsnet, det_width_lcsnet, det_height_lcsnet = xywh_lcsnet

        (h, w) = image.shape[:2]
        length = w / 2

        for i in range(self.num_lms_pipnet):
            x_pred = lms_pred_merge[i * 2] * det_width_pipnet
            y_pred = lms_pred_merge[i * 2 + 1] * det_height_pipnet
            coord = (int(x_pred) + det_xmin_pipnet, int(y_pred) + det_ymin_pipnet)
            cv2.circle(image, coord, 1, (0, 255, 0), 1)
            # cv2.putText(image, '%d' % (i),
            #             coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if i == 96 or i == 97:
                # draw gaze
                pos = coord
                dx = -length * np.sin(pitch_predicted) * np.cos(yaw_predicted)
                dy = -length * np.sin(yaw_predicted)
                cv2.arrowedLine(image, tuple(np.round(pos).astype(np.int32)),
                                tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), (255, 255, 0),
                                2, cv2.LINE_AA, tipLength=0.18)

        cv2.putText(image, 'faceID: %s, %.2f' % (person_name, person_prob),
                    (det_xmin_lcsnet, det_ymin_lcsnet), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(image, 'Age: %d, %.2f' % (age, age_prob),
                    (det_xmin_lcsnet, det_ymin_lcsnet + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return image

    def pipeline(self, image, draw_immediate=False):
        detections, _ = self.face_detector.detect(image, self.det_thresh, 1)

        results = []
        for i in range(len(detections)):
            # pipnet
            lms_pred_merge, xywh_pipnet = self.process_pipnet(image, detections[i])
            # lcsnet
            pitch_predicted, yaw_predicted, xywh_lcsnet = self.process_lcsnet(image, detections[i])
            # facenet
            face_embedding, xywh_facenet = self.process_facenet(image, detections[i])
            # agenet
            age, age_prob, xywh_agenet = self.process_agenet(image, detections[i])

            person_name, person_prob = self.select_person(face_embedding.detach().cpu().numpy())
            if draw_immediate:
                self.draw_information(image,
                                      lms_pred_merge,
                                      pitch_predicted, yaw_predicted,
                                      person_name, person_prob,
                                      age, age_prob,
                                      xywh_pipnet, xywh_lcsnet, xywh_facenet,
                                      )

            pipnet_res = dict(lms_pred=lms_pred_merge,
                              xywh=xywh_pipnet)
            lcsnet_res = dict(pitch_pred=lms_pred_merge,
                              yaw_pred=yaw_predicted,
                              xywh=xywh_lcsnet)
            facenet_res = dict(face_embedding=face_embedding,
                               person_name=person_name,
                               person_prob=person_prob,
                               xywh=xywh_facenet)
            agenet_res = dict(age=lms_pred_merge,
                              prob=age_prob,
                              xywh=xywh_agenet)

            results.append(
                dict(
                    pipnet=pipnet_res,
                    lcsnet=lcsnet_res,
                    facenet=facenet_res,
                    agenet=agenet_res,
                )
            )
        return results
