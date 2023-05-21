import torch
import numpy as np
import cv2
from src.modeling._mano import MANO, Mesh
from PIL import Image
from torchvision import transforms

class HandTopDownRecognition:
    
    def __init__(self, device = "cuda",checkpoint = 'weights/model.bin'):
        self.mano = MANO().to(device)
        self.mano.layer = self.mano.layer.cuda()
        self.mesh_sampler = Mesh()
        self.prepare_augmentation()
        self.create_model(checkpoint, device)
    
    def prepare_augmentation(self):
        self.transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

        self.transform_visualize = transforms.Compose([           
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor()])
    
    def create_model(self, checkpoint, device):
        self._model = torch.load(checkpoint)
        setattr(self._model.trans_encoder[-1].config,'output_attentions', True)
        setattr(self._model.trans_encoder[-1].config,'output_hidden_states', True)
        self._model.trans_encoder[-1].bert.encoder.output_attentions = True
        self._model.trans_encoder[-1].bert.encoder.output_hidden_states =  True
        for iter_layer in range(4):
            self._model.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
        for inter_block in range(3):
            setattr(self._model.trans_encoder[-1].config,'device', device)
        self._model.to(device)
        
    def run_inference(self, img):
        self._model.eval()
        self.mano.eval()
        with torch.no_grad():
            img_tensor = self.transform(img)
            img_visual = self.transform_visualize(img)
            batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
            batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
            pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = self._model(batch_imgs, self.mano, self.mesh_sampler) 
        return pred_3d_joints, pred_camera
    
    def orthographic_projection(self,X, camera):
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        shape = X_trans.shape
        X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
        return X_2d
    
    def get_keypoints(self, img):
        points_3d, camera = self.run_inference(img)
        points_2d = self.orthographic_projection(points_3d.contiguous(), camera.contiguous())
        return points_2d