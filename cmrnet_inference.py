import open3d as o3
import numpy as np
import pykitti
from tqdm import tqdm
import torch
import torch.nn.functional as F
from CMRNet.utils import (to_rotation_matrix, mat2xyzrpy, merge_inputs, overlay_imgs, quat2mat,
                   quaternion_from_matrix, rotate_back, rotate_forward,
                   tvector2mat)
from CMRNet.models.CMRNet.CMRNet import CMRNet
from CMRNet.camera_model import CameraModel
from torchvision import transforms
import visibility
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R_scipy

class RefineEstimate(object):
    """
    Functionality to refine an estimate of a pose
    using CMR Net
    """
    def __init__(self, config):
        path_to_dataset = config["path_to_dataset"] 
        path_to_map = config["path_to_map"]
        weight_paths = config["weight_paths"]
        sequence = config["sequence"]
        # setup kitti manager
        self.kitti = pykitti.odometry(path_to_dataset, sequence)
        # load the downsampled map
        full_map = o3.read_point_cloud(path_to_map)
        # convert map into torch tensor. this is (3,no of points) coordinates
        voxelized = torch.tensor(full_map.points, dtype=torch.float)
        # added a extra homogeneous cordinate
        voxelized = torch.cat((voxelized, torch.ones([voxelized.shape[0], 1], dtype=torch.float)), 1)
        voxelized = voxelized.t()
        voxelized = voxelized.to("cuda")
        # this containsthe intensities of each of this point
        vox_intensity = torch.tensor(full_map.colors, dtype=torch.float)[:, 0:1].t()
        velo2cam2 = torch.from_numpy(self.kitti.calib.T_cam2_velo).float().to("cuda")
        self.map = voxelized
        self.velo2cam = velo2cam2
        self.cam2velo = self.velo2cam.inverse()
        self.map_intensity = vox_intensity
        # load the models into a list
        self.image_shape = (384,1280)
        self.models = self.load_models(weight_paths)
        # initialize a camera model used to project lidar point cloud
        self.cam_params = self.get_calib_kitti(sequence).cuda()
        self.camera_model = CameraModel(focal_length=self.cam_params[:2], 
                                        principal_point=self.cam_params[2:])
    
    def get_calib_kitti(self, sequence):
        if sequence == '00':
            return torch.tensor([718.856, 718.856, 607.1928, 185.2157])
        elif sequence == '03':
            return torch.tensor([721.5377, 721.5377, 609.5593, 172.854])
        elif sequence in ['05', '06', '07', '08', '09']:
            return torch.tensor([707.0912, 707.0912, 601.8873, 183.1104])
        else:
            raise TypeError("Sequence Not Available")
            
    def load_models(self,weight_paths):
        """
        Loads the models stored in the paths sent.
        Args:
        weight_paths: A list of paths for the models
        Returns:
        A list of models
        """
        models = []
        for i, path in enumerate(weight_paths):
            model = CMRNet(self.image_shape, use_feat_from=1, md=4, use_reflectance=False)
            checkpoint = torch.load(path, map_location='cpu')
            saved_state_dict = checkpoint['state_dict']
            model.load_state_dict(saved_state_dict)
            model = model.to("cuda")
            model.eval()
            models.append(model)
            if i == 0:
                self.occlusion_th = checkpoint['config']['occlusion_threshold']
                self.occlusion_kernel = checkpoint['config']['occlusion_kernel']
        return models
    
    def project_and_crop_pc_into_cam_frame(self, camera_pose):
        """
        Projects the pointcloud into camera frame. 
        Selects points within the view of the camera and returns them.
        Args:
        camera_pose: A 4 X 4 transformation matrix SE3 that takes points in the camera coordinate frame to world frame
        Returns:
        projected_pc: a dict with points projected in camera frame and the relevant intensities
        """
        # compute the velodyne pose as a transformation matrix from this
        rot_mat_cam = torch.from_numpy(camera_pose).float()
        rot_mat_cam = rot_mat_cam.to("cuda")
        rot_mat_velo = torch.mm(self.cam2velo, rot_mat_cam.inverse())
        # transform the velodyne point cloud positions into the current velo dyne pose
        local_map = self.map.clone()
#         local_intensity = self.map_intensity.clone()
        local_map = torch.mm(rot_mat_velo, local_map).t()
        # select indices within viewing range
        indexes = local_map[:, 1] > -25.
        indexes = indexes & (local_map[:, 1] < 25.)
        indexes = indexes & (local_map[:, 0] > -10.)
        indexes = indexes & (local_map[:, 0] < 100.)
        local_map = local_map[indexes]
#         local_intensity = local_intensity[:, indexes]
        # convert the positions into camera frame
        local_map = torch.mm(self.velo2cam, local_map.t())
        local_map = local_map[[2, 0, 1, 3], :]
#         projected_pc = {
#             "positions": local_map,
#             "intensities": local_intensity
#         }
        return local_map
    
    def preprocess_image(self, image):
        """
        Preprocess image to feed into network
        """
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        
        image = to_tensor(image)
        image = normalization(image)
        return image
        
    def project_lidar(self, pc, image_shape, shape_pad):
        """
        Project the lidar point cloud into an 2D array
        """
        uv, depth, points, refl = self.camera_model.project_pytorch(pc, image_shape, None)
        uv = uv.t().int()
        depth_img = torch.zeros(image_shape[:2], device='cuda', dtype=torch.float)
        depth_img += 1000.
        depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], image_shape[1], image_shape[0])
        depth_img[depth_img == 1000.] = 0.
        projected_points = torch.zeros_like(depth_img, device='cuda')
        projected_points = visibility.visibility2(depth_img, self.cam_params, projected_points, depth_img.shape[1],
                                                      depth_img.shape[0], self.occlusion_th,
                                                      self.occlusion_kernel)
        projected_points /= 100.
        projected_points = projected_points.unsqueeze(0)
        projected_points = F.pad(projected_points, shape_pad)
        return projected_points
        
    def refine_pose_estimate(self, predicted_pose, image):
        """
        Get a refined estimate using CMRNet
        """
        processed_image = self.preprocess_image(image)
        image_shape = (processed_image.shape[1], processed_image.shape[2], processed_image.shape[0])
        processed_image = processed_image.cuda()
        shape_pad = [0, 0, 0, 0]
        shape_pad[3] = (self.image_shape[0] - processed_image.shape[1])
        shape_pad[1] = (self.image_shape[1] - processed_image.shape[2])
        processed_image = F.pad(processed_image, shape_pad)
        # project point cloud into 2D array
        local_pc_in_cam_frame = self.project_and_crop_pc_into_cam_frame(predicted_pose)
        pc_image = self.project_lidar(local_pc_in_cam_frame, image_shape, shape_pad)
        # iteratively refine
        RT_cumulative = torch.eye(4).cuda()
        images = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                T_predicted, R_predicted = model(processed_image.unsqueeze(0), pc_image.unsqueeze(0))
                R_predicted = quat2mat(R_predicted[0])
                T_predicted = tvector2mat(T_predicted[0])
                RT_predicted = torch.mm(T_predicted, R_predicted)
                RT_cumulative = torch.mm(RT_cumulative, RT_predicted)
                out3 = overlay_imgs(processed_image, pc_image.unsqueeze(0))
                local_pc_in_cam_frame = rotate_forward(local_pc_in_cam_frame, RT_predicted)
                pc_image = self.project_lidar(local_pc_in_cam_frame, image_shape, shape_pad)
                images.append(out3)
        # neural net predicted relative pose for refining velo coordinates. need to get into cam frame
        RT_cumulative = RT_cumulative.cpu().numpy()
        old_axes=R_scipy.from_matrix(RT_cumulative[:3,:3]).as_euler(seq='xyz')
        order=[1,2,0]
        new_axes=old_axes[order]
        RT_cumulative[:3,:3] = R_scipy.from_euler('xyz',new_axes).as_matrix()
        RT_cumulative[:3,-1] = RT_cumulative[order,-1]
        return RT_cumulative, images

    def update(self, current_transform, img_rgb):
        refinement, _ = self.refine_pose_estimate(
            current_transform, img_rgb)
        # perform refinement
        cmr_global_transform_estimate = current_transform @ refinement
        gps_pos = cmr_global_transform_estimate[:3, -1]
        return gps_pos


# example to use
if __name__ == '__main__':
    # create the config dictionary
    config = {
        "weight_paths": ['./CMRNet/checkpoints/iter1.tar','./CMRNet/checkpoints/iter2.tar','./CMRNet/checkpoints/iter3.tar'],
        "path_to_map": "./map-00_0.1_0-300.pcd",
        "path_to_dataset": "./CMRNet/KITTI_ODOMETRY",
        "sequence": "00"
    }
    # create inference class
    cmr_manager = RefineEstimate(config)
    # generate a perturbed pose and feed it to the net
    kitti = pykitti.odometry("/home/arcot/Projects/SLAM_Project/dataset", "00")
    pose1 = copy.copy(kitti.poses[100])
    cam0velo = kitti.calib.T_cam0_velo
    cam2velo = kitti.calib.T_cam2_velo
    cam0_to_cam2 = np.linalg.inv(cam2velo) @ cam0velo
    # print(pose1)
    pose1 =  cam0_to_cam2 @ pose1
    # print(pose1)
    # print(cam0_to_cam2)
    R = R_scipy.from_euler('xyz', [15,0,0], degrees = True).as_matrix()
    T = np.eye(4)
    T[:3,:3] = R
    T[0,-1] = 1
    pose1 = pose1 @ T
    # pose1[0,-1] += 2
    image = kitti.get_cam2(100)
    pose, images = cmr_manager.refine_pose_estimate(pose1, image)
    # print(pose1)
    print(pose)
    print(np.linalg.inv(T))
    for i in range(len(images)):
        plt.figure(figsize=(20,10))
        plt.imshow(images[i])
    plt.show()
