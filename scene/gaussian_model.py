#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d, build_scaling_rotation_4d
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import sh_channels_4d

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L.transpose(1, 2) @ L
            symm = strip_symmetric(actual_covariance)
            return symm
        
        def build_covariance_from_scaling_rotation_4d(scaling, scaling_modifier, rotation_l, rotation_r, dt=0.0):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_11 = actual_covariance[:,:3,:3]
            cov_12 = actual_covariance[:,0:3,3:4]
            cov_t = actual_covariance[:,3:4,3:4]
            current_covariance = cov_11 - cov_12 @ cov_12.transpose(1, 2) / cov_t
            symm = strip_symmetric(current_covariance)
            if dt.shape[1] > 1:
                mean_offset = (cov_12.squeeze(-1) / cov_t.squeeze(-1))[:, None, :] * dt[..., None]
                mean_offset = mean_offset[..., None]  # [num_pts, num_time, 3, 1]
            else:
                mean_offset = cov_12.squeeze(-1) / cov_t.squeeze(-1) * dt
            return symm, mean_offset.squeeze(-1)
        
        def build_covariance_t(scaling, scaling_modifier, rotation_l, rotation_r):
            L = build_scaling_rotation_4d(scaling_modifier * scaling, rotation_l, rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            cov_t = actual_covariance[:,3:4,3:4]
            return cov_t.squeeze(-1)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        if not self.rot_4d:
            self.covariance_activation = build_covariance_from_scaling_rotation
        else:
            self.covariance_activation = build_covariance_from_scaling_rotation_4d
            self.cov_t_activation = build_covariance_t

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, force_sh_3d: bool = False, sh_degree_t : int = 0, current_timestamp : float = 0.0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.force_sh_3d = force_sh_3d
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d or self.force_sh_3d:
            assert self.gaussian_dim == 4
        self.env_map = torch.empty(0)
        
        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.current_timestamp = current_timestamp
        self.setup_functions()
        self.opt_states = {}

    def capture(self):
        if self.gaussian_dim == 3:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        elif self.gaussian_dim == 4:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.t_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self._t,
                self._scaling_t,
                self._rotation_r,
                self.rot_4d,
                self.env_map,
                self.active_sh_degree_t
            )
    
    def restore(self, model_args, training_args):
        if self.gaussian_dim == 3:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif self.gaussian_dim == 4:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            t_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._t,
            self._scaling_t,
            self._rotation_r,
            self.rot_4d,
            self.env_map,
            self.active_sh_degree_t) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.t_gradient_accum = t_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    def clone_by_mask(self, mask : torch.Tensor, gaussians : "GaussianModel", opt, new_gaussians : "GaussianModel"):
       #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        # new_gaussian.restore([self.active_sh_degree, self._xyz[mask], self._features_dc[mask], self._features_rest[mask], self._scaling[mask], self._rotation[mask],
        #                     self._opacity[mask], self.max_radii2D[mask], self.xyz_gradient_accum[mask], self.t_gradient_accum[mask], self.denom[mask],
        #                     self.spatial_lr_scale[mask], self._t[mask], self._scaling_t[mask], self._rotation_r[mask], self.rot_4d, self.env_map[mask], self.active_sh_degree_t], None)
        #new_gaussian.active_sh_degree = self.active_sh_degree
        if new_gaussians is None:
            self._xyz = gaussians._xyz[mask].cpu()
            self._features_dc = gaussians._features_dc[mask].cpu()
            self._features_rest = gaussians._features_rest[mask].cpu()
            self._scaling = gaussians._scaling[mask].cpu()
            self._rotation = gaussians._rotation[mask].cpu()
            self._opacity = gaussians._opacity[mask].cpu()
            self.max_radii2D = gaussians.max_radii2D[mask].cpu()
            self.xyz_gradient_accum = gaussians.xyz_gradient_accum[mask].cpu()
            self.t_gradient_accum = gaussians.t_gradient_accum[mask].cpu()
            self.denom = gaussians.denom[mask].cpu()
            #self.spatial_lr_scale = gaussians.spatial_lr_scale[mask].cpu()
            self._t = gaussians._t[mask].cpu()
            self._scaling_t = gaussians._scaling_t[mask].cpu()
            self._rotation_r = gaussians._rotation_r[mask].cpu()

            self.rot_4d = gaussians.rot_4d
            # if gaussians.env_map is not None:
            #     self.env_map = gaussians.env_map[mask].cpu()
            #new_gaussian.active_sh_degree_t = self.active_sh_degree_t
            #new_gaussian.percent_dense = self.percent_dense
            #new_gaussian.optimizer = self.optimizer
            #new_gaussian.max_sh_degree = self.max_sh_degree
            self.gaussian_dim = gaussians.gaussian_dim
            self.time_duration = gaussians.time_duration
            self.force_sh_3d = gaussians.force_sh_3d
            self.max_sh_degree_t = gaussians.max_sh_degree_t
            #self.training_setup(opt)
            #new_gaussian.setup_functions()
            self.opt_states.clear()
            for group in gaussians.optimizer.param_groups:
                optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
                #attr = self.get_param_group_corresponding_attr(group["name"])
                if optimizer_state is not None and len(optimizer_state) != 0:
                    state_content = {}
                    state_content["step"] = optimizer_state["step"].cpu()
                    state_content["exp_avg"] = optimizer_state["exp_avg"][mask].cpu()
                    state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].cpu()
                    self.opt_states[group["name"]] = state_content
        else:

            new_gaussians._xyz = torch.cat([new_gaussians._xyz, gaussians._xyz[mask].cpu()])
            new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, gaussians._features_dc[mask].cpu()])
            new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, gaussians._features_rest[mask].cpu()])
            new_gaussians._scaling = torch.cat([new_gaussians._scaling, gaussians._scaling[mask].cpu()])
            new_gaussians._rotation = torch.cat([new_gaussians._rotation, gaussians._rotation[mask].cpu()])
            new_gaussians._opacity = torch.cat([new_gaussians._opacity, gaussians._opacity[mask].cpu()])
            new_gaussians.max_radii2D = torch.cat([new_gaussians.max_radii2D, gaussians.max_radii2D[mask].cpu()])
            new_gaussians.xyz_gradient_accum = torch.cat([new_gaussians.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].cpu()])
            new_gaussians.t_gradient_accum = torch.cat([new_gaussians.t_gradient_accum, gaussians.t_gradient_accum[mask].cpu()])
            new_gaussians.denom = torch.cat([new_gaussians.denom, gaussians.denom[mask].cpu()])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).cpu()
            new_gaussians._t = torch.cat([new_gaussians._t, gaussians._t[mask].cpu()])
            new_gaussians._scaling_t = torch.cat([new_gaussians._scaling_t, gaussians._scaling_t[mask].cpu()])
            new_gaussians._rotation_r = torch.cat([new_gaussians._rotation_r, gaussians._rotation_r[mask].cpu()])

            # new_gaussians._xyz = gaussians._xyz[mask].cpu()
            # new_gaussians._features_dc = gaussians._features_dc[mask].cpu()
            # new_gaussians._features_rest = gaussians._features_rest[mask].cpu()
            # new_gaussians._scaling = gaussians._scaling[mask].cpu()
            # new_gaussians._rotation = gaussians._rotation[mask].cpu()
            # new_gaussians._opacity = gaussians._opacity[mask].cpu()
            # new_gaussians.max_radii2D = gaussians.max_radii2D[mask].cpu()
            # new_gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[mask].cpu()
            # new_gaussians.t_gradient_accum = gaussians.t_gradient_accum[mask].cpu()
            # new_gaussians.denom = gaussians.denom[mask].cpu()
            # #self.spatial_lr_scale = gaussians.spatial_lr_scale[mask].cpu()
            # new_gaussians._t = gaussians._t[mask].cpu()
            # new_gaussians._scaling_t = gaussians._scaling_t[mask].cpu()
            # new_gaussians._rotation_r = gaussians._rotation_r[mask].cpu()

            # new_gaussians.rot_4d = gaussians.rot_4d
            # if gaussians.env_map is not None:
            #     self.env_map = gaussians.env_map[mask].cpu()
            #new_gaussian.active_sh_degree_t = self.active_sh_degree_t
            #new_gaussian.percent_dense = self.percent_dense
            #new_gaussian.optimizer = self.optimizer
            #new_gaussian.max_sh_degree = self.max_sh_degree
            # new_gaussians.gaussian_dim = gaussians.gaussian_dim
            # new_gaussians.time_duration = gaussians.time_duration
            # new_gaussians.force_sh_3d = gaussians.force_sh_3d
            # new_gaussians.max_sh_degree_t = gaussians.max_sh_degree_t
            # for group in gaussians.optimizer.param_groups:
            #     optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            #     #attr = self.get_param_group_corresponding_attr(group["name"])
            #     if optimizer_state is not None and len(optimizer_state) != 0:
            #         state_content = {}
            #         state_content["step"] = optimizer_state["step"].cpu()
            #         state_content["exp_avg"] = optimizer_state["exp_avg"][mask].cpu()
            #         state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].cpu()
            #         new_gaussians.opt_states[group["name"]] = state_content

    def clone_from_cpu(self, gaussians_segments):
        xyz_list = []
        for segment in gaussians_segments:
            xyz_list.append(segment._xyz.cuda())
        #del self.optimizer.state[self._xyz]
        self._xyz = nn.Parameter(torch.cat(xyz_list))

        features_dc_list = []
        for segment in gaussians_segments:
            features_dc_list.append(segment._features_dc.cuda())
        self._features_dc = nn.Parameter(torch.cat(features_dc_list))

        features_rest_list = []
        for segment in gaussians_segments:
            features_rest_list.append(segment._features_rest.cuda())
        self._features_rest = nn.Parameter(torch.cat(features_rest_list))

        scaling_list = []
        for segment in gaussians_segments:
            scaling_list.append(segment._scaling.cuda())
        self._scaling = nn.Parameter(torch.cat(scaling_list))

        rotation_list = []
        for segment in gaussians_segments:
            rotation_list.append(segment._rotation.cuda())
        self._rotation = nn.Parameter(torch.cat(rotation_list))

        opacity_list = []
        for segment in gaussians_segments:
            opacity_list.append(segment._opacity.cuda())
        self._opacity = nn.Parameter(torch.cat(opacity_list))

        t_list = []
        for segment in gaussians_segments:
            t_list.append(segment._t.cuda())
        self._t = nn.Parameter(torch.cat(t_list))

        scaling_t_list = []
        for segment in gaussians_segments:
            scaling_t_list.append(segment._scaling_t.cuda())
        self._scaling_t = nn.Parameter(torch.cat(scaling_t_list))

        rotation_r_list = []
        for segment in gaussians_segments:
            rotation_r_list.append(segment._rotation_r.cuda())
        self._rotation_r = nn.Parameter(torch.cat(rotation_r_list))

        max_radii2D_list = []
        for segment in gaussians_segments:
            max_radii2D_list.append(segment.max_radii2D.cuda())
        self.max_radii2D = torch.cat(max_radii2D_list)

        xyz_gradient_accum_list = []
        for segment in gaussians_segments:
            xyz_gradient_accum_list.append(segment.xyz_gradient_accum.cuda())
        self.xyz_gradient_accum = torch.cat(xyz_gradient_accum_list)

        t_gradient_accum_list = []
        for segment in gaussians_segments:
            t_gradient_accum_list.append(segment.t_gradient_accum.cuda())
        self.t_gradient_accum = torch.cat(t_gradient_accum_list)

        denom_list = []
        for segment in gaussians_segments:
            denom_list.append(segment.denom.cuda())
        self.denom = torch.cat(denom_list)

        self.rot_4d = gaussians_segments[0].rot_4d
        self.gaussian_dim = gaussians_segments[0].gaussian_dim
        self.time_duration = gaussians_segments[0].time_duration
        self.force_sh_3d = gaussians_segments[0].force_sh_3d
        self.max_sh_degree_t = gaussians_segments[0].max_sh_degree_t
    
    def clone_to(self, gaussians : "GaussianModel"):
        #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        # new_gaussian.restore([self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
        #                     self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.t_gradient_accum, self.denom,
        #                     self.spatial_lr_scale, self._t, self._scaling_t, self._rotation_r, self.rot_4d, self.env_map, self.active_sh_degree_t], None)
        #gaussians.active_sh_degree = self.active_sh_degree
        state_dict = {}
        if gaussians._xyz in gaussians.optimizer.state:
            state_dict["xyz"] = gaussians.optimizer.state[gaussians._xyz]
            del gaussians.optimizer.state[gaussians._xyz]
        gaussians._xyz = nn.Parameter(self._xyz.cuda())

        if gaussians._features_dc in gaussians.optimizer.state:
            state_dict["f_dc"] = gaussians.optimizer.state[gaussians._features_dc]
            del gaussians.optimizer.state[gaussians._features_dc]
        gaussians._features_dc = nn.Parameter(self._features_dc.cuda())

        if gaussians._features_rest in gaussians.optimizer.state:
            state_dict["f_rest"] = gaussians.optimizer.state[gaussians._features_rest]
            del gaussians.optimizer.state[gaussians._features_rest]
        gaussians._features_rest = nn.Parameter(self._features_rest.cuda())

        if gaussians._scaling in gaussians.optimizer.state:
            state_dict["scaling"] = gaussians.optimizer.state[gaussians._scaling]
            del gaussians.optimizer.state[gaussians._scaling]
        gaussians._scaling = nn.Parameter(self._scaling.cuda())

        if gaussians._rotation in gaussians.optimizer.state:
            state_dict["rotation"] = gaussians.optimizer.state[gaussians._rotation]
            del gaussians.optimizer.state[gaussians._rotation]
        gaussians._rotation = nn.Parameter(self._rotation.cuda())

        if gaussians._opacity in gaussians.optimizer.state:
            state_dict["opacity"] = gaussians.optimizer.state[gaussians._opacity]
            del gaussians.optimizer.state[gaussians._opacity]
        gaussians._opacity = nn.Parameter(self._opacity.cuda())

        gaussians.max_radii2D = self.max_radii2D.cuda()

        gaussians.xyz_gradient_accum = self.xyz_gradient_accum.cuda()
        gaussians.t_gradient_accum = self.t_gradient_accum.cuda()
        gaussians.denom = self.denom.cuda()
        #gaussians.spatial_lr_scale = self.spatial_lr_scale.cuda()
        if gaussians._t in gaussians.optimizer.state:
            state_dict["t"] = gaussians.optimizer.state[gaussians._t]
            del gaussians.optimizer.state[gaussians._t]
        gaussians._t = nn.Parameter(self._t.cuda())

        if gaussians._scaling_t in gaussians.optimizer.state:
            state_dict["scaling_t"] = gaussians.optimizer.state[gaussians._scaling_t]
            del gaussians.optimizer.state[gaussians._scaling_t]
        gaussians._scaling_t = nn.Parameter(self._scaling_t.cuda())

        if gaussians._rotation_r in gaussians.optimizer.state:
            state_dict["rotation_r"] = gaussians.optimizer.state[gaussians._rotation_r]
            del gaussians.optimizer.state[gaussians._rotation_r]
        gaussians._rotation_r = nn.Parameter(self._rotation_r.cuda())

        gaussians.rot_4d = self.rot_4d
        # if self.env_map is not None:
        #     gaussians.env_map = self.env_map.cuda()
        #gaussians.active_sh_degree_t = self.active_sh_degree_t
        #gaussians.percent_dense = self.percent_dense
        #new_gaussian.optimizer = self.optimizer
        #gaussians.max_sh_degree = self.max_sh_degree
        gaussians.gaussian_dim = self.gaussian_dim
        gaussians.time_duration = self.time_duration
        gaussians.force_sh_3d = self.force_sh_3d
        gaussians.max_sh_degree_t = self.max_sh_degree_t
        return state_dict
    
    def get_state_dict(self):
        #new_gaussian = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        # new_gaussian.restore([self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation,
        #                     self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.t_gradient_accum, self.denom,
        #                     self.spatial_lr_scale, self._t, self._scaling_t, self._rotation_r, self.rot_4d, self.env_map, self.active_sh_degree_t], None)
        #gaussians.active_sh_degree = self.active_sh_degree
        state_dict = {}
        if self._xyz in self.optimizer.state:
            state_dict["xyz"] = self.optimizer.state[self._xyz]
            del self.optimizer.state[self._xyz]
        #gaussians._xyz = nn.Parameter(self._xyz.cuda())

        if self._features_dc in self.optimizer.state:
            state_dict["f_dc"] = self.optimizer.state[self._features_dc]
            del self.optimizer.state[self._features_dc]
        #gaussians._features_dc = nn.Parameter(self._features_dc.cuda())

        if self._features_rest in self.optimizer.state:
            state_dict["f_rest"] = self.optimizer.state[self._features_rest]
            del self.optimizer.state[self._features_rest]
        #gaussians._features_rest = nn.Parameter(self._features_rest.cuda())

        if self._scaling in self.optimizer.state:
            state_dict["scaling"] = self.optimizer.state[self._scaling]
            del self.optimizer.state[self._scaling]
        #gaussians._scaling = nn.Parameter(self._scaling.cuda())

        if self._rotation in self.optimizer.state:
            state_dict["rotation"] = self.optimizer.state[self._rotation]
            del self.optimizer.state[self._rotation]
        #gaussians._rotation = nn.Parameter(self._rotation.cuda())

        if self._opacity in self.optimizer.state:
            state_dict["opacity"] = self.optimizer.state[self._opacity]
            del self.optimizer.state[self._opacity]
        #gaussians._opacity = nn.Parameter(self._opacity.cuda())

        #gaussians.max_radii2D = self.max_radii2D.cuda()

        #gaussians.xyz_gradient_accum = self.xyz_gradient_accum.cuda()
        #gaussians.t_gradient_accum = self.t_gradient_accum.cuda()
        #gaussians.denom = self.denom.cuda()
        #gaussians.spatial_lr_scale = self.spatial_lr_scale.cuda()
        if self._t in self.optimizer.state:
            state_dict["t"] = self.optimizer.state[self._t]
            del self.optimizer.state[self._t]
        #gaussians._t = nn.Parameter(self._t.cuda())

        if self._scaling_t in self.optimizer.state:
            state_dict["scaling_t"] = self.optimizer.state[self._scaling_t]
            del self.optimizer.state[self._scaling_t]
        #gaussians._scaling_t = nn.Parameter(self._scaling_t.cuda())

        if self._rotation_r in self.optimizer.state:
            state_dict["rotation_r"] = self.optimizer.state[self._rotation_r]
            del self.optimizer.state[self._rotation_r]
        #gaussians._rotation_r = nn.Parameter(self._rotation_r.cuda())

        #gaussians.rot_4d = self.rot_4d
        # if self.env_map is not None:
        #     gaussians.env_map = self.env_map.cuda()
        #gaussians.active_sh_degree_t = self.active_sh_degree_t
        #gaussians.percent_dense = self.percent_dense
        #new_gaussian.optimizer = self.optimizer
        #gaussians.max_sh_degree = self.max_sh_degree
        #gaussians.gaussian_dim = self.gaussian_dim
        #gaussians.time_duration = self.time_duration
        #gaussians.force_sh_3d = self.force_sh_3d
        #gaussians.max_sh_degree_t = self.max_sh_degree_t
        return state_dict
        

    def reset_param_groups(self):
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["params"][0] = self._xyz
            if group["name"] == "f_dc":
                group["params"][0] = self._features_dc
            if group["name"] == "f_rest":
                group["params"][0] = self._features_rest
            if group["name"] == "opacity":
                group["params"][0] = self._opacity
            if group["name"] == "scaling":
                group["params"][0] = self._scaling
            if group["name"] == "rotation":
                group["params"][0] = self._rotation
            if group["name"] == "t":
                group["params"][0] = self._t
            if group["name"] == "scaling_t":
                group["params"][0] = self._scaling_t
            if group["name"] == "rotation_r":
                group["params"][0] = self._rotation_r

    def get_param_group_corresponding_attr(self, group_name):
        if group_name == "xyz":
            return self._xyz
        if group_name == "f_dc":
            return self._features_dc
        if group_name == "f_rest":
            return self._features_rest
        if group_name == "opacity":
            return self._opacity
        if group_name == "scaling":
            return self._scaling
        if group_name == "rotation":
            return self._rotation
        if group_name == "t":
            return self._t
        if group_name == "scaling_t":
            return self._scaling_t
        if group_name == "rotation_r":
            return self._rotation_r
        return None

    def append_from_gaussians_gpu(self, mask : torch.Tensor, gaussians : "GaussianModel", new_gaussians : "GaussianModel"):
        if new_gaussians is None:
            self._xyz = torch.cat([self._xyz, gaussians._xyz[mask].cpu()])
            self._features_dc = torch.cat([self._features_dc, gaussians._features_dc[mask].cpu()])
            self._features_rest = torch.cat([self._features_rest, gaussians._features_rest[mask].cpu()])
            self._scaling = torch.cat([self._scaling, gaussians._scaling[mask].cpu()])
            self._rotation = torch.cat([self._rotation, gaussians._rotation[mask].cpu()])
            self._opacity = torch.cat([self._opacity, gaussians._opacity[mask].cpu()])
            self.max_radii2D = torch.cat([self.max_radii2D, gaussians.max_radii2D[mask].cpu()])
            self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].cpu()])
            self.t_gradient_accum = torch.cat([self.t_gradient_accum, gaussians.t_gradient_accum[mask].cpu()])
            self.denom = torch.cat([self.denom, gaussians.denom[mask].cpu()])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).cpu()
            self._t = torch.cat([self._t, gaussians._t[mask].cpu()])
            self._scaling_t = torch.cat([self._scaling_t, gaussians._scaling_t[mask].cpu()])
            self._rotation_r = torch.cat([self._rotation_r, gaussians._rotation_r[mask].cpu()])
            # if gaussians.env_map is not None:
            #     self.env_map = torch.cat([self.env_map, gaussians.env_map[mask]]).cpu()
            for group in gaussians.optimizer.param_groups:
                optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
                if optimizer_state is not None:
                    state_content = {}
                    state_content["step"] = optimizer_state["step"].cpu()
                    state_content["exp_avg"] = optimizer_state["exp_avg"][mask].cpu()
                    state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].cpu()
                    #attr = self.get_param_group_corresponding_attr(group["name"])
                    tgh_gaussian_state = self.opt_states.get(group["name"], None)
                    if tgh_gaussian_state is not None:
                        tgh_gaussian_state["step"] = state_content["step"]
                        tgh_gaussian_state["exp_avg"] = torch.cat([tgh_gaussian_state["exp_avg"], state_content["exp_avg"]])
                        tgh_gaussian_state["exp_avg_sq"] = torch.cat([tgh_gaussian_state["exp_avg_sq"], state_content["exp_avg_sq"]])
                        #self.opt_states[group["name"]] = tgh_gaussian_state
                    else:
                        self.opt_states[group["name"]] = state_content
        else:
            new_gaussians._xyz = torch.cat([new_gaussians._xyz, self._xyz, gaussians._xyz[mask].cpu()])
            new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, self._features_dc, gaussians._features_dc[mask].cpu()])
            new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, self._features_rest, gaussians._features_rest[mask].cpu()])
            new_gaussians._scaling = torch.cat([new_gaussians._scaling, self._scaling, gaussians._scaling[mask].cpu()])
            new_gaussians._rotation = torch.cat([new_gaussians._rotation, self._rotation, gaussians._rotation[mask].cpu()])
            new_gaussians._opacity = torch.cat([new_gaussians._opacity, self._opacity, gaussians._opacity[mask].cpu()])
            new_gaussians.max_radii2D = torch.cat([new_gaussians.max_radii2D, self.max_radii2D, gaussians.max_radii2D[mask].cpu()])
            new_gaussians.xyz_gradient_accum = torch.cat([new_gaussians.xyz_gradient_accum, self.xyz_gradient_accum, gaussians.xyz_gradient_accum[mask].cpu()])
            new_gaussians.t_gradient_accum = torch.cat([new_gaussians.t_gradient_accum, self.t_gradient_accum, gaussians.t_gradient_accum[mask].cpu()])
            new_gaussians.denom = torch.cat([new_gaussians.denom, self.denom, gaussians.denom[mask].cpu()])
            #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale[mask]]).cpu()
            new_gaussians._t = torch.cat([new_gaussians._t, self._t, gaussians._t[mask].cpu()])
            new_gaussians._scaling_t = torch.cat([new_gaussians._scaling_t, self._scaling_t, gaussians._scaling_t[mask].cpu()])
            new_gaussians._rotation_r = torch.cat([new_gaussians._rotation_r, self._rotation_r, gaussians._rotation_r[mask].cpu()])
            # if gaussians.env_map is not None:
            #     self.env_map = torch.cat([self.env_map, gaussians.env_map[mask]]).cpu()
            # for group in gaussians.optimizer.param_groups:
            #     optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            #     if optimizer_state is not None:
            #         state_content = {}
            #         state_content["step"] = optimizer_state["step"].cpu()
            #         state_content["exp_avg"] = optimizer_state["exp_avg"][mask].cpu()
            #         state_content["exp_avg_sq"] = optimizer_state["exp_avg_sq"][mask].cpu()
            #         #attr = self.get_param_group_corresponding_attr(group["name"])
            #         tgh_gaussian_state = self.opt_states.get(group["name"], None)
            #         if tgh_gaussian_state is not None:
            #             state_content["exp_avg"] = torch.cat([new_gaussians.opt_states.get(group["name"])["exp_avg"], tgh_gaussian_state["exp_avg"], state_content["exp_avg"]])
            #             state_content["exp_avg_sq"] = torch.cat([new_gaussians.opt_states.get(group["name"])["exp_avg_sq"], tgh_gaussian_state["exp_avg_sq"], state_content["exp_avg_sq"]])
            #             #self.opt_states[group["name"]] = tgh_gaussian_state
            #         new_gaussians.opt_states[group["name"]] = state_content

        

    def append_from_gaussians_cpu(self, gaussians : "GaussianModel"):
        self._xyz = nn.Parameter(torch.cat([self._xyz, gaussians._xyz.cuda()]))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc, gaussians._features_dc.cuda()]))
        self._features_rest = nn.Parameter(torch.cat([self._features_rest, gaussians._features_rest.cuda()]))
        self._scaling = nn.Parameter(torch.cat([self._scaling, gaussians._scaling.cuda()]))
        self._rotation = nn.Parameter(torch.cat([self._rotation, gaussians._rotation.cuda()]))
        self._opacity = nn.Parameter(torch.cat([self._opacity, gaussians._opacity.cuda()]))
        self.max_radii2D = torch.cat([self.max_radii2D, gaussians.max_radii2D.cuda()])
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, gaussians.xyz_gradient_accum.cuda()])
        self.t_gradient_accum = torch.cat([self.t_gradient_accum, gaussians.t_gradient_accum.cuda()])
        self.denom = torch.cat([self.denom, gaussians.denom.cuda()])
        #self.spatial_lr_scale = torch.cat([self.spatial_lr_scale, gaussians.spatial_lr_scale]).cuda()
        self._t = nn.Parameter(torch.cat([self._t, gaussians._t.cuda()]))
        self._scaling_t = nn.Parameter(torch.cat([self._scaling_t, gaussians._scaling_t.cuda()]))
        self._rotation_r = nn.Parameter(torch.cat([self._rotation_r, gaussians._rotation_r.cuda()]))
        # if gaussians.env_map is not None:
        #     self.env_map = torch.cat([self.env_map, gaussians.env_map.cuda()])

    def append_state_from_gaussian_cpu(self, gaussians : "GaussianModel", state_dict):
        for k, v in gaussians.opt_states.items():
            #optimizer_state = gaussians.optimizer.state.get(group["params"][0], None)
            state_content = {}
            state_content["step"] = v["step"].cuda()
            state_content["exp_avg"] = v["exp_avg"].cuda()
            state_content["exp_avg_sq"] = v["exp_avg_sq"].cuda()
            attr = self.get_param_group_corresponding_attr(k)
            self_state = self.optimizer.state.get(attr, None)
            if self_state is not None and len(self_state) != 0:
                self_state["step"] = state_content["step"]
                self_state["exp_avg"] = torch.cat([self_state["exp_avg"], state_content["exp_avg"]])
                self_state["exp_avg_sq"] = torch.cat([self_state["exp_avg_sq"], state_content["exp_avg_sq"]])
            else:
                if state_dict is not None:
                    previous_state = state_dict[k]
                    previous_state["step"] = state_content["step"]
                    previous_state["exp_avg"] = state_content["exp_avg"]
                    previous_state["exp_avg_sq"] = state_content["exp_avg_sq"]
                    self.optimizer.state[attr] = previous_state

    def clone_state_from_gaussians_cpu(self, gaussians_segments, state_dict):
        step_dict = {}
        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        for segment in gaussians_segments:
            for k,v in segment.opt_states.items():
                if k not in step_dict:
                    step_dict[k] = []
                if k not in exp_avg_dict:
                    exp_avg_dict[k] = []
                if k not in exp_avg_sq_dict:
                    exp_avg_sq_dict[k] = []
                step_dict[k].append(v["step"].cuda())
                exp_avg_dict[k].append(v["exp_avg"].cuda())
                exp_avg_sq_dict[k].append(v["exp_avg_sq"].cuda())
        for k,v in gaussians_segments[0].opt_states.items():
            attr = self.get_param_group_corresponding_attr(k)
            previous_state = state_dict[k]
            #previous_state["step"] = step_dict[k][0]
            previous_state["exp_avg"] = torch.cat(exp_avg_dict[k])
            previous_state["exp_avg_sq"] = torch.cat(exp_avg_sq_dict[k])
            self.optimizer.state[attr] = previous_state
                

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim = 1))
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim = 1)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_max_sh_channels(self):
        if self.gaussian_dim == 3 or self.force_sh_3d:
            return (self.max_sh_degree+1)**2
        elif self.gaussian_dim == 4 and self.max_sh_degree_t == 0:
            return sh_channels_4d[self.max_sh_degree]
        elif self.gaussian_dim == 4 and self.max_sh_degree_t > 0:
            return (self.max_sh_degree+1)**2 * (self.max_sh_degree_t + 1)
    
    def get_cov_t(self, scaling_modifier = 1):
        if self.rot_4d:
            L = build_scaling_rotation_4d(scaling_modifier * self.get_scaling_xyzt, self._rotation, self._rotation_r)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance[:,3,3].unsqueeze(1)
        else:
            return self.get_scaling_t * scaling_modifier

    def get_marginal_t(self, timestamp, scaling_modifier = 1): # Standard
        sigma = self.get_cov_t(scaling_modifier)
        return torch.exp(-0.5*(self.get_t-timestamp)**2/sigma) # / torch.sqrt(2*torch.pi*sigma)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_current_covariance_and_mean_offset(self, scaling_modifier = 1, timestamp = 0.0):
        return self.covariance_activation(self.get_scaling_xyzt, scaling_modifier, 
                                                              self._rotation, 
                                                              self._rotation_r,
                                                              dt = timestamp - self.get_t)
    def get_current_cov_and_mean_t(self, scaling_modifier = 1):
        return self.get_t, self.cov_t_activation(self.get_scaling_xyzt, scaling_modifier, self._rotation, self._rotation_r)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pcd_cpu(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cpu") * 1.2 - 0.1) * (self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).float()
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cpu")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            # dist_t = torch.clamp_min(distCUDA2(fused_times.repeat(1,3)), 1e-10)[...,None]
            dist_t = torch.zeros_like(fused_times, device="cpu") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cpu")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cpu"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cpu")
        
        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def create_from_pth(self, path, spatial_lr_scale):
        assert self.gaussian_dim == 4 and self.rot_4d
        self.spatial_lr_scale = spatial_lr_scale
        init_4d_gaussian = torch.load(path)
        fused_point_cloud = init_4d_gaussian['xyz'].cuda()
        features_dc = init_4d_gaussian['features_dc'].cuda()
        features_rest = init_4d_gaussian['features_rest'].cuda()
        fused_times = init_4d_gaussian['t'].cuda()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        scales = init_4d_gaussian['scaling'].cuda()
        rots = init_4d_gaussian['rotation'].cuda()
        scales_t = init_4d_gaussian['scaling_t'].cuda()
        rots_r = init_4d_gaussian['rotation_r'].cuda()

        opacities = init_4d_gaussian['opacity'].cuda()
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.transpose(1, 2).requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        self._t = nn.Parameter(fused_times.requires_grad_(True))
        self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
        self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.gaussian_dim == 4: # TODO: tune time_lr_scale
            if training_args.position_t_lr_init < 0:
                training_args.position_t_lr_init = training_args.position_lr_init
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            l.append({'params': [self._t], 'lr': training_args.position_t_lr_init * self.spatial_lr_scale, "name": "t"})
            l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})
            if self.rot_4d:
                l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr, "name": "rotation_r"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            # if param_group["name"] == "t" and self.gaussian_dim == 4:
            #     lr = self.xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            #     return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }
        if self.gaussian_dim == 4:
            d["t"] = new_t
            d["scaling_t"] = new_scaling_t
            if self.rot_4d:
                d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(padded_grad >= grad_threshold, True, False).sum()}, num_to_split_pos: {selected_pts_mask.sum()}")
        
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N,1)
                means_t = torch.zeros((stds_t.size(0), 1),device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 4),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,1,1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N, 1)
            new_xyz = new_xyzt[...,0:3]
            new_t = new_xyzt[...,3:4]
            new_scaling_t = self.scaling_inverse_activation(self.get_scaling_t[selected_pts_mask].repeat(N,1) / (0.8*N))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_t, grad_t_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(f"num_to_densify_pos: {torch.where(grads >= grad_threshold, True, False).sum()}, num_to_clone_pos: {selected_pts_mask.sum()}")
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_rotation_r)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None, prune_only=False):
        if not prune_only:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            if self.gaussian_dim == 4:
                grads_t = self.t_gradient_accum / self.denom
                grads_t[grads_t.isnan()] = 0.0
            else:
                grads_t = None

            self.densify_and_clone(grads, max_grad, extent, grads_t, max_grad_t)
            self.densify_and_split(grads, max_grad, extent, grads_t, max_grad_t)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]
        
    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def set_current_timestamp(self, current_timestamp : float):
        self.current_timestamp = current_timestamp