import math
from scene.gaussian_segment import GaussianSegment
from scene.gaussian_model import GaussianModel
import torch
import time

class TemperalGaussianHierarchy():

    def __init__(self, sh_degree : int, level_count : int, max_layer_length : float, gaussian_dim : int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False, 
                 force_sh_3d: bool = False, sh_degree_t : int = 0):
        self.sh_degree = sh_degree
        self.gaussian_dim = gaussian_dim
        self.rot_4d = rot_4d
        self.force_sh_3d = force_sh_3d
        self.sh_degree_t = sh_degree_t
        self.level_count = level_count
        self.max_layer_length = max_layer_length
        self.time_duration = time_duration
        static_layer = GaussianModel(sh_degree, gaussian_dim, time_duration, rot_4d, force_sh_3d, sh_degree_t)
        self.layers = [[static_layer]]
        for level in range(1, level_count + 1):
            current_layer_length = max_layer_length / 2**(level - 1)
            current_layer = []
            segment_count = math.ceil((time_duration[1] - time_duration[0]) / current_layer_length)
            for ind in range(segment_count):
                segment = GaussianModel(sh_degree, gaussian_dim, time_duration, rot_4d, force_sh_3d, sh_degree_t)
                current_layer.append(segment)

            self.layers.append(current_layer)

    def create_from_gaussians(self, gaussians : GaussianModel, opt, o_th : float = 0.05):
        mean_t, cov_t = gaussians.get_current_cov_and_mean_t()
        #o_th_cuda = torch.full((), o_th)
        effect_range = torch.sqrt(-2 * torch.log(torch.tensor(o_th, device="cuda"))) * torch.abs(cov_t)
        gaussians_start = torch.clamp(mean_t - effect_range, min=0)
        gaussians_end = torch.clamp(mean_t + effect_range, min= 0)
        gaussians_last_level_ind = torch.zeros(gaussians_start.shape[0], device="cuda")
        mask = torch.full(gaussians_start.shape, True, dtype=torch.bool, device="cuda").squeeze(-1)
        # gaussians_last_level_end = torch.empty(gaussians_end.shape[0])
        # mask = torch.empty(self.level_count, gaussians_end.shape[0])
        for level in range(1, self.level_count + 1):
            last_layer_length = 0 if level == 1 else self.max_layer_length / 2**(level - 2)
            current_length = self.max_layer_length / 2**(level - 1)
            gaussians_level_start = torch.floor(gaussians_start / current_length).squeeze(-1)
            gaussians_level_end = torch.floor(gaussians_end / current_length).squeeze(-1)
            #mask = (mask & (gaussians_level_start != gaussians_level_end))
            last_segment_count = 1 if level == 1 else math.ceil((self.time_duration[1] - self.time_duration[0]) / last_layer_length)
            for ind in range(last_segment_count):
                self.layers[level - 1][ind].clone_by_mask(mask & (gaussians_level_start != gaussians_level_end) & (ind == gaussians_last_level_ind), gaussians, opt, None)
            
            #mask = ~mask
            mask = (mask & (gaussians_level_start == gaussians_level_end))

            if level == self.level_count:
                for ind in range(math.ceil((self.time_duration[1] - self.time_duration[0]) / current_length)):
                    self.layers[level][ind].clone_by_mask(mask & (ind == gaussians_level_start), gaussians, opt, None)
            gaussians_last_level_ind = gaussians_level_start

    def update_from_gaussians(self, gaussians : GaussianModel, opt, new_gaussians : GaussianModel, o_th : float = 0.05):
        mean_t, cov_t = gaussians.get_current_cov_and_mean_t()
        effect_range = torch.sqrt(-2 * torch.log(torch.tensor(o_th, device="cuda"))) * torch.abs(cov_t)
        gaussians_start = torch.clamp(mean_t - effect_range, min=0)
        gaussians_end = torch.clamp(mean_t + effect_range, min= 0)
        gaussians_last_level_ind = torch.zeros(gaussians_start.shape[0], dtype=torch.int64, device = "cuda")
        mask = torch.full(gaussians_start.shape, True, dtype=torch.bool, device="cuda").squeeze(-1)
        for level in range(1, self.level_count + 1):
            #level_start = time.time()
            #prepare_start = time.time()
            last_layer_length = 0 if level == 1 else self.max_layer_length / 2**(level - 2)
            current_length = self.max_layer_length / 2**(level - 1)
            gaussians_level_start = torch.floor(gaussians_start / current_length).to(torch.int64).squeeze(-1)
            gaussians_level_end = torch.floor(gaussians_end / current_length).to(torch.int64).squeeze(-1)
            #mask = mask & (gaussians_level_start != gaussians_level_end)
            #last_segment_count = 1 if level == 1 else math.ceil((self.time_duration[1] - self.time_duration[0]) / last_layer_length)
            active_mask = mask & (gaussians_level_start != gaussians_level_end)
            replace_ind = 0 if level == 1 else math.floor(gaussians.current_timestamp / last_layer_length)
            segments_for_update = torch.unique(gaussians_last_level_ind[active_mask], sorted = False)
            #prepare_end = time.time()
            #torch.cuda.synchronize()
            #print(f"prepare timne: {prepare_end - prepare_start:.6f} seconds")
            replace_flag = False
            for ind in segments_for_update:
                #mask_compute_start1 = time.time()
                idx = ind.item()
                actual_mask = active_mask & (idx == gaussians_last_level_ind)
                #mask_compute_end1 = time.time()
                #torch.cuda.synchronize()
                #print(f"mask compute time1: {mask_compute_end1 - mask_compute_start1:.6f} seconds")
                #mem_copy_start = time.time()
                if idx == replace_ind:
                    self.layers[level - 1][idx].clone_by_mask(actual_mask, gaussians, opt, new_gaussians)
                    replace_flag = True
                else:
                    self.layers[level - 1][idx].append_from_gaussians_gpu(actual_mask, gaussians, new_gaussians)
            if not replace_flag:
                self.layers[level - 1][replace_ind].clone_by_mask(active_mask & (replace_ind == gaussians_last_level_ind), gaussians, opt, new_gaussians)
                #mem_copy_end = time.time()
                #torch.cuda.synchronize()
                #print(f"memory copy time: {mem_copy_end - mem_copy_start:.6f} seconds")
                #print("test")
            #mask_compute_start2 = time.time()
            
            mask = (mask & (gaussians_level_start == gaussians_level_end))
            #mask_compute_end2 = time.time()
            #torch.cuda.synchronize()
            #print(f"mask compute time2: {mask_compute_end2 - mask_compute_start2:.6f} seconds")
            #level_end = time.time()
            #torch.cuda.synchronize()
            #print(f"level time:{level_end - level_start:.6f}second")
            replace_flag = False
            if level == self.level_count:
                segments_for_update = torch.unique(gaussians_level_start[mask], sorted = False)
                for ind in segments_for_update:
                    idx = ind.item()
                    if idx == math.floor(gaussians.current_timestamp / current_length):
                        self.layers[level][idx].clone_by_mask(mask & (idx == gaussians_level_start), gaussians, opt, new_gaussians)
                        replace_flag = True
                    else:
                        self.layers[level][idx].append_from_gaussians_gpu(mask & (idx == gaussians_level_start), gaussians, new_gaussians)
                if not replace_flag:
                    replace_idx = math.floor(gaussians.current_timestamp / current_length)
                    self.layers[level][replace_idx].clone_by_mask(mask & (replace_idx == gaussians_level_start), gaussians, opt, new_gaussians)
            gaussians_last_level_ind = gaussians_level_start

    def put_current_related_gaussians(self, timestamp : float, gaussians : "GaussianModel"):
        gaussians.set_current_timestamp(timestamp)
        state_dict = gaussians.get_state_dict()
        #state_dict = self.layers[0][0].clone_to(gaussians)
        gaussians_segments = [self.layers[0][0]]
        for level in range(1, self.level_count + 1):
            current_ind = math.floor(timestamp / (self.max_layer_length / 2**(level - 1)))
            gaussians_segments.append(self.layers[level][current_ind])
            #gaussians.append_from_gaussians_cpu(self.layers[level][current_ind])
        gaussians.clone_from_cpu(gaussians_segments)
        gaussians.reset_param_groups()
        gaussians.clone_state_from_gaussians_cpu(gaussians_segments, state_dict)
        # gaussians.append_state_from_gaussian_cpu(self.layers[0][0], state_dict)
        # for level in range(1, self.level_count + 1):
        #     current_ind = math.floor(timestamp / (self.max_layer_length / 2**(level - 1)))
        #     gaussians.append_state_from_gaussian_cpu(self.layers[level][current_ind], None)
        torch.cuda.empty_cache()

    def capture(self, gaussians : GaussianModel, opt):
        new_gaussians = GaussianModel(self.sh_degree, self.gaussian_dim, self.time_duration, self.rot_4d, self.force_sh_3d, self.sh_degree_t)
        self.update_from_gaussians(gaussians, opt, new_gaussians)
        # active_sh_degree = gaussians.active_sh_degree
        # _xyz = self.layers[0][0]._xyz
        # _features_dc = self.layers[0][0]._features_dc
        # _features_rest = self.layers[0][0]._features_rest
        # _scaling = self.layers[0][0]._scaling
        # _rotation = self.layers[0][0]._rotation
        # _opacity = self.layers[0][0]._opacity
        # max_radii2D = self.layers[0][0].max_radii2D
        # xyz_gradient_accum = self.layers[0][0].xyz_gradient_accum
        # t_gradient_accum = self.layers[0][0].t_gradient_accum
        # denom = self.layers[0][0].denom
        # opt_states = {}
        # opt_states.update(self.layers[0][0].opt_states)
        # spatial_lr_scale = gaussians.spatial_lr_scale
        # _t = self.layers[0][0]._t
        # _scaling_t = self.layers[0][0]._scaling_t
        # _rotation_r = self.layers[0][0]._rotation_r
        # rot_4d = gaussians.rot_4d
        if gaussians.env_map is not None:
            env_map = gaussians.env_map.cpu()
        else:
            env_map = None
        # active_sh_degree_t = gaussians.active_sh_degree_t
        # for level in range(1, self.level_count + 1):
        #     current_length = self.max_layer_length / 2**(level - 1)
        #     for ind in range(math.ceil((self.time_duration[1] - self.time_duration[0]) / current_length)):
        #             _xyz = torch.cat([_xyz, self.layers[level][ind]._xyz])
        #             _features_dc = torch.cat([_features_dc, self.layers[level][ind]._features_dc])
        #             _features_rest = torch.cat([_features_rest, self.layers[level][ind]._features_rest])
        #             _scaling = torch.cat([_scaling, self.layers[level][ind]._scaling])
        #             _rotation = torch.cat([_rotation, self.layers[level][ind]._rotation])
        #             _opacity = torch.cat([_opacity, self.layers[level][ind]._opacity])
        #             max_radii2D = torch.cat([max_radii2D, self.layers[level][ind].max_radii2D])
        #             xyz_gradient_accum = torch.cat([xyz_gradient_accum, self.layers[level][ind].xyz_gradient_accum])
        #             t_gradient_accum = torch.cat([t_gradient_accum, self.layers[level][ind].t_gradient_accum])
        #             denom = torch.cat([denom, self.layers[level][ind].denom])
        #             opt_states.update(self.layers[level][ind].opt_states)
        #             _t = torch.cat([_t, self.layers[level][ind]._t])
        #             _scaling_t = torch.cat([_scaling_t, self.layers[level][ind]._scaling_t])
        #             _rotation_r = torch.cat([_rotation_r, self.layers[level][ind]._rotation_r])
        return (
                gaussians.active_sh_degree,
                new_gaussians._xyz,
                new_gaussians._features_dc,
                new_gaussians._features_rest,
                new_gaussians._scaling,
                new_gaussians._rotation,
                new_gaussians._opacity,
                new_gaussians.max_radii2D,
                new_gaussians.xyz_gradient_accum,
                new_gaussians.t_gradient_accum,
                new_gaussians.denom,
                new_gaussians.opt_states,
                gaussians.spatial_lr_scale,
                new_gaussians._t,
                new_gaussians._scaling_t,
                new_gaussians._rotation_r,
                gaussians.rot_4d,
                env_map,
                gaussians.active_sh_degree_t
            )