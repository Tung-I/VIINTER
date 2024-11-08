import torch
import numpy as np
import imageio
import os
import os.path as osp
from PIL import Image
import argparse
from networks import VIINTER
from utils import linterp
from tqdm import tqdm
import time
from fvcore.nn import FlopCountAnalysis

device = torch.device('cuda')

# Measure FLOPs of the model
def measure_flops(net, grid_inp, zi):
    # Only measure FLOPs of a single forward pass with `grid_inp` and `zi`
    with torch.no_grad():
        flops = FlopCountAnalysis(net, (grid_inp, zi))
    return flops.total()

def main(args):
    model_path = args.model_path
    lin_sample_num = args.lin_sample_num
    data_dir = args.data_dir
    dataset = args.dataset
    inter_fn = linterp

    output_dir = osp.join(data_dir, 'viinter/inference')
    frames_dir = osp.join(output_dir, 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if dataset == 'dynerf':
        H, W = (1014, 1352)
    elif dataset == 'llff':
        H, W = (756, 1008)
    elif dataset == 'mipnerf360':
        # H, W = (822, 1237)
        H, W = (1039, 1558)
        # H, W = (519, 779)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    # Load model
    net = VIINTER(n_emb=2, norm_p=args.p, inter_fn=inter_fn, D=args.D, 
                  z_dim = args.z_dim, in_feat=2, out_feat=3, W=args.W, with_res=False, with_norm=True)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # Prepare grid
    coords_h = np.linspace(-1, 1, H, endpoint=False)
    coords_w = np.linspace(-1, 1, W, endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
    grid_inp = torch.FloatTensor(xy_grid).view(-1, 2).contiguous().unsqueeze(0).to(device)

    # Infer
    frames_out = []
    flops_sum = 0
    start_time = time.time()
    with torch.no_grad():
        z0 = net.ret_z(torch.LongTensor([0]).to(device)).squeeze()
        z1 = net.ret_z(torch.LongTensor([1]).to(device)).squeeze()

        for a in tqdm(torch.linspace(0, 1, lin_sample_num)):
            zi = inter_fn(a, z0, z1).unsqueeze(0)
            out = torch.zeros((grid_inp.shape[-2], 3))
            _b = 8192 * 4
            for ib in range(0, len(out), _b):
                out[ib:ib+_b] = net.forward_with_z(grid_inp[:, ib:ib+_b], zi).cpu()
            generated = torch.clamp(out.view(H, W, 3), 0, 1).numpy()  # Adjust 480, 640 to your grid size
            frames_out.append(np.uint8(255 * np.clip(generated, 0, 1)))

            flops = measure_flops(net, grid_inp[:, ib:ib+_b], zi)
            flops_sum += flops

    infer_time = time.time() - start_time
    infer_fps = lin_sample_num / infer_time
    print(f'Inference time: {infer_time:.2f}s, FPS: {infer_fps:.2f}')
    print(f'Average GFLOPs: {flops_sum / lin_sample_num / 1e9:.2f}')
    if args.save_frames:
        # Save frames as GIF
        imageio.mimsave(f'{output_dir}/animation.gif', frames_out, fps=21)

        # Save each frame as PNG
        for i, f in enumerate(frames_out):
            imageio.imsave(f'{frames_dir}/{i:03d}.png', f)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer and save intermediate frames from a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to save the output frames.')
    parser.add_argument('--lin_sample_num', type=int, default=100, help='Number of intermediate frames to generate.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=8, type=int)
    parser.add_argument('--z_dim', default=128, type=int)
    parser.add_argument('--save_frames', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
