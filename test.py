import argparse
import os

import torch
from torch.autograd import Variable

from model import G
from util import is_image, load_image, save_image

parser = argparse.ArgumentParser(description='DeepRendering-implementation')
parser.add_argument('--dataset', required=True, help='unity')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--n_channel_input', type=int, default=3, help='input channel')
parser.add_argument('--n_channel_output', type=int, default=3, help='output channel')
parser.add_argument('--n_generator_filters', type=int, default=64, help="number of generator filters")
opt = parser.parse_args()

netG_model = torch.load(opt.model)
netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
netG.load_state_dict(netG_model['state_dict_G'])
root_dir = 'dataset/{}/test/'.format(opt.dataset)
image_dir = 'dataset/{}/test/albedo'.format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image(x)]

for image_name in image_filenames:
    albedo_image = load_image(root_dir + 'albedo/' + image_name)
    direct_image = load_image(root_dir + 'direct/' + image_name)
    normal_image = load_image(root_dir + 'normal/' + image_name)
    depth_image = load_image(root_dir + 'depth/' + image_name)
    gt_image = load_image(root_dir + 'gt/' + image_name)

    albedo = Variable(albedo_image).view(1, -1, 256, 256).cuda()
    direct = Variable(direct_image).view(1, -1, 256, 256).cuda()
    normal = Variable(normal_image).view(1, -1, 256, 256).cuda()
    depth = Variable(depth_image).view(1, -1, 256, 256).cuda()
    
    netG = netG.cuda()


    out = netG(torch.cat((albedo, direct, normal, depth), 1))
    out = out.cpu()
    out_img = out.data[0]

    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    save_image(out_img, "result/{}/{}".format(opt.dataset, image_name))