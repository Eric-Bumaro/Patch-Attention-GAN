import copy
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import LayerNorm
from model.biformer import BiFormer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0 and gpu_ids[0] >= 0:
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class Biformer_Encoder(BiFormer):
    def __init__(self, intput_nc, dim):
        super().__init__(in_chans=intput_nc,
                         depth=[2, 2, 2],
                         embed_dim=[64, 96, 128],
                         mlp_ratios=[3, 3, 3],
                         # ------------------------------
                         n_win=4,
                         kv_downsample_mode='identity',
                         kv_per_wins=[-1, -1, -1],
                         topks=[1, 4, 16],
                         side_dwconv=5,
                         before_attn_dwconv=3,
                         layer_scale_init_value=-1,
                         qk_dims=[64, 96, 128],
                         head_dim=32,
                         param_routing=False, diff_routing=False, soft_routing=False,
                         pre_norm=True,
                         pe=None,
                         kv_downsample_kernels=[2, 2, 1],
                         kv_downsample_ratios=[2, 2, 1])
        # step 1: remove unused classifier head & norm
        del self.head  # classification head
        del self.norm  # head norm

        # step 2: add extra norms for dense tasks
        self.extra_norms = nn.ModuleList()
        scale = 4
        for i in range(len(self.embed_dim)):
            self.extra_norms.append(LayerNorm([self.embed_dim[i], dim // scale, dim // scale]))
            scale *= 2

    def forward(self, x: torch.Tensor):
        out = []
        for i in range(len(self.embed_dim)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # DONE: check the inconsistency -> no effect on performance
            # in the version before submission:
            # x = self.extra_norms[i](x)
            # out.append(x)
            out.append(self.extra_norms[i](x))
        # 96 * 64 * 64， 192 * 32 * 32， 384 * 16 * 16， 768 * 8 * 8
        return list(out)


# class Generator(nn.Module):
#     def __init__(self, encoder, init_type, init_gain, gpu_ids, output_nc=3, attention_num=8):
#         super().__init__()
#         self.encoder = encoder
#         self.embed_dim = self.encoder.embed_dim
#         self.output_nc = output_nc
#         self.attention_num = attention_num
#
#         self.decode_modules_content = nn.ModuleList()
#         embed_dim = self.embed_dim[::-1]
#         self.decode_modules_content.append(
#             nn.Sequential(nn.GELU(),
#                           nn.ConvTranspose2d(embed_dim[0], embed_dim[1], 3, 2, 1, 1),
#                           nn.InstanceNorm2d(embed_dim[1])
#                           )
#         )
#         for i in range(1, len(self.embed_dim) - 1):
#             up_block = nn.Sequential(
#                 nn.GELU(),
#                 nn.ConvTranspose2d(embed_dim[i], embed_dim[i + 1], 3, 2, 1, 1),
#                 nn.InstanceNorm2d(embed_dim[i + 1])
#             )
#             self.decode_modules_content.append(up_block)
#         self.decode_modules_content.append(
#             nn.Sequential(
#                 nn.GELU(),
#                 nn.ConvTranspose2d(embed_dim[-1], embed_dim[-1], 3, 2, 1, 1),
#                 nn.InstanceNorm2d(embed_dim[-1])
#             )
#         )
#         self.output_layer_content = nn.Sequential(
#             nn.GELU(),
#             nn.ConvTranspose2d(embed_dim[-1], output_nc * attention_num, 3, 2, 1, 1),
#             nn.Tanh()
#         )
#
#         self.decode_modules_mask = copy.deepcopy(self.decode_modules_content)
#         self.output_layer_mask = nn.Sequential(
#             nn.GELU(),
#             nn.ConvTranspose2d(embed_dim[-1], attention_num + 1, 3, 2, 1, 1),
#             nn.Softmax(dim=1)
#         )
#
#         # step 4: initialization & load ckpt
#         init_net(self, init_type, init_gain, gpu_ids)
#
#         # step 5: convert sync bn, as the batch size is too small in segmentation
#         # TODO: check if this is correct
#         if gpu_ids[0] != -1:
#             nn.SyncBatchNorm.convert_sync_batchnorm(self)
#
#     def decode_feature(self, features):
#         x_content = features[-1]
#         x_mask = features[-1]
#         for i in range(len(self.embed_dim) - 1):
#             x_content = self.decode_modules_content[i](x_content)
#             x_mask = self.decode_modules_mask[i](x_mask)
#         x_content = self.decode_modules_content[-1](x_content)
#         x_content = self.output_layer_content(x_content)
#         x_mask = self.decode_modules_mask[-1](x_mask)
#         x_mask = self.output_layer_mask(x_mask)
#         return x_content, x_mask
#
#     def forward(self, x: torch.Tensor):
#         encode = self.encoder(x)
#         content, mask = self.decode_feature(encode)
#         foreground_mask_list = []
#         foreground = 0
#         for i in range(self.attention_num):
#             content_index_start = i * self.output_nc
#             content_index_end = (i + 1) * self.output_nc
#             foreground_mask_list.append(mask[:, i, :, :])
#             mask_ = mask[:, i, :, :].repeat(1, 3, 1, 1)
#             foreground = content[:, content_index_start:content_index_end, :, :] * mask_ + foreground
#         background_mask = mask[:, -1, :, :].repeat(1, 3, 1, 1)
#         background = x * background_mask
#         output = background + foreground
#         return output, content, background_mask, foreground_mask_list
#

class resnet_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, 0)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def forward(self, input):
        x = F.pad(input, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 'reflect')
        x = self.conv2_norm(self.conv2(x))

        return input + x


class Generator(nn.Module):
    def __init__(self, encoder, init_type, init_gain, gpu_ids, input_nc=3, output_nc=3, ngf=32, n_blocks=9):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.nb = n_blocks
        self.conv1 = nn.Conv2d(input_nc, ngf, 7, 1, 0)
        self.conv1_norm = nn.InstanceNorm2d(ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 3, 2, 1)
        self.conv2_norm = nn.InstanceNorm2d(ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1)
        self.conv3_norm = nn.InstanceNorm2d(ngf * 4)

        self.resnet_blocks = []
        for i in range(n_blocks):
            self.resnet_blocks.append(resnet_block(ngf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv1_content = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_content = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_content = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_content = nn.InstanceNorm2d(ngf)
        self.deconv3_content = nn.Conv2d(ngf, 27, 7, 1, 0)

        self.deconv1_attention = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.deconv1_norm_attention = nn.InstanceNorm2d(ngf * 2)
        self.deconv2_attention = nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1)
        self.deconv2_norm_attention = nn.InstanceNorm2d(ngf)
        self.deconv3_attention = nn.Conv2d(ngf, 10, 1, 1, 0)

        self.tanh = torch.nn.Tanh()
        init_net(self, init_type, init_gain, gpu_ids)

    def init_weight(self):
        self.apply(weights_init_normal)

    def forward(self, input):
        x = F.pad(input, (3, 3, 3, 3), 'reflect')
        x = F.relu(self.conv1_norm(self.conv1(x)))
        x = F.relu(self.conv2_norm(self.conv2(x)))
        x = F.relu(self.conv3_norm(self.conv3(x)))
        x = self.resnet_blocks(x)
        # x = self.resnet_blocks1(x)
        # x = self.resnet_blocks2(x)
        # x = self.resnet_blocks3(x)
        # x = self.resnet_blocks4(x)
        # x = self.resnet_blocks5(x)
        # x = self.resnet_blocks6(x)
        # x = self.resnet_blocks7(x)
        # x = self.resnet_blocks8(x)
        # x = self.resnet_blocks9(x)
        x_content = F.relu(self.deconv1_norm_content(self.deconv1_content(x)))
        x_content = F.relu(self.deconv2_norm_content(self.deconv2_content(x_content)))
        x_content = F.pad(x_content, (3, 3, 3, 3), 'reflect')
        content = self.deconv3_content(x_content)
        image = self.tanh(content)
        image1 = image[:, 0:3, :, :]
        # print(image1.size()) # [1, 3, 256, 256]
        image2 = image[:, 3:6, :, :]
        image3 = image[:, 6:9, :, :]
        image4 = image[:, 9:12, :, :]
        image5 = image[:, 12:15, :, :]
        image6 = image[:, 15:18, :, :]
        image7 = image[:, 18:21, :, :]
        image8 = image[:, 21:24, :, :]
        image9 = image[:, 24:27, :, :]
        # image10 = image[:, 27:30, :, :]

        x_attention = F.relu(self.deconv1_norm_attention(self.deconv1_attention(x)))
        x_attention = F.relu(self.deconv2_norm_attention(self.deconv2_attention(x_attention)))
        # x_attention = F.pad(x_attention, (3, 3, 3, 3), 'reflect')
        # print(x_attention.size()) [1, 64, 256, 256]
        attention = self.deconv3_attention(x_attention)

        softmax_ = torch.nn.Softmax(dim=1)
        attention = softmax_(attention)

        attention1_ = attention[:, 0:1, :, :]
        attention2_ = attention[:, 1:2, :, :]
        attention3_ = attention[:, 2:3, :, :]
        attention4_ = attention[:, 3:4, :, :]
        attention5_ = attention[:, 4:5, :, :]
        attention6_ = attention[:, 5:6, :, :]
        attention7_ = attention[:, 6:7, :, :]
        attention8_ = attention[:, 7:8, :, :]
        attention9_ = attention[:, 8:9, :, :]
        attention10_ = attention[:, 9:10, :, :]

        attention1 = attention1_.repeat(1, 3, 1, 1)
        # print(attention1.size())
        attention2 = attention2_.repeat(1, 3, 1, 1)
        attention3 = attention3_.repeat(1, 3, 1, 1)
        attention4 = attention4_.repeat(1, 3, 1, 1)
        attention5 = attention5_.repeat(1, 3, 1, 1)
        attention6 = attention6_.repeat(1, 3, 1, 1)
        attention7 = attention7_.repeat(1, 3, 1, 1)
        attention8 = attention8_.repeat(1, 3, 1, 1)
        attention9 = attention9_.repeat(1, 3, 1, 1)
        attention10 = attention10_.repeat(1, 3, 1, 1)

        output1 = image1 * attention1
        output2 = image2 * attention2
        output3 = image3 * attention3
        output4 = image4 * attention4
        output5 = image5 * attention5
        output6 = image6 * attention6
        output7 = image7 * attention7
        output8 = image8 * attention8
        output9 = image9 * attention9
        # output10 = image10 * attention10
        output10 = input * attention10

        o = output1 + output2 + output3 + output4 + output5 + output6 + output7 + output8 + output9 + output10

        return o, content, attention10, \
               [attention1, attention2, attention3, attention4, attention5, attention6,
                attention7, attention8, attention9, attention10]


class Discriminator(nn.Module):
    def __init__(self, init_type, init_gain, gpu_ids, input_nc, ndf=32):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        super(Discriminator, self).__init__()
        use_bias = True
        n_layers = 3
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        init_net(self, init_type, init_gain, gpu_ids)

    def init_weight(self):
        self.apply(weights_init_normal)

    def forward(self, img):
        output = self.model(img)
        return output.squeeze(0).squeeze(0).squeeze(0)


if __name__ == "__main__":
    from thop import profile

    image = torch.randn(1, 3, 256, 256)
    encoder = Biformer_Encoder(intput_nc=3, dim=256)
    net = Generator(encoder, "normal", 0.02, [-1], output_nc=3)
    flops, params = profile(net, inputs=(image,))
    print('{:<30}  {:<8} GFLops'.format('Computational complexity: ', flops / 1e9))
    print('{:<30}  {:<8} KB'.format('Number of parameters: ', params / 1e3))