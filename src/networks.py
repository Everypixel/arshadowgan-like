# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Generator_with_Refin(nn.Module):
    def __init__(self, encoder):
        """Generator initialization

        Args:
            encoder: an encoder for Unet generator
        """
        super(Generator_with_Refin, self).__init__()

        # declare Unet generator
        self.generator = smp.Unet(
            encoder_name=encoder,
            classes=1,
            activation='identity',
        )
        # replace the first conv block in generator (6 channels tensor as input)
        self.generator.encoder.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.generator.segmentation_head = nn.Identity()

        # RGB-shadow mask as output before refinement module
        self.SG_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        # refinement module
        self.refinement = torch.nn.Sequential()
        for i in range(4):
            self.refinement.add_module(f'refinement{3*i+1}', nn.BatchNorm2d(16))
            self.refinement.add_module(f'refinement{3*i+2}', nn.ReLU())
            self.refinement.add_module(f'refinement{3*i+3}', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))

        # RGB-shadow mask as output after refinement module
        self.output1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward for generator

        Args:
            x: torch.FloatTensor or torch.cuda.FloatTensor - input tensor with images and masks
        """
        x = self.generator(x)
        out1 = self.SG_head(x)

        x = self.refinement(x)
        x = self.output1(x)
        return out1, x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        """Discriminator initialization

        Args:
            input_shape (tuple): shape of input image
        """
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        """Discriminator forward
        """
        return self.model(img)


class ARShadowGAN(nn.Module):
    def __init__(self, model_path_attention, model_path_SG, encoder_att='resnet34', encoder_SG='resnet18', device='cuda:0'):
        """ARShadowGAN-like initialization

        Args:
            model_path_attention: path to attention weights
            model_path_SG: path to SG weigths
            encoder_att: encoder for attention
            encoder_SG: encoder for shadow-generation
            device: device for inference
        """
        super(ARShadowGAN, self).__init__()

        self.device = torch.device(device)
        self.model_att = smp.Unet(
            classes=2,
            encoder_name=encoder_att,
            activation='sigmoid'
        )
        self.model_att.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.model_att.load_state_dict(torch.load(model_path_attention))
        self.model_att.to(device)

        self.model_SG = Generator_with_Refin(encoder_SG)
        self.model_SG.load_state_dict(torch.load(model_path_SG))
        self.model_SG.to(device)

    def forward(self, tensor_att, tensor_SG):
        """Forward for ARShadowGAN-like

        Args:
            tensor_att: tensor for attention block
            tensor_SG: tensor for SG block
        """

        # inference attention model
        self.model_att.eval()
        with torch.no_grad():
            robject_rshadow_tensor = self.model_att(tensor_att)

        robject_rshadow_np = robject_rshadow_tensor.cpu().numpy()

        robject_rshadow_np[robject_rshadow_np >= 0.5] = 1
        robject_rshadow_np[robject_rshadow_np < 0.5] = 0
        robject_rshadow_np = 2 * (robject_rshadow_np - 0.5)

        robject_rshadow_tensor = torch.cuda.FloatTensor(robject_rshadow_np)

        tensor_SG = torch.cat((tensor_SG, robject_rshadow_tensor), axis=1)

        # inference shadow-generation model
        self.model_SG.eval()
        with torch.no_grad():
            output_mask1, output_mask2 = self.model_SG(tensor_SG)

        result = torch.add(tensor_SG[:,:3, ...], output_mask2)

        return result, output_mask2
