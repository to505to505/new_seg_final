import torch
import torch.nn as nn


def Consecutive_Conv_Block(C_in, C_out, add_batch_norm = True):
    """
    Creates a Consecutive Convolutional Block (used in U-Net's encoder and decoder layers) with/without Batch Norm

    Arguments:
        C_in: (int) Number of channels in the input image
        C_out: (int) Number of channels produced by the convolution
        add_batch_norm: (bool) Whether to add batch norm layers or not
    """
    if add_batch_norm:
        
        CC_Block = nn.Sequential(
                                 nn.Conv2d(C_in, C_out, kernel_size=3, padding="same"),
                                 nn.BatchNorm2d(num_features=C_out),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(C_out, C_out, kernel_size=3, padding="same"),
                                 nn.BatchNorm2d(num_features=C_out),
                                 nn.ReLU(inplace=True),
                                )

    else:
        
        CC_Block = nn.Sequential(
                                 nn.Conv2d(C_in, C_out, kernel_size=3, padding="same"),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(C_out, C_out, kernel_size=3, padding="same"),
                                 nn.ReLU(inplace=True),
                                )
        
    return CC_Block


class UNet(nn.Module):

    """
    Models the 5 (encoder) layer U-Net
    """

    def __init__(self, main_C_in=3, num_classes=1, add_batch_norm=True):
        """
        Constructor for the UNet class

        Arguments:
            main_C_in: (int) Number of channels in the main input image
            num_classes: (int) Number of final output classes
        """

        super(UNet, self).__init__()

        self.CC_Block1_Encoder = Consecutive_Conv_Block(main_C_in, C_out=64, add_batch_norm=add_batch_norm)
        self.CC_Block2_Encoder = Consecutive_Conv_Block(C_in=64, C_out=128, add_batch_norm=add_batch_norm)
        self.CC_Block3_Encoder = Consecutive_Conv_Block(C_in=128, C_out=256, add_batch_norm=add_batch_norm)
        self.CC_Block4_Encoder = Consecutive_Conv_Block(C_in=256, C_out=512, add_batch_norm=add_batch_norm)
        self.CC_Block5_Encoder = Consecutive_Conv_Block(C_in=512, C_out=1024, add_batch_norm=add_batch_norm)

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.UpConv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2
        )
        self.CC_Block1_Decoder = Consecutive_Conv_Block(C_in=1024, C_out=512)
        self.UpConv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2
        )
        self.CC_Block2_Decoder = Consecutive_Conv_Block(C_in=512, C_out=256)
        self.UpConv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2
        )
        self.CC_Block3_Decoder = Consecutive_Conv_Block(C_in=256, C_out=128)
        self.UpConv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2
        )
        self.CC_Block4_Decoder = Consecutive_Conv_Block(C_in=128, C_out=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        """
        Instance Method that implements the forward pass through U-Net

        Arguments:
            x = (torch.Tensor) The input image

        Returns:
            out = (torch.Tensor) The predicted logits
        """

        # Encoder Path
        e_x1 = self.CC_Block1_Encoder(x)
        e_x2 = self.MaxPool(e_x1)
        e_x3 = self.CC_Block2_Encoder(e_x2)
        e_x4 = self.MaxPool(e_x3)
        e_x5 = self.CC_Block3_Encoder(e_x4)
        e_x6 = self.MaxPool(e_x5)
        e_x7 = self.CC_Block4_Encoder(e_x6)
        e_x8 = self.MaxPool(e_x7)
        e_x9 = self.CC_Block5_Encoder(e_x8)

        # Decoder Path
        # Concatenation of upconv. output with the corresponding encoder layer output is affected along the channel-axis
        # [in torch.cat, image = (N, C, H, W) and dim = 1 ==> Channel-Axis (C)]
        u_x1 = self.UpConv1(e_x9)
        u_x2 = self.CC_Block1_Decoder(torch.cat((e_x7, u_x1), dim=1))
        u_x3 = self.UpConv2(u_x2)
        u_x4 = self.CC_Block2_Decoder(torch.cat((e_x5, u_x3), dim=1))
        u_x5 = self.UpConv3(u_x4)
        u_x6 = self.CC_Block3_Decoder(torch.cat((e_x3, u_x5), dim=1))
        u_x7 = self.UpConv4(u_x6)
        u_x8 = self.CC_Block4_Decoder(torch.cat((e_x1, u_x7), dim=1))

        # 1*1 Conv for generating the output logits
        out = self.output(u_x8)

        return out
