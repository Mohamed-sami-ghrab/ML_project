import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection_channels, dropout_prob=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2),
        )
        
        conv_channels = (in_channels // 2) + skip_connection_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),   
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
        )

    def forward(self, x, skip_connection):
        out = self.up(x)
        if out.shape != skip_connection.shape: 
            out = nn.functional.interpolate(out, size=(skip_connection.shape[2:]), mode='bilinear', align_corners=False)
        out = torch.cat((out, skip_connection), dim=1)
        return self.conv(out)

class UNet(nn.Module):
    def __init__(self, encoder, width, height, initial_channels, dropout):
        super().__init__()

        self.encoder = encoder
        self.return_nodes = {
            'relu': 'x0',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4', 
        }
        self.extractor = create_feature_extractor(encoder, return_nodes=self.return_nodes)
        
        # Calculate feature map shapes
        _dummy = torch.zeros((1, initial_channels, width, height))
        _out_dummy = self.extractor(_dummy)
        
        self._x0_channels = _out_dummy['x0'].shape[1]
        self._layer1_channels = _out_dummy['layer1'].shape[1]
        self._layer2_channels = _out_dummy['layer2'].shape[1]
        self._layer3_channels = _out_dummy['layer3'].shape[1]
        self._layer4_channels = _out_dummy['layer4'].shape[1]
        
        # Decoder Layers
        self.decode_layer4 = _DecoderBlock(self._layer4_channels, self._layer3_channels, self._layer3_channels, dropout)
        self.decode_layer3 = _DecoderBlock(self._layer3_channels, self._layer2_channels, self._layer2_channels, dropout)
        self.decode_layer2 = _DecoderBlock(self._layer2_channels, self._layer1_channels, self._layer1_channels, dropout)
        self.decode_layer1 = _DecoderBlock(self._layer1_channels, self._x0_channels, self._x0_channels, dropout)
        
        # Final Upsampling: Learnable, not just interpolation
        self.out_up = nn.ConvTranspose2d(in_channels=self._x0_channels, out_channels=32, kernel_size=2, stride=2)
        
        # Final Prediction
        self.decode_out = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for module in [self.decode_layer4, self.decode_layer3, self.decode_layer2, 
                       self.decode_layer1, self.out_up, self.decode_out]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder = self.extractor(x)
        
        out = self.decode_layer4(encoder['layer4'], encoder['layer3'])
        out = self.decode_layer3(out, encoder['layer2'])
        out = self.decode_layer2(out, encoder['layer1'])
        out = self.decode_layer1(out, encoder['x0'])
        
        # Apply learnable upsampling
        out = self.out_up(out)
        
        # Final convolution
        out = self.decode_out(out)
        
        # Safety interpolation if input size was odd or rounding issues occurred
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        return out