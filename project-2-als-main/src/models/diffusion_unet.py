import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
import math 

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class _DecoderBlockDiffusion(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 skip_connection_channels,
                 time_embeeding_dim,
                 ):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2),
        )
        
        conv_channels = (in_channels // 2) + skip_connection_channels

        self.time_embeeding_conv = nn.Conv2d(in_channels=time_embeeding_dim, out_channels=out_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels=conv_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),   
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_connection, time_embeeding):
        t = time_embeeding.unsqueeze(2).unsqueeze(3)
        t = self.time_embeeding_conv(t)
       
        out = self.up(x)

        if out.shape != skip_connection.shape : 
            out = nn.functional.interpolate(out, size=(skip_connection.shape[2:]), mode='bilinear', align_corners=False)

        out = torch.cat((out, skip_connection), dim=1)

        out = self.conv1(out)
        out = out + t
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)

        return out
    
class DiffusionUNet(nn.Module):
    def __init__(self, encoder, width, height, initial_channels, time_embedder):
        super().__init__()

        self.time_embedding = time_embedder

        self.encoder = encoder
        self.return_nodes = {
            'conv1': 'conv1',
            'relu': 'x0',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4', 
        }
        self.extractor = create_feature_extractor(encoder, return_nodes=self.return_nodes)
        self.adatper = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        
        _dummy = torch.zeros((initial_channels, width, height)).unsqueeze(0)
        _dummy = self.adatper(_dummy)
        _out_dummy = self.extractor(_dummy)

        self._x0_channels = _out_dummy['x0'].shape[1]

        self._network_g_channels = _out_dummy['conv1'].shape[1]
        self._layer1_channels = _out_dummy['layer1'].shape[1]
        self._layer2_channels = _out_dummy['layer2'].shape[1]
        self._layer3_channels = _out_dummy['layer3'].shape[1]
        self._layer4_channels = _out_dummy['layer4'].shape[1]
        
        self.network_g = self.encoder.conv1
        self.network_f = nn.Conv2d(in_channels=1, out_channels=self._network_g_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.decode_layer4 = _DecoderBlockDiffusion(self._layer4_channels, self._layer3_channels, skip_connection_channels=self._layer3_channels, time_embeeding_dim=time_embedder.dim)
        self.decode_layer3 = _DecoderBlockDiffusion(self._layer3_channels, self._layer2_channels, skip_connection_channels=self._layer2_channels, time_embeeding_dim=time_embedder.dim)
        self.decode_layer2 = _DecoderBlockDiffusion(self._layer2_channels, self._layer1_channels, skip_connection_channels=self._layer1_channels, time_embeeding_dim=time_embedder.dim)
        self.decode_layer1 = _DecoderBlockDiffusion(self._layer1_channels, self._x0_channels, skip_connection_channels=self._x0_channels, time_embeeding_dim=time_embedder.dim)
        
        self.out_up = nn.ConvTranspose2d(in_channels=self._x0_channels, out_channels=3, kernel_size=2, stride=2)

        self.decode_out = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if name.startswith("encoder"):
                continue
            
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, mask, image, t):
        # out = self.adatper(x)

        out_g = self.network_g(image)
        out_f = self.network_f(mask)

        combined_input = out_g + out_f

        out = self.encoder.bn1(combined_input)
        relu_out = self.encoder.relu(out)
        out = self.encoder.maxpool(relu_out)

        layer1_out = self.encoder.layer1(out)
        layer2_out = self.encoder.layer2(layer1_out)
        layer3_out = self.encoder.layer3(layer2_out)
        layer4_out = self.encoder.layer4(layer3_out)

        # encoder = self.extractor(out)

        time_embedding = self.time_embedding(t)
        
        out = self.decode_layer4(layer4_out, layer3_out, time_embedding)
        out = self.decode_layer3(out, layer2_out, time_embedding)
        out = self.decode_layer2(out, layer1_out, time_embedding)
        out = self.decode_layer1(out, relu_out, time_embedding)
        
        out = self.out_up(out)
        
        # if out.shape != x.shape : 
        #     out = nn.functional.interpolate(out, size=(x.shape[2:]), mode='bilinear', align_corners=False)
        
        # clean_image = x[:, 1:, : , :]
        # out = torch.cat((out, clean_image), dim=1)
        out = self.decode_out(out)
        return out

