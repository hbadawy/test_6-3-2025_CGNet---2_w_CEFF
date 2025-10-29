
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, device=None):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, device=device)
        self.bn = nn.BatchNorm2d(out_planes, device=device)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim, device=None):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, device=device)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, device=device)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, device=device)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()
        # print ("guiding_map0 first ", guiding_map0.shape)

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)
        # print ("guiding_map0", guiding_map0.shape)

        guiding_map = F.sigmoid(guiding_map0)
        # print ("guiding_map", guiding_map.shape)
        # print ("x shape: ", x.shape)

        #query = self.query_conv(x) * (1 + guiding_map)

        query = self.query_conv(x) 
        # print ("query shape: ", query.shape)
        query = query * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # print ("out size: ", out.shape)

        out = self.gamma * out + x
        # print ("out2 size: ", out.shape)
        # print (self.gamma)
        # print ("out min: ", out.min(), "out max: ", out.max())
        # print ("out mean:", out.mean(), "out std: ", out.std())

        return out
    

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)

    # x = torch.rand([1,3,256, 256])
    # model = BasicConv2d(in_planes=3, out_planes=128, kernel_size=3)
    # y = model(x)
    # print (y.shape)   #torch.Size([1, 3, 256, 256])

    # import torchinfo 

    x1 = torch.rand([1, 8, 16, 16])
    x2 = torch.rand([1, 1, 128, 128])
    # print ("x1 size: " , x1.size()[2:])
    print ("x1 min: ", x1.min(), "x1 max: ", x1.max())
    print ("x mean:", x1.mean(), "x std: ", x1.std())
    model = ChangeGuideModule(in_dim=8)
    y = model(x1, x2)
    print (y.shape)   #[1, 8,256, 256]

    # x = torch.rand([1, 8,256, 256])
    # y = F.interpolate(x, (32,32), mode='bilinear', align_corners=True)
    # print (x.shape, y.shape)
