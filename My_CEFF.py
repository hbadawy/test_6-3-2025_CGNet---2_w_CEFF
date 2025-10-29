
import torch
import torch.nn as nn

class CEFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=4, bias=False, device=None):
        super(CEFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias, device=device), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias, device=device))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        # print ("inp feats:  ", len(inp_feats))    #2
        x1 = inp_feats[0]   #[1, 32, 256, 256]
        x2 = inp_feats[1]

        inp_feats = torch.cat(inp_feats, dim=1) #[1, 64, 256, 256]  ; print ("inp_feats: ", inp_feats.shape)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])#[1, 2, 32, 256, 256] 

        x1_av = self.avg_pool(x1)
        x2_av = self.avg_pool(x2)
        # print ("x1_av: ", x1_av.shape)  #[1, 32, 1, 1]
        x_sum = x1_av + x2_av
        # print ("x_sum: ", x_sum.shape)
        feats_Z = self.conv_du(x_sum)
        # print ("feats_Z: ", feats_Z.shape)   #[1, 4, 1, 1]

        # inp_feats = torch.cat(inp_feats, dim=1)
        # inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        
        # feats_U = torch.sum(inp_feats, dim=1)
        # feats_S = self.avg_pool(feats_U)
        # feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        # print ("attn vector: ", len(attention_vectors))
        # print ("attn vector[0]: ", attention_vectors[0].shape)   #[1, 32, 1, 1]
        attention_vectors = torch.cat(attention_vectors, dim=1)   #[1, 64, 1, 1]
        # print ("attn vectors after cat: ", attention_vectors.shape)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)  #[1, 2, 32, 1, 1]
        # print ("attn vectors after view: ", attention_vectors.shape)

        
        attention_vectors = self.softmax(attention_vectors)   #[1, 2, 32, 1, 1]
        # print ("attn vectors after softmax: ", attention_vectors.shape)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        # print ("max value: ", feats_V.max(), "min value: ", feats_V.min())
        
        return feats_V     
    

if __name__ == "__main__":
    x1 = torch.rand([1,128,128,128])   #[1,32,256,256]
    x2 = torch.rand([1,128,128,128])   #[1,32,256,256]
    x =[x1, x2]
    model = CEFF(in_channels=128, height=2)
    y = model(x)
    print (y.shape)
