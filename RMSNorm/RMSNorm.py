import argparse
import torch
import torch.nn as nn
import Config.Modelconfig as Modelconfig

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    args = parser.parse_args()

    norm = RMSNorm(args.dim, args.norm_eps)
    x = torch.randn(1, 50, args.dim)
    output = norm(x)
    print(output.shape)