from modules import AttentionBlock, TripleResBlock, conv, conv1x1, deconv, nn, torch
import math
import torch.nn.functional as F
class FENet_DRNet_Base(nn.Module):
    def __init__(self, task=None):
        super().__init__()
        self.task = task
    
    def get_inverse_gain(self, q):
        q = q - 1   # -0.1

        lower_index = int(math.floor(q))
        upper_index = int(math.ceil(q))
        decimal = q - lower_index
        if lower_index < 0:
            y_quant_inv = torch.abs(self.InverseGain[upper_index]) * (1 / decimal)
        else:
            y_quant_inv = torch.abs(self.InverseGain[lower_index]).pow(
                1 - decimal
            ) * torch.abs(self.InverseGain[upper_index]).pow(decimal)
        y_quant_inv = y_quant_inv.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return y_quant_inv

    def get_gain(self, q):
        # if q = 0.9
        q = q - 1   # -0.1
        lower_index = int(math.floor(q)) # -1
        upper_index = int(math.ceil(q)) # 0
        decimal = q - lower_index # 0.9
        
        if lower_index < 0:
            y_quant = torch.abs(self.Gain[upper_index]) * decimal
        else:
            y_quant = torch.abs(self.Gain[lower_index]).pow(1 - decimal) * torch.abs(
                self.Gain[upper_index]
            ).pow(decimal)
        y_quant = y_quant.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return y_quant
    
        
    def get_config_from_task_id(self, task, type):
        assert type in ['fenet', 'drnet']
        config_dict = {
            'obj': {
                'net': FENet_FPN if type == 'fenet' else DRNet_FPN,
                'F': 256,
                'levels': 8,
                },
            'seg': {
                'net': FENet_FPN if type == 'fenet' else DRNet_FPN,
                'F': 256,
                'levels': 8,
                },
            'alt1': {
                'net': FENet_DKN if type == 'fenet' else DRNet_DKN,
                'F': 128,
                'levels': 6,
                },
            'dn53': {
                'net': FENet_DKN if type == 'fenet' else DRNet_DKN,
                'F': 256,
                'levels': 6,
                },
        }
        return config_dict[task]
    
        
    def cal_feature_padding_size(self, shape):
        if self.task in ["obj", "seg"]:
            ps_list = [64, 32, 16, 8]
        elif self.task in ["alt1", "dn53"]:
            ps_list = [32, 16, 8]
        else:
            raise NotImplementedError
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

    def feature_padding(self, features, pad_info):
        paddings = pad_info["paddings"]
        padded_features = [
            F.pad(f, paddings[i], mode="reflect") for i, f in enumerate(features)
        ]
        return padded_features

    def feature_unpadding(self, features, pad_info):
        unpaddings = pad_info["unpaddings"]
        unpadded_features = [F.pad(f, unpaddings[i]) for i, f in enumerate(features)]
        return unpadded_features

class FENet_FPN(nn.Module):
    def __init__(self, F, N, M) -> None:
        super().__init__()
        self.block1 = nn.Sequential(conv(F, N), TripleResBlock(N))

        self.block2 = nn.Sequential(
            conv(F + N, N),
            TripleResBlock(N),
            AttentionBlock(N),
        )

        self.block3 = nn.Sequential(
            conv(F + N, N),
            TripleResBlock(N),
        )

        self.block4 = nn.Sequential(
            conv(F + N, M),
            AttentionBlock(M),
        )

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        p2, p3, p4, p5 = tuple(p_layer_features)
        y = self.block1(p2)
        y = self.block2(torch.cat([y, p3], dim=1))
        y = self.block3(torch.cat([y, p4], dim=1))
        y = self.block4(torch.cat([y, p5], dim=1))
        return y


class DRNet_FPN(nn.Module):
    def __init__(self, F=256, N=192, M=320) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 2, N, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(deconv(M, N), TripleResBlock(N), conv1x1(N, F))

        self.p4Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )

        self.p3Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )
        self.p2Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )

        self.decoder_attention = AttentionBlock(M)

        self.fmb23 = FeatureMixingBlock(F)
        self.fmb34 = FeatureMixingBlock(F)
        self.fmb45 = FeatureMixingBlock(F)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p2 = self.p2Decoder(y_hat)
        p3 = self.fmb23(p2, self.p3Decoder(y_hat))
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p2, p3, p4, p5]


class FENet_DKN(nn.Module):
    def __init__(self, F, N, M) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            conv(F, N),
            TripleResBlock(N),
            AttentionBlock(N),
        )

        self.block2 = nn.Sequential(
            conv(2*F+N, N),
            TripleResBlock(N),
        )

        self.block3 = nn.Sequential(
            conv(4*F+N, M),
            AttentionBlock(M),
        )
    
    def forward(self, p_layer_features):
        p3, p4, p5 = tuple(p_layer_features)
        y = self.block1(p3)
        y = self.block2(torch.cat([y, p4], dim=1))
        y = self.block3(torch.cat([y, p5], dim=1))
        return y

class DRNet_DKN(nn.Module):
    def __init__(self, F=256, N=192, M=320) -> None:
        super().__init__()
        class FeatureMixingBlock(nn.Module):
            def __init__(self, F) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(F, F, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(F * 3, F * 2, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU()
                )
            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            conv1x1(N, F*4)
        )

        self.p4Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F*2)
        )

        self.p3Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F)
        )
        self.decoder_attention = AttentionBlock(M)

        self.fmb34 = FeatureMixingBlock(F)
        self.fmb45 = FeatureMixingBlock(F * 2)
        
    
    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p3 = self.p3Decoder(y_hat)
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p3, p4, p5]