from .modules import AttentionBlock, TripleResBlock, conv, conv1x1, deconv, nn, torch


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
            conv(2 * F + N, N),
            TripleResBlock(N),
        )

        self.block3 = nn.Sequential(
            conv(4 * F + N, M),
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
                    nn.Conv2d(F, F, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(F * 3, F * 2, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            deconv(M, N), TripleResBlock(N), conv1x1(N, F * 4)
        )

        self.p4Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F * 2),
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
        self.decoder_attention = AttentionBlock(M)

        self.fmb34 = FeatureMixingBlock(F)
        self.fmb45 = FeatureMixingBlock(F * 2)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p3 = self.p3Decoder(y_hat)
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p3, p4, p5]
