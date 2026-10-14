import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralEnhancementModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.freq_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_weights = self.fc(x)
        fft_feat = torch.fft.rfft2(x, norm="ortho")
        fft_mag = torch.abs(fft_feat)
        fft_phase = torch.angle(fft_feat)
        fft_mag_filtered = fft_mag * torch.sigmoid(self.freq_conv(fft_mag))
        real_reconstruct = fft_mag_filtered * torch.cos(fft_phase)
        imag_reconstruct = fft_mag_filtered * torch.sin(fft_phase)
        reconstructed = torch.fft.irfft2(
            torch.complex(real_reconstruct, imag_reconstruct), 
            s=(x.shape[-2], x.shape[-1]), 
            norm="ortho"
        )
        return (x * channel_weights) + 0.1 * reconstructed

class TemporalEnhancementModule(nn.Module):
    def __init__(self, channels: int, n_div: int = 8):
        super().__init__()
        self.channels = channels
        self.fold = channels // n_div
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, batch_temporal_size: int = 3) -> torch.Tensor:
        bt, c, h, w = x.shape
        if bt % batch_temporal_size != 0:
            return self.conv(x)
        b = bt // batch_temporal_size
        t = batch_temporal_size
        x_reshaped = x.view(b, t, c, h, w)
        out = torch.zeros_like(x_reshaped)
        out[:, :-1, :self.fold] = x_reshaped[:, 1:, :self.fold]
        out[:, 1:, self.fold:2*self.fold] = x_reshaped[:, :-1, self.fold:2*self.fold]
        out[:, :, 2*self.fold:] = x_reshaped[:, :, 2*self.fold:]
        out = out.view(bt, c, h, w)
        return self.conv(out)

class C3_TS(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, stride=1)
        self.cv3 = nn.Conv2d(2 * c_, c2, kernel_size=1)
        self.m = nn.Sequential(*(nn.Sequential(
            nn.Conv2d(c_, c_, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        ) for _ in range(n)))
        self.sem = SpectralEnhancementModule(c2)
        self.tem = TemporalEnhancementModule(c2)

    def forward(self, x):
        feat = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)
        out = self.cv3(feat)
        out = self.sem(out)
        out = self.tem(out)
        return out
