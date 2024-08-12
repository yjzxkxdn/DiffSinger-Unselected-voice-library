import torch
import torch.nn.functional as F
import torch.nn as nn
import onnxruntime

def upsample(signal, factor):
    '''
        signal: B x C X T
        factor: int
        return: B x C X T*factor
    '''
    signal = nn.functional.interpolate(
        torch.cat(
            (signal,signal[:,:,-1:]),
            2
        ), 
        size=signal.shape[-1] * factor + 1, 
        mode='linear', 
        align_corners=True
    )
    signal = signal[:,:,:-1]
    return signal

class vocoder(nn.Module):

    def forward(self, mel):

        # mel: B x T x C
        # upsample to 512x

        # 把B x T x C 变成 B x T x 1
        mel = mel[:, :,0:1]  # B x T x 1

        mel = mel.transpose(1, 2)  # B x 1 x T
        
        mel = upsample(mel, 512)  # B x 1 x 512T
        mel = mel[0,0,:].unsqueeze(0)  # 1 x 512T

        return mel
    
if __name__ == '__main__':
    # 导出onnx
    model = vocoder()

    
    # 导出onnx：
    # 输入：mel
    # 输出：mel2
    mel = torch.randn(1, 100, 128)

    input_names = ['mel']
    output_names = ['mel2']
    # 设置动态轴：
    dynamic_axes = { 'mel': {1: 'n_frames'},'mel2': {1: 'n_samples'} }
    
    torch.onnx.export(model, (mel), 'mel2.onnx', input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=15)
