import torch
import torch.nn.functional as F
import onnxruntime
    
def rerelu(x):
    return 4 * torch.relu(x - 3.25) - 4 * torch.relu(x - 3.75) - 4 * torch.relu(x - 5.25) + 4 * torch.relu(x - 5.75) + 4 * torch.relu(x - 6.25) - 4 * torch.relu(x - 6.75) - 2 * torch.relu(x - 11.25) + 2 * torch.relu(x - 11.75) + 2 * torch.relu(x - 15.25) - 2 * torch.relu(x - 15.75) - 2 * torch.relu(x - 20.25) + 2 * torch.relu(x - 20.75) + 2 * torch.relu(x - 23.25) - 2 * torch.relu(x - 23.75) - 2 * torch.relu(x - 35.25) + 2 * torch.relu(x - 35.75) + 2 * torch.relu(x - 40.25) - 2 * torch.relu(x - 40.75) - 2 * torch.relu(x - 43.25) + 2 * torch.relu(x - 43.75) + 2 * torch.relu(x - 49.25) - 2 * torch.relu(x - 49.75) - 2 * torch.relu(x - 61.25) + 2 * torch.relu(x - 61.75)

def rerelu2(x):
    return 1 - 2 * torch.relu(x-1.25)+ 2 * torch.relu(x-1.75) + 2 * torch.relu(x-2.25) - 2 * torch.relu(x-2.75) - 2 * torch.relu(x-3.25) + 2 * torch.relu(x-3.75) 

def rerelu3(x):
    return 2 * torch.relu(x-1.25)- 2 * torch.relu(x-1.75)

def rerelu4(x):
    return 8 * torch.relu(x) - 8 * torch.relu(x-1) - 0.3 * torch.relu(x-1) + 0.3 * torch.relu(x-21)

class AC(torch.nn.Module):
    def __init__(self):
        super(AC, self).__init__()
    def forward(self, duration, token ,f0):
        token = rerelu(token) # (1, token_num)

        padded_token = F.pad(token, [-1, 1], mode='constant', value=0)
        sum_token = token + padded_token

        dur_cumsum = torch.cumsum(duration, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0) # 记录每个token的起始位置

        padded_dur = F.pad(duration, [-1, 1], mode='constant', value=0)

        new_dur = rerelu2(sum_token) * padded_dur + duration

        new_dur_cumsum = dur_cumsum_prev + new_dur # 记录每个token的结束位置

        new_token = rerelu3(token)

        pos_idx = torch.arange(duration.sum(-1).max())[None, None].to(duration.device) # [1, 1, T_speech]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < new_dur_cumsum[:, :, None])

        mel_temp = new_token.T * token_mask

        mel = rerelu4((torch.cumsum(mel_temp, 2)* mel_temp).sum(1))

        # 把mel复制128遍凑成128帧，[1,t]->[1,t,128]
        mel = mel.unsqueeze(2).repeat(1,1,128)

        return mel




if __name__ == '__main__':
    # test
    ac = AC()
    duration = torch.tensor([[2,5,5,2,2,13,5, 5]])  # (1, token_num)
    token = torch.tensor([[6,66, 16, 48, 34, 36, 30, 16]])  # (1, token_num)
    f0_freams = torch.sum(duration, 1).max() 
    print(f0_freams)
    f0 = torch.rand(1, f0_freams)  # (1, fream_num)
    print(f0.shape)
    # 导出onnx：

    input_names = ['durations', 'tokens', 'f0']
    output_names = ['mel']
    # 设置动态轴：
    dynamic_axes = {'durations': { 1: 'token_num'}, 'tokens': { 1: 'token_num'}, 'f0': { 1: 'f0_freams'},'mel': {1: 'f0_freams'}}
    
    torch.onnx.export(ac, (duration, token, f0), 'ac.onnx', input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=11)



