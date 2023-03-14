import json
from vocoder.env import AttrDict
from vocoder.models.hifigan import Generator
from vocoder.utils import load_checkpoint
import torch


# load HiFi-GAN
def load_model(weights_fpath, config_fpath, verbose=True):
    global _model, _device, output_sampling_rate

    if verbose:
        print('Building Hifi-GAN')

    # load config json and convert to attribute dict that can be passed to hifi-gan directly
    with open(config_fpath) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    output_sampling_rate = h.sampling_rate
    print(output_sampling_rate)

    # set seed and device
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    _model = Generator(h).to(_device)
    _model.load(weights_fpath)


def is_loaded():
    return _model is not None


def infer_waveform(mel, progress_callback=None):
    if _model is None:
        raise Exception("Please load hifi-gan in memory before using it")

    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        y_g_hat = _model(mel)
        audio = y_g_hat.squeeze()
    audio = audio * 32768.0
    audio = audio.cpu().numpy()

    return audio, output_sampling_rate
