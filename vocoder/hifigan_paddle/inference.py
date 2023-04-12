import numpy as np
from paddlespeech.t2s.modules.normalizer import ZScore

from vocoder.wavernn import hparams as hp
from vocoder.hifigan_paddle.hifigan import HiFiGANGenerator, HiFiGANInference
import paddle
import yaml
from yacs.config import CfgNode


def load_model(ckpt_fpath, conf_fpath, stats_fpath, verbose=True):
    global _vocoder, _hifigan_normalizer, output_sampling_rate
    if verbose:
        print('Building Hifi-GAN')

    with open(conf_fpath) as f:
        hifigan_config = CfgNode(yaml.safe_load(f))
    _vocoder = HiFiGANGenerator(**hifigan_config["generator_params"])
    _vocoder.load(ckpt_fpath)

    output_sampling_rate = hifigan_config["fs"]
    _vocoder.eval()
    # Load stats file:
    stat = np.load(stats_fpath)
    mu, std = stat
    # mu = paddle.to_tensor(mu)
    mu = paddle.to_tensor(mu.reshape(mu.shape[0], -1))
    std = paddle.to_tensor(std.reshape(std.shape[0], -1))
    # std = paddle.to_tensor(std)
    _hifigan_normalizer = ZScore(mu, std)


def infer_waveform(mel, progress_callback=None):
    if _vocoder is None:
        raise Exception("Please load hifi-gan in memory before using it")

    hifigan_inference = HiFiGANInference(_hifigan_normalizer, _vocoder)
    hifigan_inference.eval()
    mel = paddle.to_tensor(mel)
    wav = hifigan_inference(mel).flatten()
    # wav = wav / np.abs(wav).max() * 0.3
    wav = wav.cpu().numpy()

    return wav, output_sampling_rate
