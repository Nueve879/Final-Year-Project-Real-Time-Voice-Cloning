import yaml
import paddle
from paddlespeech.t2s.modules.normalizer import ZScore
from yacs.config import CfgNode
from paddlespeech.t2s.frontend import English
from synthesizer.fastspeech2.fastspeech2 import FastSpeech2, FastSpeech2Inference
import numpy as np


class Synthesizer:
    def __init__(self, ckpt_fpath, conf_fpath, stats_fpath, phone_dict_fpath=None):
        self._synthesizer = None
        self._fastspeech2_normalizer = None
        self._frontend = None
        self._ckpt_fpath = ckpt_fpath
        self._conf_fpath = conf_fpath
        self._phone_dict_fpath = phone_dict_fpath
        self._stats_fpath = stats_fpath

    def load_model(self, speaker_dict=None, verbose=True):

        if verbose:
            print('Building FastSpeech2')

        # speaker dict
        if speaker_dict is not None:
            with open(speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)

        # phoneme dict
        if self._phone_dict_fpath is not None:
            with open(self._phone_dict_fpath, 'rt') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)

        # config
        with open(self._conf_fpath) as f:
            fastspeech2_cfg = CfgNode(yaml.safe_load(f))

        odim = fastspeech2_cfg.n_mels

        # model
        self._synthesizer = FastSpeech2(idim=vocab_size, odim=odim, **fastspeech2_cfg["model"])

        # load_model
        self._synthesizer.set_state_dict(paddle.load(self._ckpt_fpath)["main_params"])
        self._synthesizer.eval()

        # load stats
        mu, std = np.load(self._stats_fpath)
        mu = paddle.to_tensor(mu)
        std = paddle.to_tensor(std)
        self._fastspeech2_normalizer = ZScore(mu, std)

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._synthesizer is not None

    def synthesize_spectrograms(self, texts, embeds):
        # load model
        if not self.is_loaded():
            self.load_model()

        # Preprocess text
        self._frontend = English(self._phone_dict_fpath)
        input_ids = self._frontend.get_input_ids(texts, merge_sentences=True)
        phone_ids = input_ids["phone_ids"][0]

        # embeds
        embeds_paddle = paddle.to_tensor(embeds)

        # inference
        fastspeech2_inference = FastSpeech2Inference(self._fastspeech2_normalizer, self._synthesizer)
        fastspeech2_inference.eval()
        with paddle.no_grad():
            mel = fastspeech2_inference(phone_ids, spk_emb=embeds_paddle)
        return mel
