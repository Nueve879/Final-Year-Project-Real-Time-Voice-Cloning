import vocoder.hifigan_paddle.inference as hifigan_paddle
import vocoder.wavernn.inference as wavernn
from encoder import inference as encoder
from synthesizer.fastspeech2.inference import Synthesizer as Fastspeech2_Syn
from synthesizer.inference import Synthesizer as Tacotron_Syn
from pathlib import Path
import numpy as np
import torch
import os
import sys
import soundfile as sf


def gen_wav_once(synthesizer, in_fpath, texts, embed, filename, syn_name):
    if syn_name == 'tacotron':
        texts = texts.split('\n')
        embeds = [embed] * len(texts)
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        # spec = specs[0]
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        # Synthesizing the waveform:the longer the spectrogram, the more time-efficient the vocoder.
        generated_wav, output_sample_rate = vocoder.infer_waveform(spec)

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * synthesizer.hparams.hop_size).astype(np.int32)   # !!! modified: .astype(np.int32)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [generated_wav[start:end] for start, end in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * output_sample_rate))] * len(breaks)
        generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        ## Post-generation
        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        # generated_wav = encoder.preprocess_wav(generated_wav)        # !!! modified encoder.preprocess_wav: normalize
        generated_wav = generated_wav / np.abs(generated_wav).max() * 0.97
    elif syn_name == 'fastspeech2':
        specs = synthesizer.synthesize_spectrograms(texts, embed)
        specs = specs.T
        generated_wav, output_sample_rate = vocoder.infer_waveform(specs)

    # Save it on the disk
    model = os.path.basename(in_fpath)
    filename = f"{filename}_{syn_name}_{vocoder.__name__.__str__().split('.')[1]}_{model}"
    sf.write(filename, generated_wav, output_sample_rate)

    print("\nSaved output as %s\n\n" % filename)


def gen_voice(
              refwav_fpath,
              input_texts,
              filename,
              syn_name=None,
              encoder_path=None,
              syn_ckpt_fpath=None,
              syn_config_fpath=None,
              syn_stats_fpath=None,
              syn_phone_dict_fpath=None,
              voc_ckpt_fpath=None,
              voc_config_fpath=None,
              voc_stats_fpath=None):

    # Print environment information:
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
              "%.1fGb total memory.\n" %
              (torch.cuda.device_count(),
               device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    # Load model
    ## encoder
    encoder.load_model(encoder_path)

    ## synthesizer
    if syn_name == 'tacotron':
        synthesizer = Tacotron_Syn(syn_ckpt_fpath)
    elif syn_name == 'fastspeech2':
        synthesizer = Fastspeech2_Syn(syn_ckpt_fpath, syn_config_fpath, syn_stats_fpath, syn_phone_dict_fpath)

    # compute embed
    # encoder_wav = synthesizer.preprocess_wav(refwav_fpath)
    encoder_wav = encoder.preprocess_wav(refwav_fpath)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    ## vocoder
    vocoder.load_model(voc_ckpt_fpath, voc_config_fpath, voc_stats_fpath)

    gen_wav_once(synthesizer, refwav_fpath, input_texts, embed, filename, syn_name=syn_name)


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        my_txt = ""
        print("reading from :", sys.argv[1])
        with open(sys.argv[1], "r") as f:
            for line in f.readlines():
                my_txt += line
        txt_file_name = sys.argv[1]
        wav_file_name = sys.argv[2]
        print("Input text is: \n", my_txt)

        syn_name = sys.argv[3]
        voc_name = sys.argv[4]


        # if sys.argv[4] == 'wavernn':
        #     vocoder = wavernn
        #     gen_voice(
        #         refwav_fpath=wav_file_name,
        #         input_texts=my_txt,
        #         filename=txt_file_name,
        #         syn_name='tacotron',
        #         encoder_path=Path("saved_models/default/encoder/encoder.pt"),
        #         syn_ckpt_fpath=Path("saved_models/default/synthesizer/synthesizer.pt"),
        #         voc_ckpt_fpath=Path("saved_models/default/vocoder/wavernn.pt")
        #     )
        if sys.argv[4] == 'wavernn':
            vocoder = wavernn
            gen_voice(
                refwav_fpath=wav_file_name,
                input_texts=my_txt,
                filename=txt_file_name,
                syn_name=syn_name,
                encoder_path=Path("saved_models/default/encoder/encoder.pt"),
                syn_ckpt_fpath="saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/snapshot_iter_66200.pdz",
                syn_config_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/default.yaml"),
                syn_stats_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/speech_stats.npy"),
                syn_phone_dict_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/phone_id_map.txt"),
                voc_ckpt_fpath=Path("saved_models/default/vocoder/wavernn.pt")
            )
        elif sys.argv[4] == 'hifigan_paddle':
            vocoder = hifigan_paddle
            gen_voice(
                refwav_fpath=wav_file_name,
                input_texts=my_txt,
                filename=txt_file_name,
                syn_name=syn_name,
                encoder_path=Path("saved_models/default/encoder/encoder.pt"),
                syn_ckpt_fpath="saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/snapshot_iter_66200.pdz",
                syn_config_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/default.yaml"),
                syn_stats_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/speech_stats.npy"),
                syn_phone_dict_fpath=Path("saved_models/default/synthesizer/fastspeech2_vctk_ckpt_1.2.0/phone_id_map.txt"),
                voc_ckpt_fpath="saved_models/default/vocoder/hifigan_vctk/snapshot_iter_2500000.pdz",
                voc_config_fpath=Path("saved_models/default/vocoder/hifigan_vctk/default.yaml"),
                voc_stats_fpath=Path("saved_models/default/vocoder/hifigan_vctk/feats_stats.npy")
                )
        else:
            print("Currently only 'wavernn', 'hifigan', 'hifigan_paddle' are supported")

    else:
        print("please input the file name and select a vocoder ('hifigan' / 'wavernn')")
        exit(1)

