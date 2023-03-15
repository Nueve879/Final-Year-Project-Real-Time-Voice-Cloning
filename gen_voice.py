import vocoder.hifigan.inference as hifigan
import vocoder.wavernn.inference as wavernn
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from pathlib import Path
import numpy as np
import torch
import os
import sys
import soundfile as sf

def gen_wav_once(synthesizer, in_fpath, texts, embed, filename):
    texts = texts.split('\n')
    embeds = [embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    # spec = specs[0]
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    # If seed is specified, reset torch seed and reload vocoder
    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav, output_sample_rate = vocoder.infer_waveform(spec)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [generated_wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * synthesizer.sample_rate))] * len(breaks)
    generated_wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    ## Post-generation
    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)
    generated_wav = generated_wav / np.abs(generated_wav).max() * 0.97

    # Save it on the disk
    model = os.path.basename(in_fpath)
    filename = f"{filename}_{vocoder.__name__.__str__().split('.')[1]}_{model}"
    sf.write(filename, generated_wav, synthesizer.sample_rate)

    print("\nSaved output as %s\n\n" % filename)


def gen_voice(encoder_path, synthesizer_path, vocoder_path, in_fpath, input_texts, filename, config_fpath=None):
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print environment information (for debugging purposes)
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
    encoder.load_model(encoder_path)
    synthesizer = Synthesizer(synthesizer_path)
    vocoder.load_model(vocoder_path, config_fpath)

    # compute embed
    encoder_wav = synthesizer.load_preprocess_wav(in_fpath)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # synthesize and vocode
    gen_wav_once(synthesizer, in_fpath, input_texts, embed, filename)


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        my_txt = ""
        print("reading from :", sys.argv[1])
        with open(sys.argv[1], "r") as f:
            for line in f.readlines():
                my_txt += line
        txt_file_name = sys.argv[1]
        wav_file_name = sys.argv[2]
        print("Input text is: \n", my_txt)

        if sys.argv[3] == 'wavernn':
            vocoder = wavernn
            gen_voice(
                Path("saved_models/default/encoder/encoder.pt"),
                Path("saved_models/default/synthesizer/synthesizer.pt"),
                Path("saved_models/default/vocoder/wavernn.pt"),
                wav_file_name, my_txt, txt_file_name
            )
        elif sys.argv[3] == 'hifigan':
            vocoder = hifigan
            gen_voice(
                Path("saved_models/default/encoder/encoder.pt"),
                Path("saved_models/default/synthesizer/synthesizer.pt"),
                Path("saved_models/default/vocoder/hifigan.pt"),
                wav_file_name, my_txt, txt_file_name,
                Path("saved_models/default/vocoder/config16k.json")
            )
        else:
            print("Currently only 'wavernn' and 'hifigan' are supported")

    else:
        print("please input the file name and select a vocoder ('hifigan' / 'wavernn')")
        exit(1)

