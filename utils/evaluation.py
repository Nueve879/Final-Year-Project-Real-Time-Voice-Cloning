from encoder.audio import preprocess_wav
from encoder import inference as encoder
from evaluate_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Initialize encoder
encoder_path = Path('../saved_models/default/encoder/encoder.pt')
encoder.load_model(encoder_path)


# wav1_fpath = Path('../LibriSpeech/librispeech_test-other/1688/1688-142285-0001.flac')
# wav2_fpath = Path('../LibriSpeech/librispeech_test-other/1688/1688-142285-0004.flac')

# Similarity
def similarity_score_once(wav1_fpath, wav2_fpath):
    # preprocess wav
    wav1 = preprocess_wav(wav1_fpath)
    wav2 = preprocess_wav(wav2_fpath)

    # embedding
    embeds_1 = np.array(encoder.embed_utterance(wav1))
    embeds_2 = np.array(encoder.embed_utterance(wav2))
    print("Shape of embeddings: %s" % str(embeds_1.shape))

    # calculate similarity score
    similarity = np.inner(embeds_1, embeds_2)
    return similarity


wav_fpaths = list(Path("../LibriSpeech", "librispeech_test-other").glob("**/*.flac"))
def similarity_score_group(wav_fpaths):
    # We'll use a smaller version of the dataset LibriSpeech test-other to run our examples. This
    # smaller dataset contains 10 speakers with 10 utterances each. N.B. "wav" in variable names stands

    # Group the wavs per speaker and load them using the preprocessing function. It normalizes the volume, trims long silences and resamples
    # the wav to the correct sampling rate.
    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                    groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"),
                            lambda wav_fpath: wav_fpath.parent.stem)}

    ## Similarity between two utterances from each speaker
    # Embed two utterances A and B for each speaker
    embeds_a = np.array([encoder.embed_utterance(wavs[0]) for wavs in speaker_wavs.values()])
    embeds_b = np.array([encoder.embed_utterance(wavs[1]) for wavs in speaker_wavs.values()])

    # Each array is of shape (num_speakers, embed_size) which should be (10, 256) if you haven't
    # changed anything.
    print("Shape of embeddings: %s" % str(embeds_a.shape))

    # Compute the similarity matrix. The similarity of two embeddings is simply their dot
    # product, because the similarity metric is the cosine similarity and the embeddings are
    # already L2-normed.

    utt_sim_matrix = np.inner(embeds_a, embeds_b)

    ## Similarity between two speaker embeddings
    # Divide the utterances of each speaker in groups of identical size and embed each group as a
    # speaker embedding
    spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) \
                             for wavs in speaker_wavs.values()])
    spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) \
                             for wavs in speaker_wavs.values()])
    spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)

    ## Draw the plots
    fix, axs = plt.subplots(2, 2, figsize=(8, 10))
    labels_a = ["%s-A" % i for i in speaker_wavs.keys()]
    labels_b = ["%s-B" % i for i in speaker_wavs.keys()]
    mask = np.eye(len(utt_sim_matrix), dtype=np.bool)
    plot_similarity_matrix(utt_sim_matrix, labels_a, labels_b, axs[0, 0],
                           "Cross-similarity between utterances\n(speaker_id-utterance_group)")
    plot_histograms((utt_sim_matrix[mask], utt_sim_matrix[np.logical_not(mask)]), axs[0, 1],
                    ["Same speaker", "Different speakers"],
                    "Normalized histogram of similarity\nvalues between utterances")
    plot_similarity_matrix(spk_sim_matrix, labels_a, labels_b, axs[1, 0],
                           "Cross-similarity between speakers\n(speaker_id-utterances_group)")
    plot_histograms((spk_sim_matrix[mask], spk_sim_matrix[np.logical_not(mask)]), axs[1, 1],
                    ["Same speaker", "Different speakers"],
                    "Normalized histogram of similarity\nvalues between speakers")
    plt.show()


# The neural network will automatically use CUDA if it'speaker available on your machine, otherwise it
# will use the CPU. You can enforce a device of your choice by passing its name as argument to the
# constructor. The model might take a few seconds to load with CUDA, but it then executes very
# quickly.














#
#
