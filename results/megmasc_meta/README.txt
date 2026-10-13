# MEG-MASC: a high-quality magneto-encephalography dataset for evaluating natural speech processing.

Laura Gwilliams, Graham Flick, Alec Marantz, Liina Pylkkänen, David Poeppel, Jean-Rémi King

- [Paper](https://arxiv.org)
- [Data](https://osf.io/rguwj/)
- [Code](https://github.com/kingjr/meg-masc)

## Abstract
The "MEG-MASC" dataset provides a curated set of raw magnetoencephalography (MEG) recordings of 27 English speakers who listened to two hours of naturalistic stories. Each participant performed two identical sessions, involving listening to four fictional stories from the Manually Annotated Sub-Corpus (MASC) intermixed with random word lists and comprehension questions. We time-stamp the onset and offset of each word and phoneme in the metadata of the recording, and organize the dataset according to the 'Brain Imaging Data Structure' (BIDS). This data collection provides a suitable benchmark to large-scale encoding and decoding analyses of temporally-resolved brain responses to speech. We provide the Python code to replicate several validations analyses of the MEG evoked related fields such as the temporal decoding of phonetic features and word frequency. All code and MEG, audio and text data are publicly available to keep with best practices in transparent and reproducible research.

## Please cite
@article{gwilliams2022neural,
  title={Neural dynamics of phoneme sequences reveal position-invariant code for content and order},
  author={Gwilliams, Laura and King, Jean-Remi and Marantz, Alec and Poeppel, David},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={1--14},
  year={2022},
  publisher={Nature Publishing Group}
}

## Task organisation
Each subject listened to four unique stories:
  - task-0 : 'lw1',
  - task-1 : 'cable_spool_fort',
  - task-2 : 'easy_money',
  - task-3 : 'The_Black_Widow'

Stories were presented in a different order to each participant:

  participant_id	:	task_order
  sub-01	:	[0, 1, 2, 3]
  sub-02	:	[0, 1, 3, 2]
  sub-03	:	[0, 2, 3, 1]
  sub-04	:	[3, 0, 1, 2]
  sub-05	:	[2, 3, 1, 0]
  sub-06	:	[0, 2, 1, 3]
  sub-07	:	[0, 3, 1, 2]
  sub-08	:	[3, 1, 0, 2]
  sub-09	:	[2, 1, 3, 0]
  sub-10	:	[1, 2, 3, 0]
  sub-11	:	[1, 3, 2, 0]
  sub-12	:	[2, 0, 3, 1]
  sub-13	:	[1, 3, 0, 2]
  sub-14	:	[1, 0, 3, 2]
  sub-15	:	[2, 1, 0, 3]
  sub-16	:	[3, 0, 2, 1]
  sub-17	:	[1, 2, 3, 0]
  sub-18	:	[2, 0, 1, 3]
  sub-19	:	[0, 3, 2, 1]
  sub-20	:	[2, 3, 0, 1]
  sub-21	:	[1, 2, 3, 0]
  sub-22	:	[1, 0, 2, 3]
  sub-23	:	[0, 2, 3, 1]
  sub-24	:	[3, 1, 2, 0]
  sub-25	:	[0, 1, 3, 2]
  sub-26	:	[3, 1, 0, 2]
  sub-27	:	[1, 2, 3, 0]


## Stimulus timestamps

The timing of each phoneme and each word is provided in each sub-*_ses-*_task-*_events.tsv file, for each subject, session and task. The timing links the MEG recording to the relevant speech moments of that story.

Each events file contains five columns:
  - onset (float) : onset time of event in seconds
  - duration (float) : duration of event in seconds
  - trial_type (dict) : dictionary of key:value pairs providing information about the event
  - sample (int) : onset time of event in MEG samples

## Stories.

Each participant listened to four fictional stories, over the course of two ~1h-long MEG sessions, with the exception of 5 subjects who only underwent 1 session. The stories were played in different orders across participants. These stories were originally selected because they had been annotated for their syntactic structures (MASC).  The corresponding text files can be found in stimuli/text/*.txt

## Word lists and pseudo-words.

To potentially investigate MEG responses to words independently of their narrative context, the text of these stories have been supplemented with word lists. Specifically, a random word list consisting of the unique content words (nouns, proper nouns, verbs, adverbs and adjectives) selected from the preceding text segment was added in a random order. In addition, a small fraction (<1%) of non-words were inserted into the natural sentences of the stories. The corresponding text files can be found in stimuli/text_with_wordlist/*.txt. For simplicity, the brain responses to these word lists and to these pseudo words are fully discarded from the present study.

## Audio synthesis.

Each of these stories was synthesized with Mac OS Mojave © version 10.14 text-to-speech. Voices (n=3 female) and speech rates (145 - 205 words per minute) varied every 5-20 sentences. The inter-sentence interval randomly varied between 0 and 1,000 ms. Both speech rate and inter-sentence intervals were sampled from a uniform distribution. Each `text_with_wordlist` files was divided into ~3 min sound files, which can be found in stimuli/audio/*.wav.

## Forced Alignment.

The timing of words and phonemes were inferred from the forced-alignment between the wav and text files, using the ‘gentle aligner’ from the Python module lowerquality (https://github.com/lowerquality/gentle). We discarded the words that did not get a forced alignment through this procedure. Analysis of the Mel spectrogram and of the phonetic decoding led to better results when using gentle than when using the Penn Forced Aligner originally used in Gwilliams et al MASC. The timing of each word and phoneme can be found in the events.tsv of each individual recording session.

## Verification. To verify that the forced alignment did not have a systematic bias, we systematically check the MEG decoding of phonetic features for each sound file separately.


﻿References
----------
Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896

Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scientific Data, 5, 180110. https://doi.org/10.1038/sdata.2018.110
