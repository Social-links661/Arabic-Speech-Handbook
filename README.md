# Arabic Speech AI: Research Foundation
## A Complete Map of Arabic STT & TTS: Data, Models, Benchmarks, and Engineering

| | |
|---|---|
| **Version** | 1.0 |
| **Compiled** | August 2026 |
| **Purpose** | Source material and blueprint for a comprehensive, educational Arabic speech repository |
| **Scope** | ASR/STT, TTS, and every adjacent task (dialect ID, emotion, translation, pronunciation assessment), plus tooling, evaluation, and production engineering |
| **How to use** | Each numbered section maps to one page or folder of the future repo and closes with its primary references. Facts are current as of the compile date; re-verify licenses and leaderboard rankings before publishing |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Benchmarks and Leaderboards](#2-benchmarks-and-leaderboards)
3. [ASR and General Speech Corpora](#3-asr-and-general-speech-corpora)
4. [TTS-Grade Corpora](#4-tts-grade-corpora)
5. [Task-Specific Corpora and Adjacent Tasks](#5-task-specific-corpora-and-adjacent-tasks)
6. [Community Hubs and Legacy Catalogs](#6-community-hubs-and-legacy-catalogs)
7. [Licensing and Data Governance](#7-licensing-and-data-governance)
8. [Open ASR Models](#8-open-asr-models)
9. [Open TTS Models](#9-open-tts-models)
10. [Speech-LLMs and Omni Models](#10-speech-llms-and-omni-models)
11. [Commercial APIs (MENA-Focused)](#11-commercial-apis-mena-focused)
12. [Phonology, Orthography, and Transliteration](#12-phonology-orthography-and-transliteration)
13. [The Nine Core Challenges](#13-the-nine-core-challenges)
14. [Vocoders and Neural Audio Codecs](#14-vocoders-and-neural-audio-codecs)
15. [Text Resources for LMs and TTS Frontends](#15-text-resources-for-lms-and-tts-frontends)
16. [Tooling Landscape](#16-tooling-landscape)
17. [Evaluation Methodology](#17-evaluation-methodology)
18. [Production: Voice Agents, Streaming, and Telephony](#18-production-voice-agents-streaming-and-telephony)
19. [Data Collection, Annotation, and Legal](#19-data-collection-annotation-and-legal)
20. [Ecosystem Directory](#20-ecosystem-directory)
21. [Reading List](#21-reading-list)
22. [Learning Resources](#22-learning-resources)
23. [Gap Analysis](#23-gap-analysis)
24. [Proposed Repository Structure](#24-proposed-repository-structure)

---

## 1. Executive Summary

Five facts define the Arabic speech field in mid-2026:

1. **ASR has gone multi-dialect.** The era of MSA-only broadcast models is over. NVIDIA FastConformer Arabic and commercial engines (Munsit, ElevenLabs Scribe, Cohere Transcribe, Hamsa) top the leaderboards, Meta's Omnilingual ASR reset the multilingual baseline, and Whisper large-v3 remains the universal open baseline while degrading hard on dialects (average WER around 37% across the Open Universal Arabic ASR Leaderboard's multi-dialect test sets).

2. **TTS had its breakout in 2025-2026.** F5-TTS (flow matching) became the de-facto open architecture: SILMA TTS, Habibi-TTS (12+ dialects in one unified model), and a wave of community fine-tunes (Egyptian, Najdi, MSA). Habibi reports human-eval parity with ElevenLabs v3 (alpha) on intelligibility, speaker similarity, and naturalness.

3. **Evaluation matured on three fronts.** Static benchmarks (Open Universal Arabic ASR Leaderboard, SawtArabi, Habibi benchmark, SILMA TTS benchmark), human-preference arenas (Navid AI's Arabic TTS Arena), and task-specific shared tasks (Iqra'Eval for Quranic pronunciation, NADI for dialectal ASR).

4. **Licensing is the hidden production risk.** SADA's CC BY-NC-SA license propagates into every model trained on it, splitting Habibi's own checkpoints between Apache 2.0 and non-commercial. No existing resource documents this; it is a core differentiator for the repo.

5. **The gaps are documented and real.** Dialectal orthography, diacritization for TTS, code-switching, streaming/latency engineering, paralinguistics (emotion, non-verbal events), Arabic speaker verification, and anti-spoofing all remain open. Each gap is a chapter and a contribution magnet.

---

## 2. Benchmarks and Leaderboards

| Benchmark | Task | Maintainer | What it covers |
|---|---|---|---|
| **Open Universal Arabic ASR Leaderboard** | ASR | ELM Research (HF space + Interspeech 2025 paper) | Six multi-dialect test sets (SADA, Common Voice, MASC, MGB-2, FLEURS, and more); WER + CER under a published normalization recipe; robustness, speaker-adaptation, efficiency, and memory analyses; continuously updated (Gemma-4 audio and Cohere Transcribe added April 2026) |
| **SawtArabi** | TTS | HUMAIN + SDAIA + QCRI + KTH (Interspeech 2025) | First Arabic TTS benchmark covering MSA, Egyptian dialect, and Egyptian-English code-switching; ships a modified espeak-ng phonemizer and baseline checkpoints; quantifies the impact of vowelization |
| **Habibi Benchmark** | TTS | SWivid / Habibi team | 11,000+ utterances across 7 dialect subsets with manually verified transcripts; metrics: WER from two ASR systems (WER-S/O), D-MOS (dialect accuracy), S-MOS (speaker similarity), N-MOS (naturalness) |
| **Arabic TTS Arena** | TTS, human preference | Navid AI (navid.sa) | Elo-style arena: two hidden models synthesize the same user-supplied sentence, native speakers vote; quality emerges from human preference rather than fixed metrics; the living complement to static TTS benchmarks |
| **SILMA Arabic TTS Benchmark** | TTS | SILMA AI | Open-source auditory assessment standard |
| **Iqra'Eval** | Pronunciation assessment (MDD) | ArabicNLP 2025 shared task | First open benchmark for Mispronunciation Detection and Diagnosis in Quranic recitation (MSA reading), built on the QuranMB dataset; phoneme-level, with error-localization and diagnosis subtasks; HF space and leaderboard; top systems (BAIC, Hafs2Vec) used synthetic errors and ~94 h of EveryAyah/QUL recitations |
| **NADI shared tasks** | Dialect ID + dialectal ASR | UBC-NLP + ArabicNLP | Annual; NADI 2025 subtask 2 covers dialectal ASR (data on HF under OpenRAIL-M, non-commercial); track yearly |
| **SAFIR** | Saudi-specific tasks | Community | Saudi Arabic evaluation |
| **LAraBench** | Multi-task incl. ASR | QCRI | The standard cross-vendor ASR comparison baseline |
| **Ramsa baselines** | ASR + TTS | MBZUAI (2026) | Emirati; Whisper-family ASR baselines, ArTST and MMS-TTS-Ara TTS baselines |

**Meta-resource:** SILMA's "Arabic AI Benchmarks and Leaderboards" blog post is a maintained index of every Arabic leaderboard; link it on the repo's benchmarks page.

**References:** [ASR Leaderboard paper](https://arxiv.org/abs/2412.13788) · [Leaderboard space](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard) · [Eval code](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard) · [SawtArabi](https://www.isca-archive.org/interspeech_2025/lodagala25_interspeech.pdf) · [Habibi paper](https://arxiv.org/abs/2601.13802) · [Arabic TTS Arena](https://huggingface.co/spaces/Navid-AI/Arabic-TTS-Arena) · [Arena methodology](https://huggingface.co/blog/Navid-AI/introducing-arabic-tts-arena) · [Iqra'Eval findings](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/) · [Iqra'Eval space](https://huggingface.co/spaces/IqraEval/SharedTask_ArabicNLP2025) · [NADI 2025 ASR data](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask2_ASR) · [SILMA benchmarks index](https://huggingface.co/blog/silma-ai/arabic-ai-benchmarks-and-leaderboards) · [Ramsa](https://arxiv.org/abs/2603.08125)

---

## 3. ASR and General Speech Corpora

| Dataset | Size | Dialects / Content | Access / License | Notes |
|---|---|---|---|---|
| **QASR** | 2,000 h | Multi-dialect broadcast (Aljazeera), 16 kHz | Research, registration via arabicspeech.org | Largest transcribed corpus; lightly supervised transcripts; punctuation, speaker info; +130M-word LM text release; known download-availability issues |
| **MGB-2** | 1,200 h | MSA + Egyptian, Gulf, Levantine, North African broadcast | Research, registration | The classic large-scale benchmark; its 9.6 h test set anchors most leaderboards |
| **SADA** | 668 h (~435 h labeled) | Najdi, Hijazi, Khaleeji + Yemeni, Egyptian, Levantine, MSA | **CC BY-NC-SA**; commercial use requires a direct arrangement | Saudi Broadcasting Authority shows, prepared by SDAIA/NCAI; under one third clean, one third noisy, one third music-overlaid; 10.7 h test set with 10 dialect labels |
| **MASC** | ~1,000 h | Multi-dialect, YouTube | CC-BY-class (verify current terms) | Core training set behind the open MSA models |
| **Common Voice (ar)** | ~157 h (92 h validated, v22) | Mostly MSA read speech, 1,632 speakers | **CC0** | The safest license in the ecosystem; small but clean |
| **Casablanca** | 48 h | 8 dialects: Algerian, Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni | Subset public; source videos referenced by YouTube URL only | Fully supervised; annotations for transcript, gender, dialect, and code-switching in both Latin and Arabic-transliterated forms |
| **Aswat** | 732 h | Multi-genre, clean speech | Verify availability | Built for SSL pretraining (wav2vec/data2vec); contributed to 11.7% WER on Common Voice and 10.3% on MGB-2 |
| **MGB-3** | 16 h | Egyptian (YouTube, multi-genre) | Research | The standard Egyptian test bed |
| **MGB-5** | 14 h | Moroccan | Research | The standard Maghrebi test bed |
| **FLEURS (ar)** | ~10 h | MSA-leaning read speech | CC-BY | Common evaluation split |
| **Mixat** | ~15 h | Emirati-English code-switching | Non-commercial | MBZUAI; its license propagates into Habibi's UAE checkpoints |
| **Ramsa** (2026) | Large (see paper) | Emirati, sociolinguistically annotated | Research | Joint ASR + TTS benchmarks with published baselines |
| **ArzEn** (LREC 2020) | 12 h | Egyptian Arabic-English code-switched spontaneous interviews, 38 bilingual speakers | Research, public | Soundproof-room recordings; the reference CS-ASR corpus |
| **TARIC-SLU** | Varies | Tunisian | Research | Used in Ara-BEST-RQ evaluations |
| **TunSwitch** (ICASSP 2024) | Varies | Tunisian code-switched | Research | From the data-collection + unsupervised-learning Tunisian ASR line of work |
| **Tarteel** | 67.4 h | Quranic recitation, 1,200 crowd reciters | Open (HF) | Classical Arabic; useful for diacritic-aware ASR |
| **EveryAyah** | Large | Quranic recitation, professional reciters | Open (HF) | Behind NVIDIA's 6.65% diacritic-aware WER result and Iqra'Eval systems |
| **Arabic Little STT** (2025) | Small | Children's speech | Research | A rare children's-speech resource |
| **CV-18 NER** | 32 h train | MSA (Common Voice 18) + entity tags | CC0 | First open benchmark for named-entity recognition from Arabic speech (Elyadata) |

**References:** [QASR](https://arxiv.org/abs/2106.13000) · [QASR on HF](https://huggingface.co/datasets/QCRI/QASR) · [arabicspeech.org (MGB)](https://arabicspeech.org/) · [SADA analysis](https://arxiv.org/abs/2508.12968) · [Casablanca](https://arxiv.org/abs/2410.04527) · [Casablanca project](https://www.dlnlp.ai/speech/casablanca/) · [Aswat](https://aclanthology.org/2023.arabicnlp-1.10/) · [Common Voice](https://commonvoice.mozilla.org/en/datasets) · [FLEURS](https://huggingface.co/datasets/google/fleurs) · [ArzEn](https://sites.google.com/view/arzen-corpus/home) · [ArzEn-ST](https://arxiv.org/abs/2211.12000) · [Ramsa](https://arxiv.org/abs/2603.08125) · [Arabic Little STT](https://arxiv.org/abs/2510.23319) · [CV18-NER](https://huggingface.co/datasets/Elyadata/CV18-NER)

## 4. TTS-Grade Corpora

| Dataset | Size | Content | Access / License | Notes |
|---|---|---|---|---|
| **ASC (Arabic Speech Corpus)** | ~3.7 h | Single male, Damascene-flavored MSA, phoneme-balanced, fully diacritized with phonetic transcripts | Free for research (verify commercial) | Nawar Halabi's classic; still the base of most academic Arabic TTS |
| **ClArTTS** | ~12 h | Classical Arabic, single male (LibriVox) | Research | MBZUAI; base of the ArTST TTS checkpoint |
| **ArVoice** (2025) | 83.5 h total (~10 h human, 7 speakers) | Multi-speaker MSA, fully diacritized transcripts | Research use, on request | Built for multi-speaker TTS, voice conversion, and deepfake-detection research; includes a modified ASC plus synthetic voices |
| **SawtArabi corpus** | Benchmark-scale | MSA + Egyptian + Egyptian-English code-switching, professional speaker | Public | Ships the modified espeak-ng phonemizer and baseline checkpoints; documents dialectal orthography decisions |
| **Habibi training data** | Large (repurposed) | 12+ regional dialects curated from open ASR corpora | Mixed (inherits source licenses) | The reference recipe for converting ASR corpora into TTS-grade data via a multi-step curation pipeline |
| **Habibi benchmark** | 11,000+ utterances | 7 dialect subsets, manually verified transcripts | Open | First standardized multi-dialect Arabic TTS benchmark |
| **Lahgtna datasets** | 54.6K-211K rows per set | Multi-dialect TTS data (see section 6) | Open (HF) | lahgtna-v3-small, lahgtna-v3-small-augmented, msa-omnivoice-tts-v1 |
| **Common Voice (ar)** | 92 h validated | Crowd read speech | CC0 | Usable for TTS after aggressive quality filtering |

**References:** [ASC on HF](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus) · [ArVoice](https://arxiv.org/abs/2505.20506) · [SawtArabi](https://www.isca-archive.org/interspeech_2025/lodagala25_interspeech.pdf) · [Habibi](https://arxiv.org/abs/2601.13802) · [lahgtna-v3-small](https://huggingface.co/datasets/oddadmix/lahgtna-v3-small)

## 5. Task-Specific Corpora and Adjacent Tasks

### 5.1 Dialect Identification (ADI)

| Resource | Size | Coverage | Notes |
|---|---|---|---|
| **ADI-5** | 74 h | 5 dialect groups (Aljazeera) | MGB-3-era shared task |
| **ADI-17** | 3,033 h train + ~57 h dev/test | 17 countries (YouTube) | MGB-5 shared task; heavily imbalanced (Jordan 25.9 h vs Iraq 815.8 h); duration-based test splits (<5s, 5-20s, >20s) |
| **ADI-20** (2025) | 3,556 h | 19 dialects + MSA, all Arab countries | Open data and open models (ECAPA-TDNN and Whisper-encoder classifiers); near-peak F1 with only ~30% of the data; zero-shot evaluation on Casablanca |
| **CTC-DID** (2026) | Model line | Streaming DID | Low-latency dialect ID for production routing |

Production pattern: a dialect-ID router in front of dialect-specialized ASR/TTS (the VoxArabica architecture).

### 5.2 Speech Emotion Recognition (SER)

| Dataset | Size | Dialect / Design | Access |
|---|---|---|---|
| **KSUEmotions** (LDC2017S12) | ~5 h, 23 speakers (Saudi, Yemen, Syria) | MSA, acted, 5 emotions | Paid (LDC) |
| **EYASE** | ~461-579 utterances | Egyptian, semi-natural (award-winning TV series), 4 emotions | Free |
| **BAVED** | 1,935 utterances, 61 speakers | 7 words x 3 emotional-intensity levels, 16 kHz | Free (GitHub) |
| **KEDAS** (2022) | 5,000 recordings | MSA, 5 emotions | Research |
| **SERTUS** | Spontaneous | Tunisian | Research |

Baselines: wav2vec2/HuBERT fine-tunes reach ~87-89% on BAVED; recent CNN-Transformer hybrids report ~98% (2026). SER connects directly to expressive TTS (emotion conditioning and non-verbal events).

### 5.3 Speech Translation

- **ArzEn-ST** (WANLP 2022, arXiv 2211.12000): the three-way extension of ArzEn (code-switched audio + monolingual Egyptian Arabic + English translations) with ASR, MT, and ST baselines; the reference CS speech-translation corpus.
- **CoVoST 2** (Arabic subset) and **IWSLT dialectal tracks** (Tunisian Arabic to English); **SeamlessM4T** as the multitask baseline.
- **DARIJA-C:** Moroccan Darija speech for translation into MSA.

### 5.4 Pronunciation Assessment, Quran Tech, and CAPT

- **Iqra'Eval (ArabicNLP 2025):** the field's first open shared task for Quranic pronunciation assessment; phoneme-level mispronunciation detection and diagnosis on **QuranMB**, public baselines (Iqra_wavlm_base), an HF leaderboard, and winning recipes built on Whisper/wav2vec2-BERT speech-to-phoneme adaptation plus synthetic-error augmentation. Track yearly.
- **"Towards a Unified Benchmark for Arabic Pronunciation Assessment"** (HUMAIN + SDAIA, Interspeech 2025).
- **QVoice** (MBZUAI, arXiv 2305.07445): Arabic mispronunciation detection and pronunciation learning.
- **CATT-Whisper** diacritized Quranic ASR: 23.26% diacritic-aware WER, first place on its leaderboard, trained on ~870 h of recitations.
- This vertical (education + religious tech) is a major MENA commercial use case and deserves its own chapter.

### 5.5 Speaker Tasks, Anti-Spoofing, and Post-Processing

- **Diarization:** pyannote 3.x and NeMo work language-agnostically; QASR's speaker labels enable Arabic speaker-ID and speaker-linking research on broadcast audio.
- **Speaker verification:** research exists on emotional and stressful Arabic conditions, but there is no large open Arabic VoxCeleb equivalent. A documented gap and contribution magnet.
- **Deepfake / anti-spoofing:** ArVoice explicitly pairs human and synthetic voices for detection research; general ASVspoof protocols apply; MBZUAI's DeepfakeJudge is the adjacent line of work. Arabic-specific anti-spoofing is nearly empty.
- **Post-processing:** punctuation restoration (QASR annotations; NVIDIA FastConformer is the first open unified Arabic ASR + punctuation model), inverse text normalization as a first-class module, and LLM correction: Whisper-Large + GPT-style post-editing consistently lowers dialectal WER (2026 study; ALLaM also evaluated). Design pattern: ASR -> LLM corrector -> ITN.

**References:** [ADI-17](https://ieeexplore.ieee.org/document/9052982/) · [ADI-17 repo](https://github.com/swshon/arabic-dialect-identification) · [ADI-20](https://arxiv.org/abs/2511.10070) · [CTC-DID](https://arxiv.org/abs/2601.12199) · [BAVED](https://github.com/40uf411/Basic-Arabic-Vocal-Emotions-Dataset) · [KSUEmotions](https://catalog.ldc.upenn.edu/LDC2017S12) · [Arabic SER baselines](https://github.com/OmarMohammed88/AR-Emotion-Recognition) · [SER hybrid study 2026](https://arxiv.org/abs/2606.10278) · [ArzEn-ST](https://aclanthology.org/2022.wanlp-1.12/) · [Iqra'Eval](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/) · [QVoice](https://arxiv.org/abs/2305.07445)

## 6. Community Hubs and Legacy Catalogs

### 6.1 High-signal individual contributors on Hugging Face

- **Ahmed Wasfy (HF: oddadmix, ARBML member):** the most prolific individual contributor to Arabic speech data today (53 datasets). Key assets:
  - **Lahgtna project:** multi-dialect Arabic TTS data, models, and demos; models cover Egyptian, Saudi, Moroccan, and Iraqi with full diacritics support (the Saudi variant is noted for quality); datasets: lahgtna-v3-small (~54.6K rows), lahgtna-v3-small-augmented (~211K rows), msa-omnivoice-tts-v1 (~31K rows).
  - **arabic-audio-collection series:** dialect podcast and story audio for underrepresented varieties: Moroccan (wak3i, ameed, noone-stories), Tunisian (deep-confessions), Libyan (a7rar-podcast), Algerian (kahwa-podcast).
  - **Arabic ASR fine-tunes for 13 dialects:** whisper-small/medium/large-v3-turbo-arabic-dialectal and qwen3-asr-0.6b-arabic-dialectal.
  - **Nabra Arabic TTS:** MSA TTS fine-tuned from Kokoro-82M (a lightweight deployment option).
- **MAdel121/arabic-egy-cleaned:** ~72 h of cleaned, aligned Egyptian ASR data (16 kHz, normalized text, digits verbalized; only ~12% of the original 570 h survived cleaning). A worked example of aggressive data cleaning.
- **Other org pages to treat as living sources:** IbrahimSalah, NAMAA-Space, SWivid, MBZUAI, QCRI, Elyadata, UBC-NLP.

### 6.2 Legacy and paid catalogs (completeness layer)

- **LDC (paid, per-corpus license):** CALLHOME Egyptian Arabic (the classic 8 kHz telephone test bed), GALE Arabic broadcast corpora, Fisher Levantine conversational telephone speech, TRANSTAC Iraqi Arabic, KSUEmotions (LDC2017S12). Verify catalog numbers on ldc.upenn.edu before citing.
- **ELRA / older EU projects:** NEMLAR Arabic broadcast news, NetDC. Historic, but still cited in Kaldi-era papers.
- **Saudi academic corpora:** King Saud University Arabic Speech Database (phonetically rich), KACST Arabic phonetic database. Limited or paid access; relevant to the history chapter and Saudi-dialect lineage.

**References:** [oddadmix profile](https://huggingface.co/oddadmix) · [arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) · [NADI 2025 ASR data](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask2_ASR) · [LDC catalog](https://catalog.ldc.upenn.edu)

## 7. Licensing and Data Governance

This deserves a dedicated repo page; nothing like it exists elsewhere.

- **The SADA cascade (case study):** SADA is CC BY-NC-SA. Every model trained on it inherits the restriction. Habibi's Unified, SAU, and UAE checkpoints are CC-BY-NC-SA explicitly because of SADA and Mixat, while its ALG, EGY, IRQ, MAR, and MSA checkpoints ship under Apache 2.0. This is the clearest real-world example of license propagation in Arabic speech.
- **Broadcast corpora (MGB-2, QASR):** research licenses via registration; not for commercial products without separate arrangements.
- **Safest options for commercial work:** Common Voice (CC0), FLEURS (CC-BY), NAMAA-Egyptian-TTS (MIT weights), self-collected or vendor data, and Apache/MIT weights trained on permissive data.
- **Contamination governance:** several public benchmarks derive from splits of public corpora (SADA splits inside newer TTS benchmarks, for example). Before publishing any comparison, verify that the benchmark's source splits were not in your training data or the baseline's. Maintain an internal frozen test shard that never touches training.

**References:** [Habibi model card (license split)](https://huggingface.co/SWivid/Habibi-TTS) · [SADA analysis](https://arxiv.org/abs/2508.12968) · [NADI 2025 license terms](https://huggingface.co/datasets/UBC-NLP/NADI2025_subtask2_ASR)

---

## 8. Open ASR Models

| Model | Architecture | Coverage | Highlights |
|---|---|---|---|
| **Whisper large-v3 / turbo** (OpenAI) | Encoder-decoder transformer, weak supervision | Multilingual incl. ar | The universal baseline: strong on MSA, weak on dialects; ~36.9% average WER across the Open Universal leaderboard's six multi-dialect test sets (Aug 2025 snapshot) |
| **Omnilingual ASR** (Meta, Nov 2025) | wav2vec2 encoders 300M-7B with CTC and LLM-decoder heads (up to ~7.8B) | 1,600+ languages incl. Arabic varieties; zero-shot extension to 5,400+ with a few in-context samples | Apache 2.0; pretrained on ~4.3M h of unlabeled audio; the 7B LLM-ASR reaches CER <10% on 78% of supported languages; ships the Omnilingual ASR Corpus (3,350 h / 348 underserved languages) and newly collected recordings for many Arabic dialects. The new massively multilingual baseline |
| **NVIDIA FastConformer Arabic** | FastConformer hybrid (CTC + RNNT) with punctuation | MSA model + unified MSA/Classical (diacritic-aware) | Topped the open leaderboard; 11.37% WER on MASC, 9.76% on MCV, 7.73% on FLEURS; the unified model reaches 6.65% diacritic-and-punctuation-aware WER on EveryAyah; first open end-to-end Arabic ASR + punctuation release |
| **ArTST v1 / v1.5 / v2 / v3** (MBZUAI) | SpeechT5-style unified text + speech | v1: MSA; v1.5: MSA with diacritics (MGB-2 + Tashkeela); v2: dialectal with 17 fine-tuned dialect checkpoints; v3: Arabic + English/French/Spanish | The most complete open Arabic speech family: ASR, TTS, and dialect ID under one pretraining; ACL 2025 paper on dialectal coverage |
| **Ara-BEST-RQ** (2026) | BEST-RQ conformer SSL, 300M/600M | Multi-dialect; pretrained on 5,640 h of crawled CC speech plus public data | The 300M model beats HuBERT/XLS-R-class baselines on MGB-3, MGB-5, TARIC-SLU, and CV19; evidence that focused Arabic pretraining beats massive multilingual SSL at equal size |
| **MMS 1B** (Meta) | wav2vec2-based, 1,000+ languages | ar included | Fine-tuned on SADA with a 4-gram LM: 40.9% WER on SADA test-clean, a measure of how hard SADA is |
| **SeamlessM4T v2** (Meta) | Multitask S2T/S2S | MSA plus Moroccan (ary) and Egyptian (arz) dialect codes | Useful for the speech-translation angle |
| **Qwen3-ASR** (Alibaba) | LLM-based ASR | Multilingual incl. ar | Open small variants plus API; community Arabic-dialect fine-tunes already exist (qwen3-asr-0.6b-arabic-dialectal) |
| **uDistil-Whisper** | Distilled Whisper with label-free filtering | ar (SADA, Casablanca evals) | The recipe for smaller, faster Arabic Whisper without labels |
| **VoxArabica** | Whisper/XLS-R with dialect-ID routing | MSA, Egyptian, Moroccan, and more | Dialect-aware ASR system with an HF demo; a good reference architecture |
| **XLS-R / wav2vec2 fine-tunes** | SSL + CTC | Community checkpoints (elgeish, jonatasgrosman) | Older but ubiquitous; ideal teaching material for the CTC fine-tuning workflow |
| **Klaam (ARBML)** | Early CTC/FastSpeech2 stack | ar | Historic community repo (ASR + TTS + classification) for the history chapter |

**References:** [Whisper](https://arxiv.org/abs/2212.04356) · [Omnilingual ASR](https://arxiv.org/abs/2511.09690) · [Omnilingual repo](https://github.com/facebookresearch/omnilingual-asr) · [Meta blog](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/) · [omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B) · [FastConformer Arabic](https://arxiv.org/abs/2507.13977) · [ArTST repo](https://github.com/mbzuai-nlp/ArTST) · [ArTST v3 ASR](https://huggingface.co/MBZUAI/artst_asr_v3) · [Dialectal Coverage (ACL 2025)](https://arxiv.org/abs/2411.05872) · [Ara-BEST-RQ](https://arxiv.org/abs/2603.21900) · [uDistil-Whisper](https://arxiv.org/abs/2407.01257) · [VoxArabica](https://arxiv.org/abs/2310.11069) · [Arab Voices survey](https://arxiv.org/abs/2601.13319)

## 9. Open TTS Models

| Model | Base | Coverage | License | Notes |
|---|---|---|---|---|
| **Habibi-TTS** (Jan 2026) | F5-TTS v1 (flow-matching DiT, ~335M) | Unified + specialized checkpoints; dialect IDs: MSA, SAU, UAE, ALG, IRQ, EGY, MAR, OMN, TUN, LEV, SDN, LBY | Unified/SAU/UAE: CC-BY-NC-SA (SADA + Mixat); ALG/EGY/IRQ/MAR/MSA: Apache 2.0 | Zero-shot voice cloning with no diacritization required (MSA-to-dialect curriculum learning); ~8,000 H100-hours of ablations; human evals competitive with ElevenLabs v3 alpha; pip package and Gradio GUI |
| **SILMA TTS v1/v2** | F5-TTS trained from scratch | Bilingual Arabic/English | Open weights | Large public + proprietary training data; SILMA also maintains the open Arabic TTS benchmark |
| **Arabic-F5-TTS-v2** (IbrahimSalah) | F5-TTS fine-tune | MSA, fully diacritized | Open | 300 h of clean audio; diacritics-in-vocabulary strategy (training data duplicated with and without tashkeel) |
| **Lahgtna** (Ahmed Wasfy) | Dialect TTS line | Egyptian, Saudi, Moroccan, Iraqi with full diacritics support | Open | Paired with the Lahgtna datasets (section 6.1); the Saudi variant is noted for quality |
| **EGTTS-V0.1** (2024) | XTTS v2 fine-tune | Egyptian | CPML (inherited) | The reference community Egyptian XTTS fine-tune; live HF demo |
| **NAMAA-Egyptian-TTS** | Chatterbox Multilingual | Conversational Egyptian (not MSA) | **MIT** (commercial-friendly) | Honestly documented limits (qaf occasionally dropped, numbers sometimes misread); a rare MIT-licensed dialect TTS |
| **Community F5 fine-tunes** | F5/Habibi checkpoints | Egyptian (16 h conversational), Najdi (NAMAA-Saudi-TTS-V2, 18.4 h full fine-tune) | Mostly CC-BY-NC-SA inherited | Worked examples of the LoRA-vs-full-fine-tune tradeoff; NAMAA documents why low-rank LoRA fails on messy conditional generation |
| **ArTST TTS** (speecht5_tts_clartts_ar) | SpeechT5 | Classical Arabic / MSA | Open | Runs in four lines via the HF TTS pipeline; needs x-vector speaker embeddings |
| **MMS-TTS-Ara** (Meta) | VITS | MSA-leaning single voice | CC-BY-NC | End-to-end with no external vocoder; weak prosody but trivially deployable |
| **XTTS-v2** (Coqui, Idiap fork) | GPT-style + HiFi-GAN | 17 languages incl. ar, zero-shot cloning | CPML (non-commercial) | Still the most-used cloning baseline in tutorials |
| **Chatterbox Multilingual** (Resemble) | 0.5B LLM-based | 23 languages incl. ar | MIT | The base of the notable open Egyptian fine-tunes (2026) |
| **Spark-TTS** | LLM + single-stream decoupled speech tokens | Community Arabic fine-tunes exist | Open | The main LLM-codec alternative to flow matching |
| **Fish Speech / OpenAudio** | LLM-based | Multilingual; Arabic among supported languages (verify quality) | Open weights | Worth benchmarking; the Arabic share of training data is small |
| **tts-arabic-pytorch** (nipponjo) | FastPitch / Tacotron2 | MSA (ASC-trained) | Open | Lightweight classic pipeline; pairs with the mantoq Arabic G2P package |
| **Nabra** (oddadmix) | Kokoro-82M fine-tune | MSA | Open | A lightweight-TTS deployment option |
| **NatiQ (QCRI), Fanar-TTS** | Research / platform | MSA (+dialect work) | Not fully open | Reference points for the research chapter |

**References:** [Habibi-TTS repo](https://github.com/SWivid/Habibi-TTS) · [Habibi weights](https://huggingface.co/SWivid/Habibi-TTS) · [F5-TTS](https://github.com/SWivid/F5-TTS) · [SILMA TTS](https://silma.ai/open-source-arabic-tts-models) · [Arabic-F5-TTS-v2](https://huggingface.co/IbrahimSalah/Arabic-F5-TTS-v2) · [NAMAA-Saudi-TTS-V2](https://huggingface.co/NAMAA-Space/NAMAA-Saudi-TTS-V2) · [NAMAA-Egyptian-TTS](https://huggingface.co/NAMAA-Space/NAMAA-Egyptian-TTS) · [EGTTS-V0.1](https://huggingface.co/OmarSamir/EGTTS-V0.1) · [EGTTS repo](https://github.com/joejoe03/Egyptian-Text-To-Speech) · [f5-tts-egyptian](https://huggingface.co/MAdel121/f5-tts-egyptian-arabic) · [ArTST TTS](https://huggingface.co/MBZUAI/speecht5_tts_clartts_ar) · [Spark-TTS](https://arxiv.org/abs/2503.01710) · [XTTS](https://arxiv.org/abs/2406.04904)

## 10. Speech-LLMs and Omni Models

- **Fanar (QCRI):** Arabic-centric multimodal platform with speech in and out and dialect support; its in-house ASR is reported to beat Whisper-large-v2 and Google ASR on LAraBench and SADA; Arabic-centric 10K-BPE speech vocabulary (75% Arabic tokens).
- **LLMVoX (MBZUAI, 2025):** a lightweight streaming TTS that bolts onto any LLM; the Arabic adaptation needed only ~450K text entries plus 1,500 h of XTTS synthetic speech; roughly 10x faster than XTTS at competitive WER; preferred over Freeze-Omni in human evals.
- **"Harmonizing the Arabic Audio Space with Data Scheduling" (2026):** the first systematic multi-task training recipe for an Arabic-centric Omni model spanning acoustic, linguistic, and paralinguistic tasks.
- **PolySpeech-100 finding (2026):** for Arabic dialects, cascade systems (Whisper-v3 + LLM) still beat end-to-end omni models (62.1% vs 52.1% dialect accuracy), the reverse of the Chinese-dialect trend. Practical takeaway: cascades remain state of the art for Arabic voice agents today; document when that flips.
- **LLM-to-Speech (2026):** a synthetic-data pipeline that turns LLM-generated dialect text into TTS training data; the emerging answer to dialect data scarcity.

**References:** [Fanar](https://arxiv.org/abs/2501.13944) · [Fanar platform](https://www.fanar.qa/en) · [Harmonizing the Arabic Audio Space](https://arxiv.org/abs/2601.12494) · [PolySpeech-100](https://arxiv.org/abs/2606.01016) · [LLM-to-Speech](https://arxiv.org/abs/2602.15675)

## 11. Commercial APIs (MENA-Focused)

**STT:** Munsit (claims #1 on the open leaderboard at ~24.5% average WER; sovereign GCC deployment), ElevenLabs Scribe, Speechmatics (historically strong Arabic), Azure Speech (per-dialect locales: ar-SA, ar-EG, ar-AE, and more), Google STT, AWS Transcribe, Deepgram (reported to trail specialized Arabic models by up to ~20 points on Gulf dialects), AssemblyAI, Gladia, Cohere Transcribe (2026), Kanari AI (QCRI lineage), Intella (Saudi), Hamsa (Arabic-first voice platform: automatic dialect detection across Saudi, Emirati, Jordanian, Egyptian, Iraqi, Levantine, and MSA; code-switching; streaming and batch; custom vocabulary; public/private-cloud and self-hosted deployment). Note: the open nadsoft/hamsa-v0.1-beta Whisper fine-tune on Jordanian audio is an unrelated older project with the same name.

**TTS:** ElevenLabs v3 (dialect-capable; the bar Habibi benchmarked against), Azure Neural (Hamed/Zariyah plus coverage of 16 Arab locales), Amazon Polly (Zeina MSA, Hala Gulf), Google Cloud TTS, PlayHT, Cartesia, Camb.ai, OpenAI TTS, Qwen3-TTS (used in 2026 for Emirati and Saudi dialect products), Hamsa (Gulf, Levantine, Egyptian, North African, and MSA voices plus voice and phone agents), Fanar platform (free, Qatar).

**Chapter angles:** per-dialect accuracy variance (always pilot on your own audio), pricing models (per-minute vs credits at scale), data residency and PDPL (KSA) / UAE data-protection constraints, and on-prem or sovereign options.

**References:** [Munsit](https://munsit.com) · [Hamsa](https://tryhamsa.com) · [Hamsa docs](https://docs.tryhamsa.com/overview/introduction) · [Fanar platform](https://www.fanar.qa/en) · [SILMA](https://silma.ai/open-source-arabic-tts-models)

---

## 12. Phonology, Orthography, and Transliteration

**MSA phoneme inventory:** 28 consonants including emphatics (Sad, Dad, Ta, Dha), pharyngeals (Hha, Ain), and uvular qaf; 3 short + 3 long vowels + 2 diphthongs; gemination (shadda); hamza behavior; sun/moon-letter assimilation. Nearly every TTS frontend bug traces back to one of these.

**Dialect phonology variation (build a full matrix in the repo):**
- qaf: [g] in Gulf and Bedouin varieties, glottal stop in Cairo and urban Levant, [q] in MSA and urban Maghreb, [k] in rural Palestinian.
- jim: [g] in Cairo, [zh] in the Levant, [j/dj] in the Gulf and Iraq.
- Interdentals (th, dh): preserved in the Gulf, Iraq, and Tunisia; merged to t/s and d/z in Egypt and the urban Levant.
- Vowel reduction and stress-shift patterns per dialect.

**Orthography and transliteration systems:**
- **Buckwalter and Safe Buckwalter:** ASCII round-trip transliteration used in Kaldi-era recipes and LDC corpora.
- **CODA / CODA*:** the Conventional Orthography for Dialectal Arabic (Habash et al.); the standard answer to "how do I transcribe dialect consistently." Adopt it as the repo's recommended transcription convention.
- **Arabizi** (chat alphabet: 3 = ain, 7 = Hha, 2 = hamza): appears in real user text and code-switched transcripts; needs a transliteration layer (Casablanca includes Arabizi-style code-switching in both scripts).

**References:** [CACM: Arabic voice generation](https://cacm.acm.org/arab-world-regional-special-section/unlocking-the-potential-of-arabic-voice-generation-technologies/) · [ASC phonetic design](https://huggingface.co/datasets/halabi2016/arabic_speech_corpus)

## 13. The Nine Core Challenges

**13.1 Diglossia and the dialect continuum.** Arabic spans 30+ spoken varieties with substantial lexical and phonological divergence. Zero-shot multilingual models (Whisper, SeamlessM4T, MMS) show high WER/CER across dialects (the Casablanca finding); MSA-trained systems do not transfer. The repo needs a dialect-coverage matrix: which dataset and which model cover which variety, at how many hours.

**13.2 No standard orthography for dialects.** The same spoken word is written multiple ways (SawtArabi's example: "thalatha" written with tha or ta). This corrupts ASR references, inflates WER unfairly, and breaks TTS frontends. Mitigations: CODA*-based normalization guidelines per dialect, CER alongside WER, human-verified benchmark transcripts.

**13.3 Diacritics (tashkeel).** Written Arabic omits short vowels; identical skeletons carry different pronunciations and meanings.
- TTS: either require diacritized input (the Arabic-F5-TTS-v2 approach: diacritics as vocabulary tokens, data duplicated with and without tashkeel) or learn implicit diacritization from scale and curriculum (Habibi needs none at inference).
- ASR: references are usually stripped for scoring; diacritic-aware WER is a separate, much harder metric (NVIDIA's unified model: 6.65% on Quranic speech).
- Speech-based diacritic restoration degrades badly versus text-based (reported DERs of 4-50%).

**13.4 Code-switching.** Arabic-English (and Arabic-French in the Maghreb) mixing is the norm. Casablanca annotates CS in both Latin script and Arabic transliteration. In TTS, English tokens inside Arabic sentences get "Arabized" pronunciation in most models, a consistently reported production complaint. SawtArabi is the reference CS-TTS benchmark; per-token language ID with language-specific phonemization is the standard mitigation.

**13.5 Numbers, dates, and inverse text normalization.** Digit rendering is a top TTS failure mode (models emit gibberish for numerals). Requires tafqit (number-to-words) with correct gender and case agreement, Eastern vs Western numeral handling, and currency/date expansion. Leaderboards convert Eastern to Western numerals during ASR scoring.

**13.6 Evaluation normalization.** The de-facto recipe (Open Universal Arabic ASR Leaderboard): strip punctuation, strip diacritics, normalize hamza/madda variants, unify numerals, then compute WER and CER. Publish the exact normalizer code; tiny normalization differences swing Arabic WER by whole points. Full methodology in section 17.

**13.7 Data quality in broadcast corpora.** SADA is under one third clean, one third noisy, one third music-overlaid. Broadcast-heavy training teaches models that background music is "normal audio." Mitigations: cleanliness-weighted sampling, denoising (Demucs, DeepFilterNet, resemble-enhance), quality gates (DNSMOS, UTMOSv2, SNR), and silence/room-tone augmentation for TTS canvas robustness.

**13.8 Phonemization.** espeak-ng's Arabic rules are imperfect (emphatics, pharyngeals, dialectal qaf realizations). SawtArabi ships improved espeak-ng rules; alternatives are Nawar Halabi's Arabic-Phonetiser, mantoq, or character-level modeling (the F5 approach) that skips phonemes entirely.

**13.9 Benchmark contamination.** When public corpora feed both training sets and benchmarks (SADA splits inside newer TTS benchmarks, for example), public comparisons can be silently contaminated. Repo rule: document data lineage for every benchmark; keep a frozen, never-trained-on internal shard.

**References:** [Leaderboard normalization](https://arxiv.org/abs/2412.13788) · [SawtArabi](https://www.isca-archive.org/interspeech_2025/lodagala25_interspeech.pdf) · [Casablanca](https://arxiv.org/abs/2410.04527) · [CATT](https://arxiv.org/abs/2407.03236) · [Arabic Diacritics in the Wild](https://arxiv.org/abs/2406.05760) · [Habibi](https://arxiv.org/abs/2601.13802)

## 14. Vocoders and Neural Audio Codecs

- **Vocoder lineage:** Griffin-Lim -> WaveNet/WaveRNN -> GAN vocoders (HiFi-GAN, BigVGAN) -> Vocos. Teach what each solves and which artifacts to listen for.
- **Neural codecs for LLM-TTS:** EnCodec, DAC, SNAC, Mimi (Moshi), and single-stream decoupled tokens (Spark-TTS). Codec choice controls the latency floor and the quality ceiling.
- **Sample-rate reality:** 16 kHz for ASR, 22.05/24/44.1 kHz for TTS, 8 kHz for telephony; resampling pitfalls; loudness normalization (EBU R128).

## 15. Text Resources for LMs and TTS Frontends

- **Diacritized text:** Tashkeela (~75M words), the WikiNews diacritized test set, the CATT benchmark set.
- **LM corpora:** the QASR 130M-word text release, Arabic Gigaword (LDC), AraMix, 101 Billion Arabic Words, and the Arabic splits of mC4/OSCAR.
- **Tokenizers:** tokenizer behavior on Arabic matters for LLM-TTS and LM fusion; point readers to open Arabic tokenizer comparisons (Tokenizer Arena-style leaderboards) and morphology-aware segmentation tradeoffs (Farasa/CAMeL).

---

## 16. Tooling Landscape

| Layer | Tools |
|---|---|
| **Diacritization** | Farasa (QCRI), CAMeL Tools + Camelira (NYU-AD), Mishkal, CATT (Abjad; open models + benchmark), Shakkelha, Fine-Tashkeel, Sadeed (Misraj). Corpora: Tashkeela, WikiNews |
| **G2P / phonemization** | espeak-ng (with SawtArabi's modifications), Arabic-Phonetiser (Halabi), mantoq, the phonemizer wrapper library |
| **Normalization / tafqit** | PyArabic, camel-tools utilities, num2words (Arabic), custom ITN rules for currencies and dates |
| **Forced alignment** | Montreal Forced Aligner (Arabic acoustic model + dictionary), ctc-forced-aligner (MMS-based), WhisperX (Arabic wav2vec2 aligner), NeMo Forced Aligner |
| **VAD / diarization** | Silero VAD, pyannote.audio 3.x, NVIDIA NeMo diarization |
| **Enhancement / filtering** | Demucs, DeepFilterNet, resemble-enhance; quality scoring with DNSMOS, UTMOSv2, NISQA |
| **Data pipelines** | Emilia-Pipe (in-the-wild speech to TTS-grade data), NeMo Speech Data Processor, Lhotse; Habibi's curation pipeline is the Arabic reference recipe; NAMAA's YouTube Audio Extractor (Gemini-assisted transcription + VAD filtering) is a worked example for building dialect ASR data from YouTube |
| **Training frameworks** | HF Transformers (Whisper fine-tuning), NVIDIA NeMo (FastConformer recipes), ESPnet (MGB-2 recipe), SpeechBrain, k2/icefall, fairseq/fairseq2, F5-TTS and Habibi-TTS repos, coqui-tts (Idiap fork), Amphion |
| **Inference / serving** | faster-whisper (CTranslate2), whisper.cpp, sherpa-onnx (streaming, on-device), Vosk (offline lightweight Arabic models for edge and mobile), WhisperLive / RealtimeSTT / RealtimeTTS for streaming prototypes, NVIDIA Triton/TensorRT |

**References:** [CATT](https://arxiv.org/abs/2407.03236) · [ArTST fine-tuning notebooks](https://github.com/mbzuai-nlp/ArTST) · [Leaderboard eval code](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard) · [Habibi-TTS repo](https://github.com/SWivid/Habibi-TTS)

## 17. Evaluation Methodology

**ASR specification:**
1. Normalize both reference and hypothesis: strip punctuation and diacritics, normalize hamza/madda and alif/ya variants, convert Eastern to Western numerals.
2. Report WER and CER together; CER is more stable for morphologically rich text.
3. Treat diacritic-aware WER as a separate advanced metric (Quranic and CA use cases).
4. Emerging additions: semantic distance (SemDist) and LLM-as-judge for meaning-preserving errors.
5. Run the contamination checklist from section 7 before publishing any comparison.

**TTS specification:**
- **Subjective:** MOS/CMOS panels; Habibi's decomposition is the Arabic reference: D-MOS (dialect pronunciation accuracy), S-MOS (speaker similarity), N-MOS (naturalness).
- **Objective:** ASR-based intelligibility (WER-S/O: word error rates from two different ASR systems), UTMOS/UTMOSv2 and DNSMOS/NISQA for quality, speaker similarity via WavLM/ECAPA embeddings.
- **Arena signal:** report Arabic TTS Arena standing where applicable; human preference catches what static metrics miss.
- **Coverage:** evaluate per dialect, on code-switched inputs, and on numeral-heavy text; these are the three standard failure surfaces.

**References:** [ASR Leaderboard methodology](https://arxiv.org/abs/2412.13788) · [Habibi metrics (WER-S/O, D/S/N-MOS)](https://arxiv.org/abs/2601.13802) · [Arabic TTS Arena](https://huggingface.co/spaces/Navid-AI/Arabic-TTS-Arena)

## 18. Production: Voice Agents, Streaming, and Telephony

- **Architecture choice:** cascade (ASR -> LLM -> TTS) still beats end-to-end omni models for Arabic dialects (PolySpeech-100: 62.1% vs 52.1%); document when that flips.
- **Streaming components:** Silero VAD, endpointing and turn-taking, streaming ASR (sherpa-onnx, NeMo cache-aware, whisper-streaming), streaming TTS (the LLMVoX pattern, chunked flow-matching inference), barge-in handling.
- **Latency budgets:** first ASR partial under 300 ms, LLM TTFT, TTS TTFB under 300 ms; end-to-end conversational target under ~800 ms.
- **Telephony reality:** 8 kHz narrowband (G.711) degrades models trained on 16 kHz+; call-center Arabic combines dialects, code-switching, noise, and crosstalk; upsampling does not recover lost bands; plan telephony-domain fine-tuning.
- **Frameworks:** LiveKit Agents, Pipecat, Vocode; regional managed stacks: Hamsa and Munsit voice/phone agents.

**References:** [PolySpeech-100](https://arxiv.org/abs/2606.01016) · [Hamsa](https://tryhamsa.com) · [Munsit](https://munsit.com)

## 19. Data Collection, Annotation, and Legal

- **Crowdsourcing guidelines:** "Best practices for crowdsourcing dialectal Arabic speech transcription" (Wray & Mubarak, 2015).
- **Transcription conventions:** CODA* for dialects; Casablanca-style code-switching tags (Latin script plus Arabic transliteration).
- **Tools:** Praat, ELAN, Label Studio; alignment-assisted transcription with Whisper pre-labels and human correction.
- **Quality control:** dual annotation with adjudication, inter-annotator WER, audio-quality gates (SNR, DNSMOS).
- **Synthetic data:** LLM-generated dialect text to TTS to ASR training (the LLM-to-Speech pipeline); XTTS-synthetic bootstrapping (the LLMVoX recipe: 1,500 h of synthetic speech sufficed).
- **Legal and ethical:** voice consent and cloning policy, speaker anonymization, PDPL (Saudi) and UAE data-protection awareness for voice data. Link primary sources; this is not legal advice.

**References:** [LLM-to-Speech](https://arxiv.org/abs/2602.15675) · [Casablanca CS annotation scheme](https://arxiv.org/abs/2410.04527)

---

## 20. Ecosystem Directory

*Maintain this section as a living page.*

- **Research labs:** QCRI (Doha), MBZUAI Speech Lab and CAMeL Lab NYUAD (Abu Dhabi), SDAIA/NCAI (Riyadh), HUMAIN (Riyadh), ELM Research (Riyadh), KAUST, UBC-NLP (Vancouver; Casablanca, NADI), Elyadata (Tunis), LIA Avignon (ADI-20).
- **Companies:** SILMA AI, Misraj (Sadeed, Baseer), NAMAA, Abjad (CATT), Kanari AI, Intella, Munsit, Hamsa, Tarteel AI, TII (Falcon), Inception/G42, Navid AI (Arabic TTS Arena, Arabic RAG Leaderboard, Yehia), ScienceSoft dialect-TTS R&D.
- **Individual contributors:** Ahmed Wasfy / oddadmix (Lahgtna, dialect audio collections, 13-dialect ASR fine-tunes, Nabra), Ibrahim Salah (F5/Spark Arabic TTS), MAdel121 (Egyptian ASR/TTS), SWivid (Habibi).
- **Communities:** ARBML, arabicspeech.org, HF Arabic speech organizations and collections.
- **Venues and shared tasks:** MGB-2/3/5 challenges, ArabicNLP (ex-WANLP) shared tasks including NADI and Iqra'Eval, Interspeech Arabic sessions, OSACT at LREC, VarDial.

**References:** [oddadmix](https://huggingface.co/oddadmix) · [Navid-AI](https://huggingface.co/Navid-AI) · [ARBML](https://huggingface.co/arbml) · [arabicspeech.org](https://arabicspeech.org/)

## 21. Reading List

*Grouped by topic; arXiv IDs included where known.*

- **Corpora:** MGB-2 (2016), MGB-3 (2017), MGB-5 (2019), QASR (2106.13000), Common Voice (2020), SADA (ICASSP 2024), MASC, Aswat (ArabicNLP 2023), Casablanca (2410.04527, EMNLP 2024), Mixat, ArzEn (LREC 2020) + ArzEn-ST (2211.12000), Ramsa (2603.08125), ADI-17 (ICASSP 2020) + ADI-20 (2511.10070).
- **ASR modeling:** Whisper (2212.04356), N-Shot Whisper on Arabic (Talafha 2023), VoxArabica (2310.11069), ArTST (2310.16621) + Dialectal Coverage and Generalization (2411.05872, ACL 2025), uDistil-Whisper (2407.01257), Open ASR for Classical and MSA / FastConformer (2507.13977), the SADA transformer study (2508.12968), Ara-BEST-RQ (2603.21900), Omnilingual ASR (2511.09690), Zero-Shot Context-Aware ASR for Arabic Varieties (2511.18774), Arabic Little STT (2510.23319), CV-18 NER (2604.02209).
- **Benchmarks:** Open Universal Arabic ASR Leaderboard (2412.13788, Interspeech 2025), SawtArabi (Interspeech 2025), Iqra'Eval (ArabicNLP 2025), LAraBench.
- **TTS:** the ASC thesis (Halabi 2016), ClArTTS (2023), NatiQ, ArVoice (2505.20506), Habibi (2601.13802), LLM-to-Speech (2602.15675), F5-TTS (ACL 2025), Spark-TTS (2503.01710), XTTS (2406.04904).
- **Diacritization:** CATT (2407.03236), Arabic Diacritics in the Wild (2406.05760), Camelira, Sadeed.
- **Speech-LLM / Omni:** Fanar (2501.13944), LLMVoX (2025, MBZUAI), Harmonizing the Arabic Audio Space (2601.12494), PolySpeech-100 (2606.01016).
- **Surveys:** "Unlocking the Potential of Arabic Voice-Generation Technologies" (CACM, 2025), the best single survey of Arabic TTS; "Arab Voices: Mapping Standard and Dialectal Arabic Speech Technology" (2601.13319, Jan 2026), the most recent full mapping of Arabic ASR models and dialect coverage (documents, for example, SeamlessM4T v2's MSA + ary + arz support).

## 22. Learning Resources

- Hugging Face Audio Course (free, hands-on Whisper and TTS).
- Jurafsky & Martin, Speech and Language Processing (3rd edition drafts): the ASR and TTS chapters.
- CMU 11-751 Speech Recognition and Understanding lectures.
- "Neural Speech Synthesis" survey (Tan et al., 2106.15561) as the TTS-chapter backbone.
- Nizar Habash, "Introduction to Arabic Natural Language Processing" for the linguistic grounding.
- ESPnet, NeMo, and SpeechBrain official tutorials; F5-TTS and Habibi-TTS repository documentation.

**References:** [HF Audio Course](https://huggingface.co/learn/audio-course) · [Neural Speech Synthesis survey](https://arxiv.org/abs/2106.15561)

## 23. Gap Analysis

Why this repository wins: existing hubs (Awesome-Arabic-AI, Awesome_Arabic_NLP, arabicspeech.org) are link lists. None offer:

1. **Textbook-depth chapters** teaching how STT and TTS actually work, in Arabic and English.
2. **A license-risk matrix** with the SADA/Habibi propagation case study; nobody covers commercial-use safety.
3. **A dialect-coverage matrix** (dataset x dialect x hours; model x dialect).
4. **Reproducible notebooks** per task: fine-tune Whisper on a dialect, fine-tune F5/Habibi, build the eval normalizer, run forced alignment for Arabic.
5. **Production engineering:** streaming ASR, latency budgets, ITN, telephony, and code-switch handling in real systems.
6. **An evaluation specification:** the exact normalization and metrics recipe plus contamination checklists.

## 24. Proposed Repository Structure

```
arabic-speech-handbook/
├── README.md                     Bilingual overview + learning paths
├── 01-foundations/               ASR, TTS, SSL, vocoders-and-codecs
├── 02-arabic-linguistics/        Phonology, dialect matrix, orthography/CODA, Arabizi, diacritics
├── 03-arabic-challenges/         Diglossia, CS, numbers/ITN, orthography noise, contamination
├── 04-datasets/                  ASR, TTS, task-specific, dialect matrix, licensing risk
├── 05-models/                    ASR, TTS, speech-LLMs, commercial APIs
├── 06-adjacent-tasks/            Dialect ID, emotion, translation, speaker, pronunciation/Quran, anti-spoofing, post-processing
├── 07-tutorials/                 Notebooks: whisper-dialect, f5-habibi, alignment, dataset pipeline, eval normalizer, dialect ID, emotion
├── 08-evaluation/                ASR spec, TTS spec, contamination checklist
├── 09-production/                Streaming, latency, telephony-8k, voice agents, deployment, failure catalog
├── 10-data-collection/           Annotation guidelines, CODA transcription, synthetic data, legal-ethics
├── 11-ecosystem/                 Labs, companies, communities, shared tasks, commercial comparison
├── 12-learning-resources/
├── 13-papers/                    Annotated reading list
└── CONTRIBUTING.md
```

---

*Verification note: license terms, model rankings, and product claims change quickly in this field. Re-verify every entry against its original source before publishing the corresponding repo page. Last compiled: August 2026.*
