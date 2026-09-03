# Visual Speech Recognition from Lip Movements

End-to-end deep learning system for transcribing spoken sentences directly from video, without using the audio signal.

The project combines computer vision, sequence modelling, and CTC-based decoding to recognize speech from lip movements. It was developed as part of my Bachelor thesis in Biomedical Engineering and focuses on building the complete visual speech recognition pipeline, from raw video frames to predicted text.

---

## Overview

Traditional speech recognition relies primarily on audio.

This project explores a different problem:

> Can spoken language be reconstructed using only visual information from a speaker's lip movements?

The system receives a short video sequence, detects and isolates the mouth region, extracts spatiotemporal visual features, models their temporal evolution, and predicts the corresponding character sequence.

The complete pipeline is:

```text
Video
  ↓
Frame extraction
  ↓
Face detection
  ↓
Facial landmarks
  ↓
Lip region extraction
  ↓
Normalization
  ↓
3D Convolutional Neural Network
  ↓
Bidirectional GRU
  ↓
Character probabilities
  ↓
CTC decoding
  ↓
Predicted sentence
```

No audio features are used during inference.

---

## Key Features

* End-to-end visual speech recognition
* Fully video-based inference without audio input
* Custom preprocessing pipeline using OpenCV and dlib
* Facial landmark-based mouth extraction
* Fixed-length temporal video sequences
* 3D convolutional feature extraction
* Bidirectional recurrent sequence modelling
* Character-level prediction
* Connectionist Temporal Classification loss
* Custom TensorFlow training pipeline
* Speaker-based training strategy
* Learning-rate scheduling
* Model checkpointing
* Prediction monitoring during training
* GRID corpus integration

---

# Problem Formulation

Lipreading can be interpreted as a sequence-to-sequence problem.

Given a sequence of video frames:

```text
X = {x₁, x₂, ..., xₜ}
```

the objective is to predict a sequence of characters:

```text
Y = {y₁, y₂, ..., yₙ}
```

where the length of the predicted text does not necessarily correspond directly to the number of video frames.

This creates an alignment problem.

There is no explicit information indicating which frame corresponds to which letter.

For example, a single spoken phoneme may span several frames, while some characters may correspond to visually similar mouth configurations.

Instead of manually aligning every video frame with a character, the system uses **Connectionist Temporal Classification** to learn the alignment automatically.

---

# Dataset

The model is trained on the **GRID audiovisual sentence corpus**.

GRID contains recordings of multiple speakers producing short sentences following a predefined grammatical structure.

Each utterance is accompanied by an alignment file containing the spoken sequence and word timing information.

The complete GRID corpus contains:

* 34 speakers
* 1,000 utterances per speaker
* fixed sentence grammar
* synchronized video, audio, and alignment data

For this project, the first **22 speakers** were used during development and training.

The repository also includes a download link for the corresponding subset and alignment files.

---

# GRID Sentence Structure

GRID sentences follow a constrained grammar:

```text
command + color + preposition + letter + digit + adverb
```

An example sentence could look like:

```text
place red at C two now
```

This constrained vocabulary makes GRID particularly useful for developing and evaluating visual speech recognition systems.

Although the vocabulary is limited compared with unrestricted natural speech, the task remains challenging because many spoken sounds produce similar visual patterns.

---

# Visual Preprocessing Pipeline

A substantial part of the project is dedicated to converting raw video into a consistent neural-network input.

The preprocessing pipeline was implemented using:

* OpenCV
* dlib
* NumPy
* TensorFlow

Each video is processed frame by frame.

---

## Face Detection

The first stage locates the speaker's face within each frame.

Once the face has been detected, facial landmarks are calculated using a landmark predictor.

These landmarks provide stable reference points for locating the mouth independently of the speaker's position within the video.

---

## Lip Landmark Extraction

The mouth region is determined from facial landmark points:

```text
48–68
```

These points describe the outer and inner contours of the lips.

Rather than passing the entire video frame to the neural network, the system extracts only the region surrounding the mouth.

This significantly reduces irrelevant information.

The network does not need to model:

* background
* clothing
* most facial regions
* image borders
* unrelated visual motion

Instead, its capacity can be focused on the part of the frame that contains the most important speech-related visual information.

---

## Grayscale Conversion

The extracted mouth regions are converted to grayscale.

Color provides relatively little information for the lipreading task compared with:

* lip geometry
* mouth opening
* movement
* teeth visibility
* temporal motion patterns

Using a single channel also reduces the input dimensionality and computational cost.

---

## Spatial Normalization

Each extracted mouth region is resized to:

```text
64 × 64
```

This produces a fixed-size spatial representation regardless of the original video dimensions or the detected face size.

Each processed frame therefore has the shape:

```text
64 × 64 × 1
```

---

## Temporal Representation

Each input example contains:

```text
75 frames
```

The complete input tensor for a single sample is therefore:

```text
75 × 64 × 64 × 1
```

or, including the batch dimension:

```text
batch × 75 × 64 × 64 × 1
```

This structure preserves the temporal dimension rather than treating individual frames as independent images.

That distinction is essential.

Lipreading depends much more on movement than on individual static mouth shapes.

---

# Text Processing

The alignment files are processed independently from the videos.

Silence tokens are removed and the remaining spoken text is converted into a sequence of character indices.

The model operates at the character level rather than predicting complete words directly.

The vocabulary contains 41 output classes representing:

* letters
* numbers
* punctuation
* whitespace
* additional supported symbols

Labels are padded to a fixed representation during batching.

The model ultimately learns a mapping between visual movement sequences and character probability sequences.

---

# Model Architecture

The neural network combines two complementary components:

1. **3D Convolutional Neural Network**
2. **Bidirectional Gated Recurrent Units**

The convolutional part learns visual and short-term motion features.

The recurrent part models longer temporal relationships between those features.

The overall architecture is:

```text
75 × 64 × 64 × 1 video
          ↓
       Conv3D
          ↓
     MaxPool3D
          ↓
       Conv3D
          ↓
     MaxPool3D
          ↓
       Conv3D
          ↓
     MaxPool3D
          ↓
 TimeDistributed Flatten
          ↓
        BiGRU
          ↓
        BiGRU
          ↓
    Dense + Softmax
          ↓
75 × 41 probability sequence
          ↓
      CTC Decoder
          ↓
       Text
```

---

# 3D Convolutional Feature Extraction

Standard 2D convolution processes each image spatially.

For video, however, useful information also exists between consecutive frames.

The model therefore uses **3D convolutions**.

A 3D convolution operates simultaneously across:

```text
time × height × width
```

allowing the network to detect short-term motion patterns directly.

The convolutional stack uses three Conv3D stages with approximately:

```text
128 → 256 → 64 filters
```

and:

```text
3 × 3 × 3 kernels
```

Pooling is applied primarily over the spatial dimensions:

```text
1 × 2 × 2
```

This is intentional.

Spatial resolution can be progressively reduced, while the temporal dimension should remain available to the sequence model.

---

# Why 3D CNNs?

A single frame can tell the model what the mouth currently looks like.

It cannot reliably tell the model what sound is being produced.

Different phonemes can generate extremely similar static lip shapes.

Motion provides additional information.

For example:

```text
closed lips
   ↓
rapid opening
   ↓
rounded configuration
```

contains much more information than any of those frames individually.

3D convolutions allow the network to capture these short temporal transitions before the features are passed to the recurrent layers.

---

# Temporal Sequence Modelling

After the convolutional blocks, spatial features are flattened independently for every time step using a `TimeDistributed` transformation.

The resulting sequence is then passed through two **Bidirectional GRU layers**.

Each GRU uses:

```text
128 units
```

with dropout regularization.

---

## Why Bidirectional GRUs?

Speech interpretation often depends on context.

A particular visual movement may be ambiguous on its own but become easier to interpret when both previous and subsequent mouth movements are considered.

A conventional recurrent network only receives past context.

A bidirectional network processes the sequence in both directions:

```text
past → present → future
future → present → past
```

The representation at each time step therefore contains information from both surrounding directions.

This is particularly useful for offline lipreading, where the complete video sequence is available before decoding begins.

---

# Character Prediction

The final dense layer produces a probability distribution over the character vocabulary for every time step.

The output tensor has the shape:

```text
batch × 75 × 41
```

Each of the 75 temporal positions contains a probability distribution over possible characters.

However, the network is not expected to produce exactly one character per frame.

This is where CTC becomes important.

---

# Connectionist Temporal Classification

Direct frame-to-character supervision would require manually defining alignments such as:

```text
frames 1–5   → p
frames 6–8   → l
frames 9–14  → a
...
```

This would be extremely difficult and unnecessary.

CTC allows the network to learn the alignment automatically.

The model predicts character probabilities at every time step, including a special blank symbol.

A raw model prediction may conceptually look like:

```text
_ _ p p _ l l _ a a a _ c c e _
```

CTC then collapses repeated characters and removes blank tokens:

```text
place
```

This makes it possible to train directly from:

```text
video → sentence
```

without explicit frame-level character labels.

---

# Loss Function

Training uses the TensorFlow/Keras CTC implementation:

```text
ctc_batch_cost
```

The objective compares the predicted temporal probability sequence with the target transcription while accounting for all possible valid alignments.

This effectively turns the network into an end-to-end sequence recognizer rather than a frame classifier.

---

# Optimization

The network is trained using the Adam optimizer.

The initial learning rate is:

```text
1e-4
```

A custom learning-rate schedule keeps the learning rate stable during the initial training phase and gradually reduces it later.

The strategy used during the experiment was approximately:

```text
Epochs 0–50
    constant learning rate

Later training
    exponential decay at defined intervals
```

The goal was to allow relatively large parameter updates while the model learned the basic visual representation and more conservative updates once convergence improved.

---

# Training Strategy

The full training schedule was configured for:

```text
500 epochs
```

Rather than training continuously on exactly the same speaker subset, the training pipeline periodically changes the active speaker data.

A new speaker set is introduced approximately every:

```text
50 epochs
```

This was designed to improve robustness and reduce the risk of the model becoming overly specialized to the facial geometry and articulation style of individual speakers.

Speaker-independent generalization is one of the core challenges in visual speech recognition.

Different people have different:

* lip geometry
* facial proportions
* articulation patterns
* speaking speeds
* head movements

The network therefore has to learn visual speech patterns rather than simply memorize individual mouths.

---

# Training Monitoring

Several custom callbacks are included in the training process.

### Prediction monitoring

A custom `ProduceExample` callback periodically decodes predictions during training.

This makes it possible to observe whether the network is actually learning meaningful character sequences rather than relying exclusively on the numerical loss.

### Training history

A `SaveHistoryCallback` stores training metrics at regular intervals.

The repository includes the resulting training history for later inspection and analysis.

### Checkpointing

Model weights are periodically saved during training.

This allows:

* recovery after interrupted training
* comparison between different training stages
* continued fine-tuning
* testing earlier checkpoints if later epochs overfit

---

# Engineering Perspective

Although the neural architecture is the central component, the project is not only a model implementation.

The complete system required building several connected layers:

```text
raw dataset
    ↓
video decoding
    ↓
computer vision preprocessing
    ↓
facial landmark detection
    ↓
ROI normalization
    ↓
text alignment parsing
    ↓
character encoding
    ↓
TensorFlow data pipeline
    ↓
spatiotemporal neural network
    ↓
CTC training
    ↓
sequence decoding
    ↓
evaluation
```

This was one of the main goals of the project: understanding how a research idea becomes a complete working machine-learning pipeline.

---

# Project Structure

```text
lipreading-cv-nlp/
│
├── datapipeline.py
│   └── TensorFlow dataset construction and batching
│
├── lipreadingnn.py
│   └── Lipreading network implementation
│
├── main.py
│   └── Training / execution entry point
│
├── model.py
│   └── 3D-CNN + BiGRU architecture
│
├── preprocessing.py
│   └── Video, landmarks, lip extraction and text processing
│
├── training_history.json
│   └── Stored training metrics
│
├── ResearchPaper_BachelorThesis_Serbian.pdf
│   └── Bachelor thesis / research paper
│
└── README.md
```

---

# Technical Summary

| Component         | Configuration                  |
| ----------------- | ------------------------------ |
| Task              | Visual Speech Recognition      |
| Input modality    | Video only                     |
| Dataset           | GRID Corpus                    |
| Framework         | TensorFlow / Keras             |
| Computer Vision   | OpenCV + dlib                  |
| Input sequence    | 75 frames                      |
| Frame resolution  | 64 × 64                        |
| Channels          | 1 grayscale                    |
| Input shape       | 75 × 64 × 64 × 1               |
| Visual encoder    | 3D CNN                         |
| Conv filters      | 128 → 256 → 64                 |
| Conv kernels      | 3 × 3 × 3                      |
| Spatial pooling   | 1 × 2 × 2                      |
| Sequence model    | 2 × Bidirectional GRU          |
| GRU units         | 128                            |
| Vocabulary        | 41 classes                     |
| Output            | Character probability sequence |
| Output shape      | 75 × 41                        |
| Loss              | CTC                            |
| Optimizer         | Adam                           |
| Initial LR        | 1e-4                           |
| Training schedule | 500 epochs                     |
| Regularization    | Dropout                        |
| Monitoring        | Custom callbacks + checkpoints |

---

# Challenges

Visual speech recognition introduces several difficulties that are not present in conventional image classification.

## Viseme ambiguity

Different phonemes can produce almost identical lip movements.

For example, visually distinguishing sounds such as:

```text
/p/
/b/
/m/
```

is difficult because all three involve similar lip closure patterns.

This means that purely local visual information is often insufficient.

Temporal and linguistic context become essential.

---

## Speaker variability

The same word can look significantly different when spoken by different people.

Variability can originate from:

* facial structure
* lip size
* articulation
* speech rate
* camera position
* expression
* head movement

A useful lipreading model therefore needs to learn representations that remain stable across speakers.

---

## Temporal alignment

Video contains many more frames than the final transcription contains characters.

The correspondence between them is unknown.

CTC solves this by treating alignment as a latent variable rather than requiring frame-level annotation.

---

## Visual information is incomplete

Some speech information is fundamentally difficult or impossible to infer from lips alone.

Sounds generated by the tongue or vocal cords may produce little visible difference.

For this reason, visual speech recognition should not be interpreted as perfect reconstruction of arbitrary speech.

It is a probabilistic inference task.

---

# Potential Applications

Visual speech recognition can complement conventional audio speech recognition in several areas.

### Noisy environments

Audio recognition can degrade significantly in:

* factories
* traffic
* public spaces
* large events
* robotics environments

Visual speech information can provide an additional modality.

### Multimodal speech recognition

Visual features can be combined with audio features to build more robust speech recognition systems.

```text
Audio ───────┐
             ├── Multimodal ASR
Video ───────┘
```

When one modality becomes unreliable, the other can contribute additional information.

### Human-robot interaction

Robots equipped with cameras can potentially use visual speech recognition together with microphone-based ASR.

This can help improve interaction when:

* background noise is high
* the speaker is far from the microphone
* multiple people are speaking
* audio becomes temporarily unavailable

### Accessibility research

Visual speech recognition is also relevant to assistive communication systems and technologies for people with hearing impairments.

---

# Limitations

The current system is a research implementation and has several important limitations.

* It is trained on a constrained vocabulary and sentence grammar.
* GRID recordings are significantly more controlled than real-world video.
* Performance on unconstrained natural speech is expected to be considerably harder.
* The system assumes that the mouth region can be reliably detected.
* Large head rotations or occlusion can degrade preprocessing.
* Speaker-independent lipreading remains difficult.
* Visual information alone cannot uniquely determine all spoken phonemes.
* The current architecture is substantially smaller than modern large-scale visual speech models.

The model should therefore be considered a research prototype rather than a production speech-recognition system.

---

# Future Work

Several directions could extend the project.

### Transformer-based temporal modelling

The BiGRU sequence encoder could be compared with:

* Temporal Convolutional Networks
* Transformers
* Conformers
* visual speech transformer architectures

Self-attention could potentially capture longer-range relationships more effectively.

### Audio-visual fusion

A natural extension would combine the visual pipeline with audio ASR.

Instead of:

```text
video → text
```

the system could use:

```text
video ─┐
       ├── fusion model → text
audio ─┘
```

This would provide a more realistic multimodal speech-recognition architecture.

### Language modelling

CTC decoding could be combined with an external language model.

The visual network would provide character probabilities while the language model would help resolve ambiguous visual sequences.

### Real-time inference

The preprocessing and prediction pipeline could be adapted for live webcam input.

This would require:

* real-time face tracking
* rolling frame buffers
* streaming inference
* incremental decoding
* latency optimization

### Edge deployment

A smaller optimized version could also be deployed on embedded GPU hardware such as NVIDIA Jetson platforms.

---

# Research Context

The project was developed as part of my Bachelor thesis in Biomedical Engineering.

The repository contains the corresponding thesis document:

```text
ResearchPaper_BachelorThesis_Serbian.pdf
```

The work was intended to connect several areas that I was interested in:

```text
Computer Vision
      +
Deep Learning
      +
Sequence Modelling
      +
Natural Language Processing
      =
Visual Speech Recognition
```

Rather than using an existing end-to-end lipreading implementation, the project focuses on building and understanding the individual components of the system.

---

## Author

**Dragomir Božoki**

AI / Machine Learning Engineer

Bachelor thesis project in Biomedical Engineering.

---

## License

This project is distributed under the MIT License.
