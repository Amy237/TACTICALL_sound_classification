# TACTICALL sound classification

This project classifies different home environment sounds using limited training data (around 3–5 recordings per sound), through transfer learning based on Google’s YAMNet model.

## Description

People with moderate to severe hearing loss may have difficulty receiving important notification sounds at home, such as doorbells or phone rings, especially when they are not wearing hearing aids. This project aims to develop a more comfortable and intuitive way to notify them, using a barely noticeable device that seamlessly blends into their daily life.

This code classifies different notification sounds using a small amount of training data (around 3 recordings per sound), enabling users to easily customize which sounds they want to recognize in their own home environment.  
By applying transfer learning with Google’s YAMNet model, the system achieves an average sound classification accuracy of around 85%.

## Getting Started

### Dependencies

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Hub
- Librosa
- Scikit-learn
- NumPy
- pydub (for optional audio conversion)
- OS: Windows 10 (tested)

Install all required packages via:
pip install -r requirements.txt

### Installing

Clone or download this repository:
git clone https://github.com/Amy237/TACTICALL_sound_classification.git
cd TACTICALL_sound_classification

Prepare your audio datasets:
Place your training .wav files into train_data/
Place your testing .wav files into test_data/

(Optional) Place sample audio for demo in example_audio/

(Optional) If your original recordings are .m4a format, use the script:
python src/convert_to_wav.py
to batch convert them into .wav files.

### Executing program

To train the model and predict sound, run:
python src/model_trained.py

This will:

Training phase:
Load training audio from train_data/
Extract embeddings using YAMNet
Train a Logistic Regression classifier
Save the model to model/yamnet_model.pkl

Prediction phase:
Load testing audio from test_data/
Predict each file’s label
Print the predicted label and confidence score

## Help

If you encounter issues with loading audio files:
Ensure your .wav files are single-channel (mono) and sampled at 16000 Hz.

If audio cannot be loaded, consider re-exporting it using FFmpeg:
ffmpeg -i input.m4a -ac 1 -ar 16000 output.wav

If TensorFlow Hub cannot load YAMNet:
Ensure that the models/yamnet_tensorflow/ folder exists and contains a valid TensorFlow SavedModel.

## Authors

Haofei Niu
haofei.design@gmail.com

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [MIT] License - see the LICENSE file for details

## Acknowledgments

* [Yamnet](https://www.kaggle.com/models/google/yamnet)
* [arjunsharma97](https://gist.github.com/arjunsharma97/0ecac61da2937ec52baf61af1aa1b759#file-m4atowav-py)
