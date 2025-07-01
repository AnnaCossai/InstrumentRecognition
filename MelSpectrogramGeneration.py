import librosa
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

NUM_MELS = 60
input_folder = "8_instruments_dataset"
all_items = os.listdir(input_folder)
audio_files = [item for item in all_items if item.endswith('.wav')]
print(audio_files[:5]) # Print the first 5 files to verify

# Define a save path
output_folder = "spectrograms" + "_" + str(NUM_MELS)
os.makedirs(output_folder, exist_ok=True)

for audio_file in tqdm(audio_files):
    # Load the audio file and its sampling rate
    scale, sr = librosa.load(os.path.join(input_folder, audio_file))

    # Calculate the mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, hop_length=512, n_mels=NUM_MELS)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # Create a plot of the mel-spectrogram for CNN without axes or legend
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.png")

    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

print("Finished processing all audio files.")
