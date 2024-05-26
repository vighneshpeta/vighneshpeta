import tensorflow as tf
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
import magenta.music as mm

def generate_beats():
    # Load the MusicVAE model
    config_name = 'cat-mel_2bar_small'
    config = configs.CONFIG_MAP[config_name]
    checkpoint = 'https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_small.tar'
    model = TrainedModel(config, batch_size=4, checkpoint_dir_or_path=checkpoint)

    # Generate a beat
    temperature = 1.0  # Controls the randomness of the generated beat
    num_steps = 32  # Length of the generated sequence
    z_size = config.hparams.z_size

    # Sample a latent vector and decode to a sequence
    z = tf.random.normal([4, z_size])
    sequences = model.sample(n=4, length=num_steps, temperature=temperature)

    # Convert to NoteSequence and save to MIDI file
    for i, ns in enumerate(sequences):
        mm.sequence_proto_to_midi_file(ns, f'generated_beat_{i}.mid')

    print("Beats generated and saved as MIDI files.")

if __name__ == "__main__":
    generate_beats()
