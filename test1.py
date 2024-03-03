import torch
from one_peace.models import from_pretrained

device = "cuda" if torch.cuda.is_available() else "cpu"
# "ONE-PEACE" can also be replaced with ckpt path
model = from_pretrained("ONE-PEACE", device=device, dtype="float32")

# process raw data
src_tokens = model.process_text(["cow", "dog", "elephant"])
src_images = model.process_image(["assets/dog.JPEG", "assets/elephant.JPEG"])
src_audios, audio_padding_masks = model.process_audio(["assets/cow.flac", "assets/dog.flac"])

with torch.no_grad():
    # extract normalized features
    text_features = model.extract_text_features(src_tokens)
    image_features = model.extract_image_features(src_images)
    audio_features = model.extract_audio_features(src_audios, audio_padding_masks)

    # compute similarity
    i2t_similarity = image_features @ text_features.T
    a2t_similarity = audio_features @ text_features.T

print("Image-to-text similarities:", i2t_similarity)
print("Audio-to-text similarities:", a2t_similarity)
