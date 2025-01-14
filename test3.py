import torch
import json
from one_peace.models import from_pretrained

id2label = json.load(open("assets/vggsound_id2label.json"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
  "ONE-PEACE_VGGSound",
    model_type="one_peace_classify",
    device=device,
    dtype="float32"
)

# process audio
audio_list = ["assets/cow.flac", "assets/dog.flac"]
src_audios, audio_padding_masks = model.process_audio(audio_list)

with torch.no_grad():
    # extract audio features
    audio_logits = model.extract_audio_features(src_audios, audio_padding_masks)
    print(audio_logits.size())
    predict_label_ids = audio_logits.argmax(1).cpu().tolist()

for audio, predict_label_id in zip(audio_list, predict_label_ids):
    predict_label = id2label[str(predict_label_id)]
    print('audio: {}, predict label: {}'.format(audio, predict_label))
