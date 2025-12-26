# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
import tempfile
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO

TARGET_FS = 16000


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"

# Device detection with fallback to CPU
def get_device():
    device_str = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available, falling back to CPU")
            return "cpu"
        try:
            # Test if CUDA device is actually usable
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return device_str
        except RuntimeError as e:
            print(f"WARNING: CUDA device error ({e}), falling back to CPU")
            return "cpu"
    return device_str

device = get_device()
print(f"Using device: {device}")
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    audios = []
    temp_files = []
    try:
        for file in files:
            # Save uploaded file to temporary file for torchaudio to load
            # torchaudio.load() cannot directly read from BytesIO for some formats (e.g., MP3)
            file_content = await file.read()
            file_io = BytesIO(file_content)
            
            # Create temporary file with appropriate extension
            file_ext = os.path.splitext(file.filename)[1] if file.filename else '.wav'
            if not file_ext:
                file_ext = '.wav'
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
                temp_files.append(tmp_file_path)
            
            # Load audio from temporary file
            data_or_path_or_list, audio_fs = torchaudio.load(tmp_file_path)

            # transform to target sample
            if audio_fs != TARGET_FS:
                resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
                data_or_path_or_list = resampler(data_or_path_or_list)

            data_or_path_or_list = data_or_path_or_list.mean(0)
            audios.append(data_or_path_or_list)
    finally:
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            except Exception:
                pass  # Ignore cleanup errors

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=TARGET_FS,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)
