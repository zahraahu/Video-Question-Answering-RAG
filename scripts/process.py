from utils.paths import VIDEO_PATH, AUDIO_PATH, KEYFRAMES_DIR, TRANSCRIPT_PATH, PROCESSED_DATA_PATH
from utils.utils import save_json
import os
import yt_dlp
import cv2
import whisper
from moviepy import VideoFileClip


def download_video(url, output_path=VIDEO_PATH):
    """
    Downloads a video from a given URL using yt-dlp.
    """
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def extract_keyframes(video_path, output_dir, interval=5):
    """
    Extracts keyframes from a video file at a fixed interval.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        raise ValueError("Failed to get FPS of the video.")
    count = 0
    saved = 0
    keyframes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % int(fps * interval) == 0:
            timestamp = count / fps
            filename = os.path.join(output_dir, f"frame_{timestamp:07.2f}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
            keyframes.append({
                "timestamp": timestamp,
                "frame_path": filename
            })
        count += 1

    cap.release()
    print(f"Saved {saved} keyframes to {output_dir}")
    return keyframes


def extract_audio_from_video(video_path, audio_output_path):
    """
    Extracts the audio from a video and saves it as a file.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_output_path)
    video.close()


def extract_transcript(audio_path, output_path):
    """
    Transcribes audio into text using OpenAI Whisper.
    """
    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))

    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    save_json(chunks, output_path)
    return chunks


def process_video(url):
    """
    Full pipeline to process the video: download, extract keyframes, and generate transcript.
    Also associates frames with transcript chunks.
    """
    # Step 1: Download the video if it doesn't exist
    if not os.path.exists(VIDEO_PATH):
        print("Downloading video...")
        download_video(url)
    else:
        print("Video already exists. Skipping download.")

    # Step 2: Extract keyframes
    print("Extracting keyframes...")
    extract_keyframes(VIDEO_PATH, KEYFRAMES_DIR, interval=5)

    # Step 3: Extract audio and generate transcript
    print("Extracting audio and generating transcript...")
    extract_audio_from_video(VIDEO_PATH, AUDIO_PATH)
    extract_transcript(AUDIO_PATH, TRANSCRIPT_PATH)

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=dARr3lGKwk8"  
    process_video(video_url)
