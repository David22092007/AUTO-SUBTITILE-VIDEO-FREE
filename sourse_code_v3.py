import os
import json
import uuid
import time
import logging
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import whisper
from TTS.api import TTS
import moviepy as mp
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import librosa
import webrtcvad
import numpy as np
from scipy.io.wavfile import write
import soundfile as sf
import requests
import re
from google import genai

logging.basicConfig(filename='C:/Projects/dub_movie.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def median(danh_sach):
    danh_sach_sap_xep = sorted(danh_sach)
    n = len(danh_sach_sap_xep)
    if n == 0:
        return None
    if n % 2 == 1:
        return danh_sach_sap_xep[n // 2]
    else:
        giua1 = danh_sach_sap_xep[n // 2 - 1]
        giua2 = danh_sach_sap_xep[n // 2]
        return (giua1 + giua2) / 2

def group_consecutive_times(times, threshold_ms=50):
    times.sort()
    groups = []
    current_group = [times[0]]
    for t in times[1:]:
        if t - current_group[-1] <= threshold_ms:
            current_group.append(t)
        else:
            groups.append(current_group)
            current_group = [t]
    groups.append(current_group)
    return groups

def split_continuous(lst):
    result = []
    current = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current.append(lst[i])
        else:
            result.append(current)
            current = [lst[i]]
    result.append(current)
    return result
def setup_directories(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

def extract_audio_ffmpeg(video_path, start_time, end_time, output_audio_path):
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-to", str(end_time),
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            filtered_data = [item for item in json.load(f)['segments'] if item is not None]
            return {'segments': filtered_data}
    return {"segments": []}

def save_checkpoint(checkpoint_file, metadata_list):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

def extract_audio_from_video(video_path,temp_dir):
    output_audio_paths=split_video(video_path, SEGMENT_DURATION=30,temp_dir)
    return output_audio_paths

def filter_audio(audio_files_path):
    y, sr = librosa.load(audio_files_path)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
        aggregate=np.median,
        metric='cosine',
        width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    mask_vocals = S_full - S_filter
    mask_background = S_filter
    S_foreground = mask_vocals * phase
    S_background = mask_background * phase
    y_vocals = librosa.istft(S_foreground)
    y_background = librosa.istft(S_background)
    sf.write('output/vocals.wav', y_vocals, sr)
    sf.write('output/background.wav', y_background, sr)
def srt_time_to_seconds(srt_time: str) -> float:
    # Tách chuỗi thành giờ, phút, giây, mili giây
    try:
        time_part, ms_part = srt_time.split(',')
        hours, minutes, seconds = map(int, time_part.split(':'))
        milliseconds = int(ms_part)

        # Tính tổng thời gian thành giây
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        return total_seconds
    except ValueError:
        raise ValueError("Định dạng thời gian không hợp lệ. Phải là HH:MM:SS,mmm")

def transcript_video(audio_file_path,temp_dir,source_language,segment_index=0,MODEL='base'):
    global metadata_list
    setup_directories(temp_dir)
    segment_id = str(uuid.uuid4())    
    try:
        for audio in audio_file_path:
            print(f"🧠 Đang xử lý: {audio}")
            cmd = (
                f"whisper {audio} --model {MODEL} --output_format srt "
                f"--language {source_language} --fp16 False --output_dir  {temp_dir}"
                
            )
            os.system(cmd)
            while i < len(lines):
                i += 1
                time_line = lines[i].strip().split(' --> ')
                start_time, end_time = time_line[0], time_line[1]
                i += 1
                text = lines[i].strip()
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit():
                    text += ' ' + lines[i].strip()
                    i += 1
                metadata = {
                    "start_time": srt_time_to_seconds(start_time),
                    "end_time": str(end_time),
                    "text": text,
                    "speed_of_speech": "2007 WPM",
                    "status": "completed"
                }
                metadata_list.append(metadata)
        return metadata_list
    except Exception as e:
        logging.error(f"Lỗi khi xử lý đoạn : {e}")
    finally:
        if os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
                logging.info(f"Đã xóa file tạm: {audio_file_path}")
            except Exception as e:
                logging.warning(f"Không thể xóa file tạm {audio_file_path}: {e}")
        if 'video' in locals():
            video.close()
def get_video_duration(video_file):
    import ffmpeg
    probe = ffmpeg.probe(video_file)
    return float(probe['format']['duration'])

# 2. Cắt video thành nhiều đoạn
def split_video(video_file, segment_duration,temp_dir):
    total_duration = get_video_duration(video_file)
    segments = []
    for i in range(0, int(total_duration), segment_duration):
        out_file = f'{temp_dir}/segment_{i}.aac'
        cmd = (
            f"ffmpeg -i {video_file} -ss {i} -t {segment_duration} "
            f"-vn -acodec copy {out_file} -y"
        )
        os.system(cmd)
        segments.append(out_file)
    return segments

# 3. Luồng xử lý Whisper
def transcribe_worker(audio_files):
    for audio in audio_files:
        print(f"🧠 Đang xử lý: {audio}")
        cmd = (
            f"whisper {audio} --model {MODEL} --output_format srt "
            f"--language {LANGUAGE} --fp16 False"
        )
        os.system(cmd)

# 4. Chia file cho các luồng
def run_threads(audio_files, num_threads=2):
    chunks = [audio_files[i::num_threads] for i in range(num_threads)]
    threads = []

    for chunk in chunks:
        t = threading.Thread(target=transcribe_worker, args=(chunk,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
def extract_transcript(input_video_path, output_dir, output_json_path, source_language):
    temp_dir = "temp_segments"
    checkpoint_file = f"checkpoint_transcript_{os.path.basename(input_video_path)}.json"
    setup_directories(temp_dir)
    logging.info(f"Bắt đầu xử lý video: {input_video_path}")
    checkpoint = load_checkpoint(checkpoint_file)
    completed_segments = {m["index"]: m for m in checkpoint["segments"] if m["status"] == "completed"}
    try:
        video = mp.VideoFileClip(input_video_path)
        duration = video.duration
        video.close()
        logging.info(f"Thời lượng video: {duration:.1f} giây")
    except Exception as e:
        logging.error(f"Lỗi khi tải video: {e}")
        print(f"Lỗi khi tải video: {e}")
        return None
    setup_directories(os.path.join(output_dir, 'original_voice'))
    output_path = os.path.join(output_dir, 'original_voice', 'main_stream.wav')
    temp_dir="temp_segments"
    metadata_list=[]
    try:
        audio_files_path = extract_audio_from_video(input_video_path, temp_dir)        
        metadata_list=transcript_video(audio_files_path,temp_dir,source_language)
        save_checkpoint(checkpoint_file, {"segments": metadata_list})
        print(f"Hoàn thành Chuyển Văn bản Thành Dọng Nói")
        output_json = {"segments": metadata_list}
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, indent=4, ensure_ascii=False)
        logging.info(f"Đã lưu transcript vào: {output_json_path}")
        print(f"Đã lưu transcript vào: {output_json_path}")
        return output_json
    except Exception as e:
        logging.error(f"Lỗi khi xử lý  : {e}")
        print(f"Lỗi khi xử lý  : {e}")


def translate_segment(segment, api_key, target_language):
    global script_letter
    try:
        if not segment["text"] or segment.get("status", "").startswith("failed"):
            return {**segment, "translated_text": ""}
        prompt = (
            f"""
            -Bạn là một công cụ dịch thuật. Nhiệm vụ của bạn là:
            -1. **Không được giữ lại bất kỳ ký tự gốc nào của văn bản gốc (ví dụ tiếng Trung, tiếng Anh...) trong kết quả.**
            -2. **Không dịch bất kỳ phần nào khác ngoài `segment['text']`**.

            -Thông tin ngữ cảnh (chỉ để bạn hiểu nội dung, KHÔNG cần dịch phần này):
            {script_letter}

            -Bây giờ, hãy DỊCH CHỈ đoạn sau:
            {segment['text']}
            
            -LƯU Ý. **Có số chổ là do cách phát âm & những từ đồng nghĩa làm cho khi dịch câu trở nên sai các phương pháp về từ như sai logic ,sai ngữ pháp , sai cách dùng từ .Bạn hãy xem và chỉnh sữa những điểm đó lại bằng trí tuệ của bạn.

            Kết quả dịch (chỉ nội dung tiếng Việt tương ứng của đoạn trên, KHÔNG thêm gì khác):
            """
        )
        client = genai.Client(api_key=api_key)
        translated_text = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        ).text
        if segment['text'] != '' and translated_text != '':
            script_letter = (script_letter) + str(segment['text'])
            return {**segment, "translated_text": translated_text.strip()}
        elif segment['text'] == '' and translated_text == '':
            return {**segment, "translated_text": ""}
    except Exception as e:
        logging.error(f"Lỗi khi dịch đoạn {segment['index']} với API key {api_key[:10]}...: {e}")
        return None

def translate_with_gemini(api_keys, segments, target_language):
    logging.info("Bắt đầu dịch văn bản bằng Google Gemini với 2 API key...")
    print("Bắt đầu dịch văn bản bằng Google Gemini với 2 API key...")
    if len(api_keys) != 2:
        raise ValueError("Cần đúng 2 API key để sử dụng 2 luồng.")
    translated_segments = []
    mid = len(segments["segments"]) // 2
    segment_groups = [segments["segments"][:mid], segments["segments"][mid:]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                lambda segs, key: [translate_segment(seg, key, target_language) for seg in segs],
                segment_groups[i], api_keys[i]
            )
            for i in range(2)
        ]
        for future in as_completed(futures):
            translated_segments.extend(future.result())
            translated_segments = [item for item in translated_segments if item is not None]
    translated_segments.sort(key=lambda x: x["index"])
    logging.info("Dịch văn bản hoàn tất.")
    print("Dịch văn bản hoàn tất.")
    return translated_segments

def process_segment_dubbing(start_time, end_time, segment_index, translated_text, target_language, temp_dir="temp_segments"):
    os.makedirs(temp_dir, exist_ok=True)
    duration_ms = int((end_time - start_time) * 1000)
    final_audio_path = os.path.join(temp_dir, f"tts_{segment_index}.wav")
    try:
        if translated_text.strip():
            tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
            wav = tts.tts(translated_text, lang=target_language)
            temp_wav_path = os.path.join(temp_dir, f"tts_temp_{segment_index}.wav")
            tts.save_wav(wav, temp_wav_path)
            tts_audio = AudioSegment.from_wav(temp_wav_path)
            os.remove(temp_wav_path)
            tts_len = len(tts_audio)
            if abs(tts_len - duration_ms) < duration_ms * 0.1:
                if tts_len > duration_ms:
                    tts_audio = tts_audio[:duration_ms]
                else:
                    silence = AudioSegment.silent(duration=duration_ms - tts_len)
                    tts_audio += silence
            elif tts_len > duration_ms:
                tts_audio = tts_audio[:duration_ms]
            elif tts_len < duration_ms:
                silence = AudioSegment.silent(duration=duration_ms - tts_len)
                tts_audio += silence
            tts_audio.export(final_audio_path, format="wav")
        else:
            AudioSegment.silent(duration=duration_ms).export(final_audio_path, format="wav")
        return {
            "index": segment_index,
            "start_time": start_time,
            "end_time": end_time,
            "output_path": final_audio_path,
            "status": "completed"
        }
    except Exception as e:
        return {
            "index": segment_index,
            "start_time": start_time,
            "end_time": end_time,
            "output_path": final_audio_path,
            "status": f"failed: {str(e)}"
        }

def detect_langue(audio_path, model_AI='tiny'):
    model = whisper.load_model(model_AI)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    return detected_lang

def cut_video_clip_30s(input_file, output_file, duration=30):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-t", str(duration),
        "-c", "copy",
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        print("Cắt video thành công.")
    except subprocess.CalledProcessError as e:
        print("Có lỗi xảy ra:", e)

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
def check_volume_of_file(file_audio_path):
    if os.path.exists(file_audio_path):
        file_size = os.path.getsize(file_audio_path)
        if int(file_size) >=2048:
            return False
        else:
            return True
    else:
        return True
def dub_movie(input_video_path, output_dir, api_keys, source_language, target_language):
    temp_dir = "temp_segments"
    checkpoint_transcript_file = f"checkpoint_transcript_{os.path.basename(input_video_path)}.json"
    checkpoint_dub_file = f"checkpoint_dub_{os.path.basename(input_video_path)}.json"
    output_json_path = os.path.join(output_dir, "transcript.json").replace(os.sep, '\\')
    setup_directories(temp_dir)
    setup_directories(output_dir)
    
    print("Bước 1: Trích xuất nội dung giọng nói...")
    if check_volume_of_file(output_json_path):
        transcript = extract_transcript(input_video_path, output_dir, output_json_path, source_language)        
        if not transcript:
            logging.error("Không thể trích xuất transcript. Kết thúc chương trình.")
            print("Không thể trích xuất transcript. Kết thúc chương trình.")
            return    
    print("Bước 2: Dịch văn bản sang tiếng Việt...")
    translated_segments = translate_with_gemini(api_keys, transcript, target_language)
    translated_json_path = os.path.join(output_dir, "translated_transcript.json").replace(os.sep, '\\')
    with open(translated_json_path, 'w', encoding='utf-8') as f:
        json.dump({"segments": translated_segments}, f, indent=4, ensure_ascii=False)
    logging.info(f"Đã lưu transcript dịch vào: {translated_json_path}")
    print(f"Đã lưu transcript dịch vào: {translated_json_path}")
    print("Bước 3: Tạo file TTS cho các đoạn...")
    checkpoint = load_checkpoint(checkpoint_dub_file)
    completed_segments = {m["index"]: m for m in checkpoint["segments"] if m["status"] == "completed"}
    metadata_list = checkpoint["segments"]
    max_workers = min(os.cpu_count() or 2, 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(process_segment_dubbing, segment["start_time"], segment["end_time"],
                           segment["index"], segment.get("translated_text", ""), target_language, temp_dir): segment
            for segment in translated_segments if segment["index"] not in completed_segments
        }
        for future in as_completed(future_to_segment):
            segment = future_to_segment[future]
            try:
                metadata = future.result()
                metadata_list.append(metadata)
                save_checkpoint(checkpoint_dub_file, {"segments": metadata_list})
                print(f"Hoàn thành tạo TTS đoạn {segment['index'] + 1}/{len(translated_segments)} "
                      f"({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
            except Exception as e:
                logging.error(f"Lỗi khi tạo TTS đoạn {segment['index'] + 1}: {e}")
                print(f"Lỗi khi tạo TTS đoạn {segment['index'] + 1}: {e}")
    output_segments = [m["output_path"] for m in metadata_list if m["status"] == "completed" and os.path.exists(m["output_path"])]
    logging.info(f"Hoàn thành {len(output_segments)}/{len(translated_segments)} file TTS.")
    print(f"Hoàn thành {len(output_segments)}/{len(translated_segments)} file TTS.")
    remove_file(checkpoint_dub_file)
    remove_file(checkpoint_transcript_file)

def extract_ids(url):
    pattern = r"space\.bilibili\.com/(\d+)/lists/(\d+)"
    match = re.search(pattern, url)
    if match:
        user_id = match.group(1)
        playlist_id = match.group(2)
        return user_id, playlist_id
    return None, None

if __name__ == "__main__":
    input_video = "C:/Videos/demo.mp4"
    output_dir = "C:/Projects/dubbed_movie"
    api_keys = ['AIzaSyDttABzgK2Nft55TKGsZK7a-6qPdRG9Dug', 'AIzaSyB1BLonp1Hr7TYDOwphxUPZDjMF8jLrS6s']
    setup_directories(os.path.join(output_dir, 'analyze'))
    analyze_file_path = os.path.join(output_dir, 'analyze', 'video_analyze_languae.mp4')
    cut_video_clip_30s(input_video, analyze_file_path)
    source_language = detect_langue(analyze_file_path)
    script_letter = ''
    target_language = "vi"
    dub_movie(input_video, output_dir, api_keys, source_language, target_language)
