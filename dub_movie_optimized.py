import os
import json
import uuid
import time
import whisper
import logging
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import moviepy as mp
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment
from pydub import AudioSegment
import librosa
import webrtcvad
import numpy as np
import os
from scipy.io.wavfile import write
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
from glob import glob
from itertools import cycle
import IPython.display as ipd
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
def split_audio_sound(file_audio_path, min_speech_duration=0.5, silence_thresh=-40, vad_mode=3):
    # Tải file audio
    y, sr = librosa.load(file_audio_path, sr=None)    
    # Chuyển đổi audio sang định dạng 16-bit PCM, 16000 Hz (yêu cầu của WebRTC VAD)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    y = (y * 32767).astype(np.int16)  # Chuyển sang 16-bit PCM
    
    # Lưu tạm file WAV để sử dụng với WebRTC VAD
    temp_wav = "temp_vad.wav"
    write(temp_wav, sr, y)
    
    # Khởi tạo WebRTC VAD
    vad = webrtcvad.Vad(vad_mode)
    
    # Đọc file WAV tạm
    audio = AudioSegment.from_wav(temp_wav)
    frame_duration_ms = 30  # Độ dài khung (ms)
    frame_duration_samples = int(sr * frame_duration_ms / 1000)
    
    # Nhận diện lời nói
    segments = []
    is_speech = False
    start_time = None
    
    for i in range(0, len(y) - frame_duration_samples, frame_duration_samples):
        frame = y[i:i + frame_duration_samples].tobytes()
        if len(frame) == frame_duration_samples * 2:  # Đảm bảo kích thước khung đúng
            if vad.is_speech(frame, sr):
                if not is_speech:
                    start_time = i / sr
                    is_speech = True
            else:
                if is_speech:
                    end_time = i / sr
                    duration = end_time - start_time
                    if duration >= min_speech_duration:
                        segments.append([float(f"{start_time:.2f}"), float(f"{end_time:.2f}")])
                    is_speech = False
    if is_speech:
        end_time = len(y) / sr
        duration = end_time - start_time
        if duration >= min_speech_duration:
            segments.append([float(f"{start_time:.2f}"), float(f"{end_time:.2f}")])
    
    # Xóa file tạm
    if os.path.exists(temp_wav):
        os.remove(temp_wav)    
    return segments         
def setup_directories(temp_dir="temp_segments"):
    """Tạo thư mục tạm nếu chưa tồn tại."""
    os.makedirs(temp_dir, exist_ok=True)

def process_segment_transcription(video_path, start_time, end_time, segment_index, source_language, temp_dir):
    """Bước 1: Trích xuất âm thanh, nhận diện giọng nói, lưu văn bản và tính WPM."""
    logging.info(f"Đang xử lý đoạn {segment_index + 1} ({start_time:.1f}s - {end_time:.1f}s)")
    setup_directories(temp_dir)
    
    segment_id = str(uuid.uuid4())
    segment_audio_path = os.path.join(temp_dir, f"audio_{segment_index}_{segment_id}.wav")
    output_segment_path = os.path.join(temp_dir, f"output_segment_{segment_index}.mp4").replace(os.sep, '\\')
    
    try:
        # Cắt đoạn video và trích xuất âm thanh
        video = mp.VideoFileClip(video_path).subclipped(start_time, end_time)
        video.audio.write_audiofile(segment_audio_path)
        if not os.path.exists(segment_audio_path):
            raise Exception(f"Không tạo được file âm thanh: {segment_audio_path}")
        logging.info(f"Đã trích xuất âm thanh cho đoạn {segment_index + 1} vào {segment_audio_path}")        
        # Chuyển giọng nói thành văn bản
        model=whisper.load_model("base")
        transcribe=model.transcribe(segment_audio_path,fp16=False,language=source_language)            
        try:
            text = transcribe['text']
            print (text)
            logging.info(f"Nhận diện văn bản đoạn {segment_index + 1}: {text}")
        except sr.UnknownValueError:
            text = ""
            logging.warning(f"Không nhận diện được giọng nói ở đoạn {segment_index + 1}")
        except sr.RequestError as e:
            text = ""
            logging.error(f"Lỗi API Speech-to-Text ở đoạn {segment_index + 1}: {e}")
        if text == "":
            return None
        else:
            metadata = {
                "index": segment_index,
                "start_time": start_time,
                "end_time": end_time,
                "output_path": output_segment_path,
                "text": text,
                "speed_of_speech": f"2007 WPM",
                "status": "completed"
            }
            return metadata
    
    except Exception as e:
        logging.error(f"Lỗi khi xử lý đoạn {segment_index + 1}: {e}")
        return {
            "index": segment_index,
            "start_time": start_time,
            "end_time": end_time,
            "output_path": output_segment_path,
            "text": "",
            "speed_of_speech": "0 WPM",
            "status": f"failed: {str(e)}"
        }
    
    finally:
        if os.path.exists(segment_audio_path):
            try:
                os.remove(segment_audio_path)
                logging.info(f"Đã xóa file tạm: {segment_audio_path}")
            except Exception as e:
                logging.warning(f"Không thể xóa file tạm {segment_audio_path}: {e}")
        if 'video' in locals():
            video.close()

def load_checkpoint(checkpoint_file):
    """Tải file checkpoint để kiểm tra trạng thái xử lý."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            filtered_data=[item for item in json.load(f)['segments'] if item is not None]
            result = {'segments': filtered_data}
            return result
    return {"segments": []}

def save_checkpoint(checkpoint_file, metadata_list):
    """Lưu trạng thái xử lý vào file checkpoint."""
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4, ensure_ascii=False)

def extract_audio_from_video(video_path, output_audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    return output_audio_path

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

def extract_transcript(input_video_path, output_dir, output_json_path, source_language):
    """Trích xuất nội dung giọng nói từ video và lưu vào file JSON."""
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
    output_path=str(output_dir)+str('/original_voice/main_stream.wav')
    audio_file = extract_audio_from_video(input_video_path,output_path)
    segments = split_audio_sound(output_path)
    
    segments_to_process = [(i, start, end) for i, (start, end) in enumerate(segments) if i not in completed_segments]
    
    metadata_list = checkpoint["segments"]
    max_workers = min(os.cpu_count() or 4, 4)  # Giới hạn số luồng để tránh quá tải API
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(process_segment_transcription, input_video_path, start, end, index, source_language, temp_dir): (index, start, end)
            for index, start, end in segments_to_process
        }
        for future in as_completed(future_to_segment):
            index, start, end = future_to_segment[future]
            try:
                metadata = future.result()
                metadata_list.append(metadata)
                save_checkpoint(checkpoint_file, {"segments": metadata_list})
                print(f"Hoàn thành đoạn {index + 1}/{len(segments)} ({start:.1f}s - {end:.1f}s)")
            except Exception as e:
                logging.error(f"Lỗi khi xử lý đoạn {index + 1}: {e}")
                print(f"Lỗi khi xử lý đoạn {index + 1}: {e}")
    
    output_json = {"segments": metadata_list}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)
    logging.info(f"Đã lưu transcript vào: {output_json_path}")
    print(f"Đã lưu transcript vào: {output_json_path}")
    
    return output_json

def translate_segment(segment, api_key, target_language):
    """Translate a single segment using the Gemini API."""
    global script_letter    
    try:
        if not segment["text"] or segment.get("status", "").startswith("failed"):
            return {**segment, "translated_text": ""}

        prompt = (
            f"""
            Bạn là một công cụ dịch thuật. Nhiệm vụ của bạn là:
            1. **Chỉ dịch đoạn văn sau sang TIẾNG VIỆT** (nằm trong biến `segment['text']`).
            2. **Không được giữ lại bất kỳ ký tự gốc nào của văn bản gốc (ví dụ tiếng Trung, tiếng Anh...) trong kết quả.**
            3. **Không dịch bất kỳ phần nào khác ngoài `segment['text']`**.

            Thông tin ngữ cảnh (chỉ để bạn hiểu nội dung, KHÔNG cần dịch phần này):
            {script_letter}

            Bây giờ, hãy DỊCH CHỈ đoạn sau:
            {segment['text']}

            Kết quả dịch (chỉ nội dung tiếng Việt tương ứng của đoạn trên, KHÔNG thêm gì khác):
            """
        )
        while True:
            try:
                client = genai.Client(api_key=api_key)
                translated_text = client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                ).text
                
                if segment['text'] !='' and translated_text !='':
                    script_letter=(script_letter)+str(segment['text'])
                    return {**segment, "translated_text": translated_text.strip()}
                elif segment['text'] =='' and translated_text =='':
                    return {**segment, "translated_text": ""}
                else:
                    None
            except Exception as e:
                logging.error(f"Lỗi khi dịch đoạn {segment['index']} với API key {api_key[:10]}...: {e}")
                break
    except:
        return None
def translate_with_gemini(api_keys, segments, target_language):
    """Dịch văn bản sang tiếng Việt bằng Google Gemini với 2 API key và 2 worker threads."""
    logging.info("Bắt đầu dịch văn bản bằng Google Gemini với 2 API key...")
    print("Bắt đầu dịch văn bản bằng Google Gemini với 2 API key...")

    if len(api_keys) != 2:
        raise ValueError("Cần đúng 2 API key để sử dụng 2 luồng.")

    translated_segments = []
    # Chia danh sách segments thành 2 phần cho 2 luồng
    mid = len(segments["segments"]) // 2
    segment_groups = [segments["segments"][:mid], segments["segments"][mid:]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Gửi nhiệm vụ dịch cho 2 luồng, mỗi luồng dùng 1 API key
        futures = [
            executor.submit(
                lambda segs, key: [translate_segment(seg, key, target_language) for seg in segs],
                segment_groups[i], api_keys[i]
                
            )
            for i in range(2)
        ]
        # Thu thập kết quả từ cả 2 luồng
        for future in concurrent.futures.as_completed(futures):
            translated_segments.extend(future.result())
            translated_segments=[item for item in translated_segments if item is not None]
    # Sắp xếp theo index để giữ thứ tự gốc
    translated_segments.sort(key=lambda x: x["index"])
    logging.info("Dịch văn bản hoàn tất.")
    print("Dịch văn bản hoàn tất.")
    print('translated_segments : \n' + str(translated_segments) + '\n')
    return translated_segments

def process_segment_dubbing(start_time, end_time, segment_index, translated_text, target_language, temp_dir="temp_segments"):
    """Bước 3: Tạo file TTS cho đoạn văn bản."""
    logging.info(f"Đang tạo TTS cho đoạn {segment_index + 1} ({start_time:.1f}s - {end_time:.1f}s)")
    setup_directories(temp_dir)    
    segment_id = str(uuid.uuid4())
    tts_path = os.path.join(temp_dir, f"tts_fist_edittion_{segment_index}.mp3")
    final_audio_path = os.path.join(temp_dir, f"tts_{segment_index}.mp3")
  
    try:
        if translated_text:
            tts = gTTS(text=translated_text, lang=target_language)
            tts.save(tts_path)
            # Rename or move to final output path if needed
            tts_audio = AudioSegment.from_mp3(tts_path)
            duration_ms = int((end_time - start_time) * 1000)
            if len(tts_audio) > duration_ms:
                speed_factor = len(tts_audio) / duration_ms
                speed_factor = min(speed_factor, 1.4)
                logging.info(f"Tăng tốc âm thanh với hệ số: {speed_factor:.2f}")
                adjusted_tts = tts_audio.speedup(playback_speed=speed_factor)
                adjusted_tts = adjusted_tts[:duration_ms]
                adjusted_tts.export(final_audio_path, format="mp3")
            elif len(tts_audio) < duration_ms:
                speed_factor = duration_ms / len(tts_audio)
                speed_factor = min(speed_factor, 1.4)
                logging.info(f"Giảm tốc độ âm thanh với hệ số: {speed_factor:.2f}")
                adjusted_tts = tts_audio.speedup(playback_speed=speed_factor)
                adjusted_tts = adjusted_tts + AudioSegment.silent(duration=duration_ms - len(adjusted_tts))
                adjusted_tts.export(final_audio_path, format="mp3")
            else:
                adjusted_tts = tts_audio
                logging.info("Thời lượng TTS khớp với thời lượng mục tiêu, không cần điều chỉnh.")
                adjusted_tts.export(final_audio_path, format="mp3")
            if not os.path.exists(final_audio_path):
                raise Exception(f"Không tạo được file âm thanh cuối: {final_audio_path}")
            logging.info(f"Đã tạo âm thanh lồng tiếng tiếng Việt cho đoạn {segment_index + 1} vào {final_audio_path}")
        else:
            # Nếu không có văn bản, tạo audio im lặng
            AudioSegment.silent(duration=int((end_time - start_time) * 1000)).export(final_audio_path, format="mp3")
            logging.info(f"Tạo audio im lặng cho đoạn {segment_index + 1} vào {final_audio_path}")    
        
        metadata = {
            "index": segment_index,
            "start_time": start_time,
            "end_time": end_time,
            "output_path": final_audio_path,
            "status": "completed"
        }
        return metadata
    
    except Exception as e:
        logging.error(f"Lỗi khi tạo TTS cho đoạn {segment_index + 1}: {e}")
        return {
            "index": segment_index,
            "start_time": start_time,
            "end_time": end_time,
            "output_path": final_audio_path,
            "status": f"failed: {str(e)}"
        }

def detect_langue(audio_path,model_AI='tiny'):
    model = whisper.load_model(model_AI)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Tạo log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Phát hiện ngôn ngữ
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)    
    return detected_lang

def cut_video_clip_30s(input_file,output_file,duration = 30):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-t", str(duration),
        "-c", "copy",
        output_file
    ]

    # Gọi ffmpeg qua subprocess
    try:
        subprocess.run(command, check=True)
        print("Cắt video thành công.")
    except subprocess.CalledProcessError as e:
        print("Có lỗi xảy ra:", e)
    

def dub_movie(input_video_path, output_dir, api_keys, source_language, target_language):
    """Chương trình chính: Dịch và tạo file TTS cho video."""
    temp_dir = "temp_segments"
    checkpoint_transcript_file = f"checkpoint_transcript_{os.path.basename(input_video_path)}.json"
    checkpoint_dub_file = f"checkpoint_dub_{os.path.basename(input_video_path)}.json"
    output_json_path = os.path.join(output_dir, "transcript.json").replace(os.sep, '\\')
    setup_directories(temp_dir)
    setup_directories(output_dir)
    
    print("Bước 1: Trích xuất nội dung giọng nói...")
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

if __name__ == "__main__":
    input_video = "C:/Videos/demo.mp4"
    output_dir = "C:/Projects/dubbed_movie"
    api_keys = ['AIzaSyDttABzgK2Nft55TKGsZK7a-6qPdRG9Dug', 'AIzaSyB1BLonp1Hr7TYDOwphxUPZDjMF8jLrS6s']
    analyze_file_path=output_dir+str('/analyze/video_analyze_languae.mp4')
    cut_video_clip_30s(input_video,analyze_file_path)
    source_language = detect_langue(analyze_file_path)
    script_letter=''
    target_language = "vi"
    
    dub_movie(input_video, output_dir, api_keys, source_language, target_language)
