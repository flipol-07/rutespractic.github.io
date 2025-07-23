# -*- coding: utf-8 -*-
"""
Script completo para la edición automática de vídeos.

Funcionalidades:
1.  Eliminación de silencios.
2.  Transcripción de audio y generación de subtítulos animados.
3.  Inserción de vídeos de stock (B-roll) basados en el contenido del discurso.
4.  Adición de música de fondo con volumen ajustado.
5.  Aplicación de un efecto de zoom dinámico para mayor engagement.

Para ejecutarlo:
python main.py --video_path "ruta/a/tu/video.mp4"
"""

import os
import argparse
import subprocess
import json
import random
import math
import re
import tempfile
import traceback
from typing import List, Dict

# Se asume que las librerías se instalan con: pip install -r requirements.txt
from moviepy.editor import (
    VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip,
    concatenate_videoclips, CompositeAudioClip, concatenate_audioclips
)
from moviepy.config import change_settings
import numpy as np
from openai import OpenAI
import requests

# --- CONFIGURACIÓN INICIAL Y VARIABLES GLOBALES ---

# Es una buena práctica configurar ImageMagick si se usa TextClip de forma intensiva.
# En un entorno local, asegúrate de que ImageMagick está instalado.
# Descomenta la siguiente línea si sabes la ruta del binario 'convert'.
# change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"}) # Ejemplo para Windows

# --- CLAVES DE API (IMPORTANTE) ---
# Se recomienda encarecidamente usar variables de entorno para las claves de API.
# Ejemplo: OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "TU_API_KEY_DE_OPENAI"
PEXELS_API_KEY = "TU_API_KEY_DE_PEXELS"

# --- PARÁMETROS DE EDICIÓN (AJUSTABLES) ---

# 1. Eliminación de silencios
SILENCE_THRESHOLD = 0.01
MIN_SILENCE_LEN = 0.6
KEEP_SILENCE_MARGIN = 0.2

# 2. Subtítulos y B-Roll
WORDS_PER_CHUNK_SUBTITLES = 3
PEXELS_CLIP_DURATION = 8
PEXELS_CLIP_SIZE_FACTOR = 0.45
PEXELS_CLIP_POSITION = ("right", "top")

# 3. Música de fondo
MUSIC_CATEGORY = "Cinematográfico / Inspirador" # Opciones: "Automático (Aleatorio)", "Corporativo / Profesional", etc.
TARGET_MUSIC_DURATION_MIN = 15
MUSIC_FLEXIBILITY_MIN = 10
MUSIC_VOLUME = 0.05

# 4. Zoom dinámico
ZOOM_LEVEL = 1.1
ZOOM_MIN_DURATION = 3.0
ZOOM_MAX_DURATION = 7.0


# --- FUNCIONES DE PROCESAMIENTO ---

def remove_silences(input_path: str, output_path: str):
    """Elimina los silencios de un vídeo y guarda el resultado."""
    print("--- Paso 1: Eliminando silencios ---")

    if not os.path.exists(input_path):
        print(f"Error: El archivo '{input_path}' no se ha encontrado.")
        return False

    video = VideoFileClip(input_path)
    if video.audio is None:
        print("Error: El vídeo no tiene pista de audio.")
        video.close()
        return False

    audio = video.audio
    duration = video.duration
    step = 0.05
    power = np.array([audio.subclip(t, min(t + step, duration)).max_volume() for t in np.arange(0, duration, step)])
    is_loud = power > SILENCE_THRESHOLD

    if np.sum(is_loud) == 0:
        print("ALERTA: No se ha detectado sonido por encima del umbral. Prueba a bajar SILENCE_THRESHOLD.")
        video.close()
        return False

    is_loud_indices = np.where(is_loud)[0]
    edges = np.diff(is_loud_indices) != 1
    split_points = np.where(edges)[0] + 1
    loud_segments_indices = np.split(is_loud_indices, split_points)

    chunks = [{'start': segment[0] * step, 'end': (segment[-1] + 1) * step}
              for segment in loud_segments_indices if len(segment) > 0]

    if not chunks:
        video.close()
        return False

    merged_chunks = []
    current_chunk = chunks[0]
    for next_chunk in chunks[1:]:
        if (next_chunk['start'] - current_chunk['end']) < MIN_SILENCE_LEN:
            current_chunk['end'] = next_chunk['end']
        else:
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
    merged_chunks.append(current_chunk)

    final_clip_times = [(max(0, chunk['start'] - KEEP_SILENCE_MARGIN),
                         min(duration, chunk['end'] + KEEP_SILENCE_MARGIN))
                        for chunk in merged_chunks]

    final_clips = [video.subclip(start, end) for start, end in final_clip_times]
    final_clip = concatenate_videoclips(final_clips)

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='medium', threads=os.cpu_count(), logger='bar')
    print(f"✅ Vídeo sin silencios guardado en: {output_path}")

    video.close()
    final_clip.close()
    return True


def add_subtitles_and_broll(input_path: str, output_path: str):
    """Añade subtítulos y vídeos de Pexels (B-roll) al vídeo."""
    print("\n--- Paso 2: Añadiendo Subtítulos y B-Roll ---")
    if not os.path.exists(input_path):
        print(f"Error: El archivo '{input_path}' no se encuentra.")
        return False

    # 1. Transcripción
    segments = _transcribe_audio(input_path)
    if not segments:
        print("Fallo en la transcripción. Abortando.")
        return False
    full_text = " ".join([seg['text'].strip() for seg in segments])

    # 2. Búsqueda y descarga de B-Roll
    video_clip_temp = VideoFileClip(input_path)
    video_duration = video_clip_temp.duration
    video_clip_temp.close()

    pexels_topics = _get_pexels_queries(full_text, video_duration)
    downloaded_pexels_videos = []
    if pexels_topics:
        downloaded_pexels_videos = _download_pexels_videos(pexels_topics)

    # 3. Creación de clips (subtítulos y pexels)
    video = VideoFileClip(input_path)
    w, h = video.w, video.h

    all_subtitle_clips = _create_subtitle_clips(segments, w, h)
    all_pexels_clips = _create_pexels_clips(downloaded_pexels_videos, w)

    # 4. Composición final
    print("Componiendo vídeo con subtítulos y B-roll...")
    final_video = CompositeVideoClip([video] + all_pexels_clips + all_subtitle_clips)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video.fps, threads=4, preset='medium', logger='bar')
    print(f"✅ Vídeo con subtítulos y B-roll guardado en: {output_path}")

    # Limpieza
    video.close()
    final_video.close()
    for clip in all_pexels_clips:
        clip.close()
    for pexels_vid in downloaded_pexels_videos:
        if os.path.exists(pexels_vid["local_path"]):
            os.remove(pexels_vid["local_path"])
    return True

# --- Funciones auxiliares para 'add_subtitles_and_broll' ---

def _transcribe_audio(video_path: str) -> List[Dict]:
    print("Transcribiendo audio con Whisper...")
    try:
        # Usamos un archivo temporal para el audio extraído
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio_file:
            audio_path = tmp_audio_file.name
        with VideoFileClip(video_path) as video_clip:
            audio = video_clip.audio
            audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        
        # Aquí se asume que 'transformers' está instalado.
        # En un entorno real, es mejor manejar la importación y la inicialización del pipeline de forma más robusta.
        from transformers import pipeline
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30)
        result = pipe(audio_path, return_timestamps=True, generate_kwargs={"language": "spanish"})
        print("✅ Transcripción completada.")
        return result.get('chunks', [])
    except Exception as e:
        print(f"Error en la transcripción: {e}")
        return []
    finally:
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.remove(audio_path)

def _get_pexels_queries(text: str, duration: float) -> List[Dict]:
    print("Consultando a ChatGPT para generar temas de búsqueda...")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Ets un expert en generar queries per Pexels. Respon en English. A partir d'un text, crea un query òptim per un vídeo que reflecteixi el concepte. Retorna una llista JSON amb 'query_pexels' i 'timestamp' (hh:mm:ss), no posis imatges més enlla de la duració del video ({duration:.2f} segons)."},
                {"role": "user", "content": text}
            ]
        )
        raw_content = response.choices[0].message.content
        clean_content = re.sub(r"```json|```", "", raw_content).strip()
        return json.loads(clean_content)
    except Exception as e:
        print(f"Error al consultar a ChatGPT: {e}")
        return []

def _download_pexels_videos(topics: List[Dict]) -> List[Dict]:
    print("Buscando y descargando vídeos de Pexels...")
    headers = {"Authorization": PEXELS_API_KEY}
    downloaded = []
    for i, topic in enumerate(topics):
        query = topic["query_pexels"]
        search_url = f"https://api.pexels.com/videos/search?query={query}&per_page=1"
        try:
            res = requests.get(search_url, headers=headers)
            res.raise_for_status()
            videos = res.json().get("videos", [])
            if videos:
                video_url = videos[0]['video_files'][0]['link']
                filename = f"temp_pexels_{i}.mp4"
                video_res = requests.get(video_url, stream=True)
                video_res.raise_for_status()
                with open(filename, 'wb') as f:
                    for chunk in video_res.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded.append({"timestamp": topic["timestamp"], "local_path": filename})
                print(f"✅ Descargado vídeo para '{query}'.")
        except requests.RequestException as e:
            print(f"Error en la solicitud a Pexels para '{query}': {e}")
    return downloaded

def _create_subtitle_clips(segments: List[Dict], w: int, h: int) -> List[TextClip]:
    all_clips = []
    print("Generando clips de subtítulos...")
    for segment in segments:
        text, timestamp = segment['text'].strip(), segment.get('timestamp')
        if not text or not timestamp: continue
        words = text.split()
        start_time, end_time = timestamp
        time_per_word = (end_time - start_time) / len(words) if words else 0
        for i in range(0, len(words), WORDS_PER_CHUNK_SUBTITLES):
            chunk_words = words[i:i + WORDS_PER_CHUNK_SUBTITLES]
            chunk_text = " ".join(chunk_words)
            chunk_start = start_time + (i * time_per_word)
            chunk_end = chunk_start + (len(chunk_words) * time_per_word)
            sub_clip = TextClip(
                chunk_text, font='Arial-Bold', fontsize=55, color='white',
                stroke_color='black', stroke_width=2.5, method='caption',
                size=(w * 0.85, None), align='center'
            ).set_start(chunk_start).set_duration(chunk_end - chunk_start)
            sub_clip = sub_clip.set_position(lambda t: ('center', h * 0.8 + 10 * math.sin((t - chunk_start) * math.pi / (chunk_end - chunk_start))))
            all_clips.append(sub_clip)
    return all_clips

def _create_pexels_clips(pexels_videos: List[Dict], w: int) -> List[VideoFileClip]:
    all_clips = []
    print("Cargando vídeos de Pexels para composición...")
    for pexels_video in pexels_videos:
        try:
            start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(pexels_video['timestamp'].split(':'))))
            temp_clip = VideoFileClip(pexels_video['local_path'])
            final_duration = min(temp_clip.duration, PEXELS_CLIP_DURATION)
            clip = (temp_clip.set_start(start_seconds)
                    .set_duration(final_duration)
                    .resize(width=w * PEXELS_CLIP_SIZE_FACTOR)
                    .set_position(PEXELS_CLIP_POSITION))
            all_clips.append(clip)
        except Exception as e:
            print(f"Warning: No se pudo procesar el clip {pexels_video['local_path']}. Error: {e}")
    return all_clips


def add_background_music(input_path: str, output_path: str):
    """Añade música de fondo a un vídeo."""
    print("\n--- Paso 3: Añadiendo Música de Fondo ---")
    
    # 1. Descargar música
    music_path = _download_music()
    if not music_path:
        print("No se pudo descargar música. Saltando este paso.")
        # Copiamos el archivo de entrada al de salida para que el pipeline continúe
        import shutil
        shutil.copy(input_path, output_path)
        return True

    # 2. Combinar vídeo y música
    video_clip = VideoFileClip(input_path)
    music_clip = AudioFileClip(music_path)
    video_duration = video_clip.duration
    music_duration = music_clip.duration

    if music_duration > video_duration:
        final_music_clip = music_clip.subclip(0, video_duration)
    else:
        num_repeats = math.ceil(video_duration / music_duration)
        looped_music = concatenate_audioclips([music_clip] * int(num_repeats))
        final_music_clip = looped_music.subclip(0, video_duration)

    final_music_clip = final_music_clip.volumex(MUSIC_VOLUME)

    if video_clip.audio:
        final_audio = CompositeAudioClip([video_clip.audio, final_music_clip])
    else:
        final_audio = final_music_clip

    video_with_music = video_clip.set_audio(final_audio)
    video_with_music.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video_clip.fps, preset='medium', threads=os.cpu_count(), logger='bar')
    print(f"✅ Vídeo con música guardado en: {output_path}")

    # Limpieza
    video_clip.close()
    music_clip.close()
    final_music_clip.close()
    final_audio.close()
    video_with_music.close()
    if os.path.exists(music_path):
        os.remove(music_path)
    return True

def _download_music() -> str:
    """Busca y descarga una pista de música usando yt-dlp."""
    print("Buscando y descargando música de fondo...")
    best_queries = {
        "Corporativo / Profesional": "motivational background", "Moderno / Tecnológico": "upbeat tech background",
        "Ambiente / Calmado": "calm ambient instrumental", "Cinematográfico / Inspirador": "inspirational cinematic background"
    }
    query = best_queries.get(MUSIC_CATEGORY, random.choice(list(best_queries.values())))
    
    min_dur = (TARGET_MUSIC_DURATION_MIN - MUSIC_FLEXIBILITY_MIN) * 60
    max_dur = (TARGET_MUSIC_DURATION_MIN + MUSIC_FLEXIBILITY_MIN) * 60

    try:
        search_command = ['yt-dlp', '--dump-json', f"ytsearch10:{query}"]
        process = subprocess.run(search_command, capture_output=True, text=True, check=True)
        videos = [json.loads(line) for line in process.stdout.strip().split('\n')]
        matching = [v for v in videos if v and 'duration' in v and min_dur <= v['duration'] <= max_dur]

        if not matching:
            print("No se encontraron vídeos que coincidan con los criterios de duración.")
            return ""

        selected_video = random.choice(matching)
        video_url = selected_video['webpage_url']
        
        output_template = "temp_music.%(ext)s"
        download_command = ['yt-dlp', '--extract-audio', '--audio-format', 'mp3', '-o', output_template, video_url]
        subprocess.run(download_command, check=True, capture_output=True)

        # Encontrar el archivo descargado
        for f in os.listdir('.'):
            if f.startswith("temp_music") and f.endswith(".mp3"):
                print(f"✅ Música descargada como: {f}")
                return f
        return ""
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error al descargar música con yt-dlp. Asegúrate de que está instalado y en el PATH. Error: {e}")
        return ""


def apply_dynamic_zoom(input_path: str, output_path: str):
    """Aplica un zoom dinámico que alterna entre planos."""
    print("\n--- Paso 4: Aplicando Zoom Dinámico ---")
    if not os.path.exists(input_path):
        print(f"Error: El archivo '{input_path}' no se encuentra.")
        return False

    video = VideoFileClip(input_path)
    clips_procesados = []
    tiempo_actual = 0
    plano_con_zoom = False

    while tiempo_actual < video.duration:
        duracion_segmento = random.uniform(ZOOM_MIN_DURATION, ZOOM_MAX_DURATION)
        fin_segmento = min(tiempo_actual + duracion_segmento, video.duration)
        subclip = video.subclip(tiempo_actual, fin_segmento)

        if plano_con_zoom:
            w, h = subclip.size
            new_w, new_h = w / ZOOM_LEVEL, h / ZOOM_LEVEL
            zoomed_clip = subclip.fx(vfx.crop, width=new_w, height=new_h, x_center=w/2, y_center=h/2).resize((w, h))
            clips_procesados.append(zoomed_clip)
        else:
            clips_procesados.append(subclip)

        plano_con_zoom = not plano_con_zoom
        tiempo_actual = fin_segmento

    final_video = concatenate_videoclips(clips_procesados)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='medium', threads=os.cpu_count(), logger='bar')
    print(f"✅ Vídeo con zoom dinámico guardado en: {output_path}")

    video.close()
    final_video.close()
    return True


def main():
    """Función principal que orquesta todo el proceso de edición."""
    parser = argparse.ArgumentParser(description="Script de edición de vídeo automática.")
    parser.add_argument("--video_path", type=str, required=True, help="Ruta al archivo de vídeo original.")
    args = parser.parse_args()

    # Validar que las API keys están presentes
    if "TU_API_KEY" in OPENAI_API_KEY or "TU_API_KEY" in PEXELS_API_KEY:
        print("Error: Por favor, introduce tus claves de API de OpenAI y Pexels en el script.")
        return

    base_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_dir = "edited_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Definir las rutas de los archivos intermedios y finales
    path_silences_removed = os.path.join(output_dir, f"{base_name}_1_silences_removed.mp4")
    path_subs_broll = os.path.join(output_dir, f"{base_name}_2_subs_broll.mp4")
    path_with_music = os.path.join(output_dir, f"{base_name}_3_with_music.mp4")
    final_path = os.path.join(output_dir, f"{base_name}_FINAL.mp4")

    try:
        # --- Ejecutar la cadena de procesamiento ---
        if remove_silences(args.video_path, path_silences_removed):
            if add_subtitles_and_broll(path_silences_removed, path_subs_broll):
                if add_background_music(path_subs_broll, path_with_music):
                    if apply_dynamic_zoom(path_with_music, final_path):
                        print(f"\n✨ ¡PROCESO COMPLETADO! ✨")
                        print(f"El vídeo final se ha guardado en: {final_path}")
    except Exception as e:
        print(f"\n🔴 Ha ocurrido un error fatal durante el proceso: {e}")
        traceback.print_exc()
    finally:
        # Opcional: limpiar archivos intermedios
        print("\nLimpiando archivos intermedios...")
        for path in [path_silences_removed, path_subs_broll, path_with_music]:
            if os.path.exists(path):
                os.remove(path)
        print("Limpieza finalizada.")


if __name__ == "__main__":
    main()
