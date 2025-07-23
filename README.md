# Editor de Vídeo Automático con IA

Este script de Python automatiza varias tareas de edición de vídeo para crear contenido dinámico y atractivo a partir de un vídeo en bruto.

## Características

-   **Eliminación de Silencios:** Detecta y recorta automáticamente las pausas largas en el discurso para hacer el vídeo más conciso.
-   **Subtítulos Animados:** Transcribe el audio usando el modelo Whisper de OpenAI y genera subtítulos animados palabra por palabra o en pequeños grupos.
-   **B-Roll Inteligente:** Consulta a GPT-4o para identificar los temas clave del discurso y busca vídeos de stock relevantes en Pexels para insertarlos como B-roll.
-   **Música de Fondo:** Descarga música libre de derechos de autor de YouTube que se ajusta a una duración y género deseados, y la mezcla con el audio original a un volumen adecuado.
-   **Zoom Dinámico:** Aplica un efecto de zoom que alterna entre el plano original y un ligero acercamiento para mantener la atención del espectador.

## Requisitos

-   Python 3.8 o superior
-   FFmpeg (debe estar instalado y accesible en el PATH del sistema)
-   ImageMagick (recomendado para la generación de subtítulos con `moviepy`)
-   Una clave de API de [OpenAI](https://platform.openai.com/signup).
-   Una clave de API de [Pexels](https://www.pexels.com/api/).

## Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/tu-repositorio.git
    cd tu-repositorio
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configura tus claves de API:**
    Abre el archivo `main.py` y reemplaza `"TU_API_KEY_DE_OPENAI"` y `"TU_API_KEY_DE_PEXELS"` con tus claves reales.

    **(Recomendado)** Para mayor seguridad, configura tus claves como variables de entorno:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export PEXELS_API_KEY="..."
    ```
    Y modifica el script para leerlas con `os.getenv()`.

## Uso

Ejecuta el script desde la terminal, proporcionando la ruta a tu archivo de vídeo con el argumento `--video_path`.

```bash
python main.py --video_path "/ruta/completa/a/mi_video.mp4"
```

El script creará una carpeta llamada `edited_videos` en el directorio actual y guardará allí los archivos de vídeo intermedios y el resultado final, llamado `nombre-original_FINAL.mp4`.

## Parámetros Personalizables

Puedes ajustar el comportamiento del script modificando las variables globales en la sección `PARÁMETROS DE EDICIÓN` dentro de `main.py`. Esto incluye el umbral de silencio, el estilo de los subtítulos, el género de la música, el nivel de zoom y más.
