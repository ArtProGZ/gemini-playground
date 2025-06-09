import mysql.connector
import requests
import paramiko
from pydub import AudioSegment
from io import BytesIO
import os
import time
import logging
import base64 # For completeness, though expecting raw audio from proxy

# --- Configuration Variables ---

# --- Deployment (ECS Server) ---
ECS_HOST = "8.138.192.253"
ECS_USER = "root"
ECS_KEY_PATH = "/Users/ufodisk/Desktop/files/ieltsspeaking.pem"  # Ensure this path is correct and accessible
REMOTE_BASE_DIR = "/var/www/myvideos/vocabulary_audio/"  # Subdirectory for vocabulary audio

# --- Database (MySQL) ---
DB_HOST = "8.138.192.253"
DB_USER = "rednotes_user"
DB_PASSWORD = "Csw123121!"
DB_DATABASE = "ieltsspeakingapp"
DB_CONNECTION_TIMEOUT = 30

# --- Proxy and API Settings ---
PROXY_BASE_URL = "https://lezhi2.deno.dev/"
# This is the API key for YOUR Deno proxy, not your direct Google Gemini API Key
PROXY_API_KEY = "AIzaSyATBTBDB7YP20BQfLj9eum9aMjkJN4bkeA"

# --- Audio Generation Settings ---
TTS_VOICE_NAME = "Kore"  # Default voice, can be overridden if needed
AUDIO_OUTPUT_FORMAT = "wav"  # User requested WAV
AUDIO_SAMPLE_RATE = 24000  # Expected from Gemini TTS (audio/L16;codec=pcm;rate=24000)
AUDIO_SAMPLE_WIDTH = 2     # For 16-bit PCM
AUDIO_CHANNELS = 1         # Mono
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts" # Target Gemini TTS model via proxy

# --- Database Queries ---
# Fetch terms that don't have both normal and slow audio URLs yet.
# Added a LIMIT for initial testing - REMOVE/ADJUST for full runs.
SELECT_TERMS_QUERY = """
    SELECT id, term, audio_url, audio_url_slow
    FROM vocabulary_terms
    WHERE audio_url IS NULL OR audio_url_slow IS NULL
    ORDER BY id
    LIMIT 5;
"""
UPDATE_NORMAL_AUDIO_QUERY = "UPDATE vocabulary_terms SET audio_url = %s, voice_id = %s WHERE id = %s;"
UPDATE_SLOW_AUDIO_QUERY = "UPDATE vocabulary_terms SET audio_url_slow = %s WHERE id = %s;" # voice_id set with normal

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_generation.log"),
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---

def connect_db():
    """Establishes and returns a MySQL connection."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_DATABASE, connection_timeout=DB_CONNECTION_TIMEOUT
        )
        logging.info(f"Successfully connected to database: {DB_DATABASE} on {DB_HOST}")
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to database: {err}")
        return None

def fetch_terms(db_conn, query):
    """Fetches vocabulary terms from the database."""
    if not db_conn or not db_conn.is_connected():
        logging.error("Database connection is not active.")
        return []
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute(query)
        terms = cursor.fetchall()
        logging.info(f"Fetched {len(terms)} terms to process.")
        return terms
    except mysql.connector.Error as err:
        logging.error(f"Error fetching terms: {err}")
        return []
    finally:
        cursor.close()

def generate_tts_via_proxy(prompt_text, tts_model_name, voice_name):
    """
    Sends a request to the Deno proxy to generate TTS audio.
    Returns raw audio bytes or None on failure.
    Assumes proxy is modified as per docs/modify_proxy_for_tts.md
    """
    tts_endpoint = f"{PROXY_BASE_URL.rstrip('/')}/v1/chat/completions"
    
    payload = {
        "model": tts_model_name,
        "input_text": prompt_text, # Custom field for proxy to pick up
        "tts_settings": {          # Custom field for proxy
            "voice": voice_name
        }
    }
    headers = {
        "Authorization": f"Bearer {PROXY_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        logging.info(f"Sending TTS request to proxy: {tts_endpoint} for model: {tts_model_name}, voice: {voice_name}, prompt: '{prompt_text[:50]}...'")
        response = requests.post(tts_endpoint, json=payload, headers=headers, timeout=90) # Increased timeout
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'audio/' in content_type:
            logging.info(f"Received raw audio data from proxy. Content-Type: {content_type}")
            # Extract sample rate from content_type if possible, e.g., 'audio/L16;codec=pcm;rate=24000'
            rate_from_header = AUDIO_SAMPLE_RATE # default
            if 'rate=' in content_type:
                try:
                    rate_from_header = int(content_type.split('rate=')[-1].split(';')[0])
                    logging.info(f"Extracted sample rate from header: {rate_from_header}")
                except ValueError:
                    logging.warning(f"Could not parse sample rate from Content-Type: {content_type}. Using default: {AUDIO_SAMPLE_RATE}")
            
            return response.content, rate_from_header
        else:
            logging.error(f"Proxy response was not direct audio. Status: {response.status_code}. Content-Type: {content_type}. Response: {response.text[:200]}")
            return None, None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling TTS proxy: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Proxy error response: {e.response.text[:500]}")
        return None, None

def save_audio_to_file(audio_bytes, local_filepath, input_sample_rate, output_format="wav"):
    """Saves raw PCM audio bytes to a WAV file using pydub."""
    if not audio_bytes:
        logging.error("No audio bytes to save.")
        return False
    try:
        audio_segment = AudioSegment.from_raw(
            BytesIO(audio_bytes),
            sample_width=AUDIO_SAMPLE_WIDTH, # 16-bit
            frame_rate=input_sample_rate,   # Use rate from header or default
            channels=AUDIO_CHANNELS         # Mono
        )
        logging.info(f"Saving audio to {output_format} at {local_filepath} with sample rate {input_sample_rate}Hz")
        audio_segment.export(local_filepath, format=output_format)
        return True
    except Exception as e:
        logging.error(f"Error saving audio to {local_filepath}: {e}. Ensure pydub is correctly installed.")
        return False

def connect_sftp():
    """Establishes and returns an SFTP client and SSH client connection to ECS."""
    try:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        expanded_key_path = os.path.expanduser(ECS_KEY_PATH)
        if not os.path.exists(expanded_key_path):
            logging.error(f"ECS Key Path does not exist: {expanded_key_path}")
            return None, None
        logging.info(f"Connecting to ECS {ECS_HOST} with key {expanded_key_path}...")
        ssh_client.connect(ECS_HOST, username=ECS_USER, key_filename=expanded_key_path, timeout=30)
        sftp = ssh_client.open_sftp()
        logging.info("Successfully connected to ECS via SFTP.")
        return sftp, ssh_client
    except Exception as e:
        logging.error(f"Error connecting to ECS via SFTP: {e}")
        return None, None

def ensure_remote_dir(sftp_client, remote_dir_path):
    """Ensures a remote directory exists, creating it and its parents if necessary."""
    parts = remote_dir_path.strip('/').split('/')
    current_dir = ""
    for part in parts:
        if not part: continue
        current_dir += "/" + part
        try:
            sftp_client.stat(current_dir)
        except FileNotFoundError:
            logging.info(f"Remote directory {current_dir} not found, creating it.")
            sftp_client.mkdir(current_dir)

def upload_to_ecs(sftp_client, local_filepath, remote_filepath):
    """Uploads a local file to ECS via SFTP, ensuring remote directory exists."""
    if not sftp_client or not sftp_client.sock.connected: # type: ignore
        logging.error("SFTP client is not connected.")
        return False
    try:
        remote_dir = os.path.dirname(remote_filepath)
        ensure_remote_dir(sftp_client, remote_dir)
        logging.info(f"Uploading {local_filepath} to ECS at {remote_filepath}")
        sftp_client.put(local_filepath, remote_filepath)
        return True
    except Exception as e:
        logging.error(f"Error uploading {local_filepath} to ECS: {e}")
        return False

def update_db(db_conn, query, params):
    """Updates the database."""
    if not db_conn or not db_conn.is_connected():
        logging.error("Database connection is not active for update.")
        return False
    cursor = db_conn.cursor()
    try:
        cursor.execute(query, params)
        db_conn.commit()
        logging.info(f"Database updated. Query: {query[:30]}... Params: {params}. Rows affected: {cursor.rowcount}")
        return True
    except mysql.connector.Error as err:
        logging.error(f"Error updating database. Query: {query[:30]}... Params: {params}. Error: {err}")
        db_conn.rollback()
        return False
    finally:
        cursor.close()

# --- Main Execution ---
def main_process():
    """Main process to generate and upload audio for vocabulary terms."""
    logging.info("Starting vocabulary audio generation process...")
    db_conn = connect_db()
    if not db_conn:
        logging.critical("Failed to connect to database. Exiting.")
        return

    sftp_client, ssh_client = connect_sftp()
    if not sftp_client:
        logging.critical("Failed to connect to ECS via SFTP. Exiting.")
        if db_conn.is_connected(): db_conn.close()
        return

    processed_terms_count = 0
    try:
        terms_to_process = fetch_terms(db_conn, SELECT_TERMS_QUERY)
        if not terms_to_process:
            logging.info("No terms found to process based on the query.")
            return

        for term_record in terms_to_process:
            term_id = term_record['id']
            term_text = term_record['term']
            logging.info(f"\n--- Processing Term ID: {term_id}, Text: '{term_text}' ---")

            # --- Normal Speed Audio ---
            if not term_record.get('audio_url'):
                logging.info("Generating NORMAL speed audio...")
                prompt_normal = f"Say the vocabulary: {term_text}."
                audio_bytes_normal, rate_normal = generate_tts_via_proxy(
                    prompt_normal, GEMINI_TTS_MODEL, TTS_VOICE_NAME
                )
                if audio_bytes_normal:
                    local_file_normal = f"temp_term_{term_id}_normal.{AUDIO_OUTPUT_FORMAT}"
                    remote_file_normal = f"{REMOTE_BASE_DIR.rstrip('/')}/term_{term_id}_normal.{AUDIO_OUTPUT_FORMAT}"
                    
                    if save_audio_to_file(audio_bytes_normal, local_file_normal, rate_normal, AUDIO_OUTPUT_FORMAT):
                        if upload_to_ecs(sftp_client, local_file_normal, remote_file_normal):
                            public_url_normal = f"http://{ECS_HOST}{remote_file_normal.replace('/var/www', '')}"
                            update_db(db_conn, UPDATE_NORMAL_AUDIO_QUERY, (public_url_normal, TTS_VOICE_NAME, term_id))
                        else:
                            logging.error(f"Failed to upload normal audio for term ID {term_id}.")
                        os.remove(local_file_normal)
                    else:
                        logging.error(f"Failed to save/compress normal audio for term ID {term_id}.")
                else:
                    logging.warning(f"Failed to generate normal audio bytes for term ID {term_id}.")
            else:
                logging.info(f"Normal audio URL already exists for term ID {term_id}: {term_record['audio_url']}")

            time.sleep(2) # Be respectful to the API and proxy

            # --- Slow Speed Audio ---
            if not term_record.get('audio_url_slow'):
                logging.info("Generating SLOW speed audio...")
                prompt_slow = f"Listen to the vocabulary: {term_text}. Now, say it again, very slowly and clearly, enunciating each sound so an English learner can easily follow."
                audio_bytes_slow, rate_slow = generate_tts_via_proxy(
                    prompt_slow, GEMINI_TTS_MODEL, TTS_VOICE_NAME
                )
                if audio_bytes_slow:
                    local_file_slow = f"temp_term_{term_id}_slow.{AUDIO_OUTPUT_FORMAT}"
                    remote_file_slow = f"{REMOTE_BASE_DIR.rstrip('/')}/term_{term_id}_slow.{AUDIO_OUTPUT_FORMAT}"

                    if save_audio_to_file(audio_bytes_slow, local_file_slow, rate_slow, AUDIO_OUTPUT_FORMAT):
                        if upload_to_ecs(sftp_client, local_file_slow, remote_file_slow):
                            public_url_slow = f"http://{ECS_HOST}{remote_file_slow.replace('/var/www', '')}"
                            update_db(db_conn, UPDATE_SLOW_AUDIO_QUERY, (public_url_slow, term_id))
                        else:
                            logging.error(f"Failed to upload slow audio for term ID {term_id}.")
                        os.remove(local_file_slow)
                    else:
                        logging.error(f"Failed to save/compress slow audio for term ID {term_id}.")
                else:
                    logging.warning(f"Failed to generate slow audio bytes for term ID {term_id}.")
            else:
                logging.info(f"Slow audio URL already exists for term ID {term_id}: {term_record['audio_url_slow']}")
            
            processed_terms_count +=1
            logging.info(f"--- Finished processing Term ID: {term_id} ---")
            time.sleep(2) # Another pause before the next term

    except Exception as e:
        logging.critical(f"An unexpected error occurred in main_process: {e}", exc_info=True)
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            logging.info("Database connection closed.")
        if sftp_client:
            sftp_client.close()
        if ssh_client and ssh_client.get_transport() and ssh_client.get_transport().is_active(): # type: ignore
            ssh_client.close()
            logging.info("SFTP/SSH connection closed.")
        logging.info(f"Vocabulary audio generation process finished. Processed {processed_terms_count} terms in this run.")

if __name__ == "__main__":
    # Ensure the ECS key path is correct and accessible before running.
    # Example: if not os.path.exists(os.path.expanduser(ECS_KEY_PATH)):
    #    print(f"ERROR: ECS Key Path not found at {os.path.expanduser(ECS_KEY_PATH)}")
    #    exit(1)
    main_process()