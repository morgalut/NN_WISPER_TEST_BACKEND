

# Hebrew Whisper Transcription Service

## Overview
The Hebrew Whisper Transcription Service is a Flask-based application designed to transcribe Hebrew audio files into text using OpenAI's Whisper model. It enhances transcription accuracy by leveraging custom Hebrew text processing, including tokenization, normalization, and spell-checking against a custom Hebrew word list. The service includes support for continuous model retraining based on feedback and improved accuracy with a curated Hebrew word list.

## Features
- **Hebrew Transcription**: Transcribes Hebrew audio files to text.
- **Text Normalization**: Uses `hebrew-tokenizer` to handle Hebrew-specific tokenization, diacritical mark removal, and punctuation normalization.
- **Word Validation**: Cross-references transcribed text with a Hebrew word list for improved accuracy.
- **Model Retraining**: Periodic model retraining based on user feedback, stored in `feedback` files.
- **Feedback Collection**: Allows users to submit feedback to correct transcriptions, improving future accuracy.

## Project Structure

hebrew_whisper/
├── app.py                     # Main Flask app entry point
├── utils/
│   ├── config.py              # Configuration constants (directories, languages, thresholds)
│   ├── routes.py              # API routes for transcription and feedback
│   └── dataset.py             # Dataset loader for fine-tuning
├── services/
│   ├── transcription_service.py   # Main service for transcription and saving files
│   ├── training_service.py        # Handles model retraining with feedback
│   ├── feedback_manager.py        # Stores user feedback for model improvements
│   ├── hebrew_word_validator.py   # Validates and corrects Hebrew words from a word list
│   └── text_processor.py          # Tokenizes and normalizes Hebrew text
└── base/
    └── heb_stopwords.txt      # Hebrew word list for validation and correction
```

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd hebrew_whisper
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Hebrew Tokenizer**:
   The project uses `hebrew-tokenizer` for Hebrew-specific text processing:
   ```bash
   pip install hebrew-tokenizer
   ```

5. **Download Whisper Model**:
   - Follow OpenAI's instructions to download and set up the Whisper model in your environment.
   - Replace `"base"` in `transcription_service.py` with the path to the Whisper model if needed.

## Configuration

- **File Paths**: Update file paths and constants in `config.py` as necessary.
- **Hebrew Word List**: Ensure that the file `heb_stopwords.txt` in the `base` directory contains common Hebrew words for accuracy enhancement.

## Running the Application

1. **Start the Flask Application**:
   ```bash
   python app.py
   ```

2. **Access the API**:
   - The app will run locally at `http://127.0.0.1:5000`.
   - The `/api/transcribe` endpoint allows file uploads for transcription.

## API Endpoints

### 1. `/api/transcribe` (POST)
Transcribe an audio file to Hebrew text.

- **Request**:
  - **File**: Upload the audio file (e.g., MP3 format).
  - **Language**: Set to `"he"` for Hebrew.
- **Response**: Returns a `.txt` file with the transcribed text.

Example `curl` command:
```bash
curl -X POST -F "file=@\"path/to/audio_file.mp3\"" -F "language=he" http://127.0.0.1:5000/api/transcribe --output transcription_output.txt
```

### 2. `/api/feedback` (POST)
Submit feedback with corrected transcription.

- **Request**:
  - JSON with `original_filename` and `corrected_transcription`.
- **Response**: Confirms receipt of feedback.

## Usage

### Transcription Service
The transcription service accepts an audio file and performs the following:
1. **Normalization**: Uses `TextProcessor` to clean Hebrew text.
2. **Word Validation**: `HebrewWordValidator` checks each word against the Hebrew word list (`heb_stopwords.txt`) and makes corrections if necessary.
3. **Saving Transcription**: The corrected transcription is saved as a `.txt` file.

### Training and Feedback
The `TrainingService` class handles model retraining. Feedback is stored, and after reaching a predefined threshold, the model retrains with updated data to improve accuracy.

## Dependencies
- `Flask`: Web framework for the API.
- `hebrew-tokenizer`: Hebrew text tokenizer.
- `torch`: Required for Whisper model.
- `spacy`: Used for NLP pipeline (with workaround due to no Hebrew model).
- `transformers`: For model tokenizer and fine-tuning.

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are welcome.

## License
This project is licensed under the MIT License.

