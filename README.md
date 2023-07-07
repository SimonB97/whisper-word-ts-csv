# Whisper Word Timestamps

This repository contains a Python script that uses OpenAI's Whisper ASR system to transcribe audio files and provide word-level timestamps. The script is designed to be flexible and allows you to choose from different Whisper models, specify the language of the audio, and set the maximum duration of the audio file. This project is largely based on [this Hugging Face project](https://huggingface.co/spaces/Matthijs/whisper_word_timestamps) and is modified to output a CSV file with words and timestamps.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use this script, you need to have Python installed on your system. The script also depends on several Python libraries, which are listed in the `requirements.txt` file.

Here are the steps to install the necessary dependencies:

1. Clone the repository:

```bash
git clone https://github.com/sbene97/whisper-word-ts-csv.git
```

2. Navigate to the cloned repository:

```bash
cd whisper-word-ts-csv
```

3. Create a new Python virtual environment:

```bash
python -m venv env
```

4. Activate the virtual environment:

- On Windows:

```bash
.\env\Scripts\activate
```

- On Unix or MacOS:

```bash
source env/bin/activate
```

5. Install the dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script from the command line using the following syntax:

```bash
python transcribe.py --audio <path/to/audio> --language <language> --length <length> --model <model>
```

Here's what each argument does:

- `--audio`: Specifies the path to the audio file you want to transcribe.
- `--language`: Specifies the language of the audio. The default is English.
- `--length`: Specifies the length of the audio file in seconds.
- `--model`: Specifies the Whisper model to use. Options include 'tiny', 'base', 'small', 'medium', and 'large'. The default is 'small'.

>

## Configuration

You can configure the script by modifying the following variables at the top of the script:

- `model`: The Whisper model to use. Options include 'tiny', 'base', 'small', 'medium', and 'large'. The default is 'small'.
- `max_duration`: The maximum duration of the audio file in seconds. The default is 600.
- `rows_out`: The number of rows to print in the output DataFrame for quick inspection. The default is 30.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the terms of the MIT license.