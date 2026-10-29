# AI Marketing Video Workflow Agent

A local Streamlit workflow for creating personalized sports marketing videos. It combines copy generation, cinematic script direction, and image-to-video generation in one workflow.

## Workflow

1. Generate a personalized slogan and product description
2. Generate an 8-second, time-coded cinematic video script
3. Generate a short image-to-video advertisement

## Local setup

1. Copy `.env.example` to `.env` and add `HF_TOKEN` and `FAL_KEY`, or put local values in `config.local.yml`.
2. Edit `config.yml` to select models, script creativity, duration, resolution, and aspect ratio.
3. Start the app:

	- Windows: double-click `run_windows.cmd`
	- Linux: run `chmod +x run_linux.sh && ./run_linux.sh`

4. Open `http://localhost:8501`.

The app is intended for local use and does not require Streamlit Community Cloud deployment.

## Configuration

`config.yml` is safe for general workflow settings. Keep private keys in environment variables or the ignored `config.local.yml` file. Environment variables take precedence over YAML credentials.

The workflow uses DeepSeek V4.1 Flash for multimodal copy and cinematic direction, and FLUX 3 Max for image-to-video generation with native audio. The cinematic prompt is designed around the configured video duration and asks for time-coded shots, camera language, lighting, sound design, product continuity, realistic movement, and an exact final slogan frame.

## Requirements

- Python 3.10+
- A Hugging Face token for text generation
- A FAL key for video generation
- Network access to the configured model providers
