# Video Ambivalence Classification

## Overview
This code provides the baseline results for the Video Level Classification on the BAH dataset with Zero Shot LVLM.
This code leverages the VideoLLaVA-7B model to perform binary emotion classification on videos. It supports multiple prompting strategies including definition-based and transcript-augmented approaches.

## Features

- Binary emotion classification (Ambivalent vs Non-Ambivalent)
- Multiple prompting strategies with varying levels of context
- Support for video transcripts to enhance classification
- Reproducible results with fixed random seeds
- Batch processing of video datasets
- Performance metrics (Accuracy, F1-score)

## Requirements

### Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install av
pip install pandas
pip install scikit-learn
pip install numpy
pip install Pillow
```

### Hardware

- CUDA-compatible GPU (recommended)
- Sufficient VRAM for 7B model (16GB+ recommended)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd video-ambivalence-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The model will be automatically downloaded from HuggingFace on first run.

## Usage

### Basic Usage

```bash
python video_ambivalence_classifier.py \
  --test_file /path/to/test_file.txt \
  --videos_root /path/to/videos/directory \
  --prompt_flag td1 \
  --output_dir ./results
```

### Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--test_file` | str | Yes | - | Path to test file containing video paths, labels, and transcripts |
| `--videos_root` | str | Yes | - | Root directory containing the video files |
| `--prompt_flag` | str | No | `td1` | Prompt type (see below) |
| `--output_dir` | str | No | `./output` | Directory to save output CSV files |
| `--gpu` | str | No | `0` | GPU device ID to use |

### Prompt Types

The `--prompt_flag` parameter controls the prompting strategy:

- **`s`** (Simple): Basic classification prompt without context
- **`d1`** (Definition 1): Includes definition of ambivalence as contradictory feelings
- **`d2`** (Definition 2): Includes definition focusing on desires for/against change
- **`td1`** (Transcript + Definition 1): Combines video transcript with first definition
- **`td2`** (Transcript + Definition 2): Combines video transcript with second definition

### Input File Format

The test file should be a CSV-like text file with the following format:

```
video_path,label,transcript
path/to/video1.mp4,0,This is the transcript for video 1
path/to/video2.mp4,1,This is the transcript for video 2
```

Where:
- `video_path`: Relative path to video file (from `videos_root`)
- `label`: Ground truth label (0 = Non-Ambivalent, 1 = Ambivalent)
- `transcript`: Text transcript of the video (can be empty for non-transcript prompts)

## Output

The script generates two CSV files in the output directory:

1. **`predictions_{prompt_flag}.csv`**: Contains video paths and predicted labels
   - `video_path`: Full path to video file
   - `response`: Predicted label (0 or 1)

2. **`ground_truth_labels.csv`**: Contains video paths and ground truth labels
   - `video_path`: Full path to video file
   - `label`: Ground truth label (0 or 1)

### Console Output

The script prints:
- Progress for each video

- Model performance metrics (Accuracy and F1-score)
- File save locations

## Examples

### Example 1: Simple Prompt
```bash
python video_ambivalence_classifier.py \
  --test_file data/test_300.txt \
  --videos_root /datasets/videos \
  --prompt_flag s \
  --output_dir results/simple
```





## Reproducibility

The code sets random seeds for reproducibility:
- PyTorch seed: 42
- NumPy seed: 42
- CUDNN deterministic mode: Enabled
- CUDNN benchmark mode: Disabled

## Performance Metrics

The script calculates:
- **F1-Score**: Macro-averaged F1-score

## Troubleshooting

### Out of Memory Error
- Reduce batch size (currently processes one video at a time)
- Use a GPU with more VRAM
- Try CPU mode (slower but uses system RAM)

### Video File Not Found
- Verify `--videos_root` path is correct
- Check that paths in test file are relative to `videos_root`
- Ensure video files exist and are readable

### Model Download Issues
- Ensure internet connection for first-time model download
- Check HuggingFace credentials if using gated models
- Verify sufficient disk space (~15GB for model)




