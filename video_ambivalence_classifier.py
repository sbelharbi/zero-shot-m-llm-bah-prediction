import torch
import os
import sys
import argparse
import pandas as pd
import numpy as np
import av
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


NUM_FRAMES = 8


def sample_video_frames(video_path, num_frames):
    """
    Sample frames uniformly from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        
    Returns:
        List of PIL Images resized to 224x224
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in frame_indices:
            img = frame.to_image()
            img = img.resize((224, 224))
            frames.append(img)
        if len(frames) >= num_frames:
            break
    container.close()
    return frames


def create_prompt(transcript, prompt_flag):
    """
    Create prompt based on the specified flag.
    
    Args:
        transcript: Video transcript text
        prompt_flag: Prompt type ('s', 'd1', 'd2', 'td1', 'td2')
        
    Returns:
        Formatted prompt string
    """
    base_instruction = "Classify the emotion in the video as either 'Non-Ambivalent' or 'Ambivalent'.\nRespond with only one word: "
    
    prompts = {
        's': f"USER: <video>\n{base_instruction}ASSISTANT:",
        
        'd1': f"USER: <video>\nDefinition: Ambivalence is the state of having contradictory or conflicting feelings or attitudes towards something or someone simultaneously.{base_instruction}ASSISTANT:",
        
        'd2': f"USER: <video>\nDefinition: Ambivalence and Hesitancy is understood as the simultaneous experience of desires for change and against change.{base_instruction}ASSISTANT:",
        
        'td1': f"USER: <video>\nVideo transcript: {transcript}. Definition: Ambivalence is the state of having contradictory or conflicting feelings or attitudes towards something or someone simultaneously.{base_instruction}ASSISTANT:",
        
        'td2': f"USER: <video>\nVideo transcript: {transcript}. Definition: Ambivalence and Hesitancy is understood as the simultaneous experience of desires for change and against change.{base_instruction}ASSISTANT:",
    }
    
    return prompts.get(prompt_flag, prompts['s'])


def predict_video_label(video_path, model, processor, transcript, prompt_flag, device):
    """
    Predict ambivalence label for a single video.
    
    Args:
        video_path: Path to video file
        model: Loaded VideoLlavaForConditionalGeneration model
        processor: VideoLlavaProcessor instance
        transcript: Video transcript text
        prompt_flag: Prompt type flag
        device: Device to run inference on
        
    Returns:
        Model's text response
    """
    print(f"Sampling {NUM_FRAMES} frames from the video...")
    video_frames = sample_video_frames(video_path, num_frames=NUM_FRAMES)

    assert len(video_frames) == NUM_FRAMES, f"Got {len(video_frames)} frames (expected {NUM_FRAMES})"
    print("First frame size:", video_frames[0].size)

    prompt = create_prompt(transcript, prompt_flag)

    inputs = processor(
        text=prompt,
        videos=[video_frames],
        return_tensors="pt",
        padding=True
    ).to(device)

    print("Running inference...")
    output = model.generate(**inputs, max_new_tokens=10)
    response = processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    print(f"\nPredicted label: {response}")
    return response


def load_model(model_id, device):
    """
    Load the VideoLLaVA model and processor.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    print("Loading model...")
    processor = VideoLlavaProcessor.from_pretrained(
        model_id,
        image_size=224,
        num_frames=NUM_FRAMES,
        vision_config={"hidden_size": 1024}
    )

    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )

    model.video_projection = torch.nn.Linear(2056, 1024).to(device)
    model.to(device)
    print("Model loaded successfully!")
    
    return model, processor


def run_all_predictions(test_file_path, test_videos_root_path, prompt_flag, output_dir, device):
    """
    Run predictions on all videos in test file.
    
    Args:
        test_file_path: Path to test file with video paths, labels, and transcripts
        test_videos_root_path: Root directory containing video files
        prompt_flag: Prompt type to use
        output_dir: Directory to save output CSV files
        device: Device to run inference on
    """
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_id = "LanguageBind/Video-LLaVA-7B-hf"
    model, processor = load_model(model_id, device)

    # Read test file
    df = pd.read_csv(test_file_path, sep="delimiter", header=None)
    df[['path', 'label', 'text']] = df[0].str.split(',', n=2, expand=True)
    
    video_path = df['path'].tolist()
    video_labels = [int(label) for label in df['label'].tolist()]
    transcript = df['text'].tolist()
    
    # Create full video paths
    full_video_path = [os.path.join(test_videos_root_path, path) for path in video_path]
    path_text_dict = dict(zip(df['path'], df['text']))
    
    all_video_responses = []
    
    # Process each video
    for path in full_video_path:
        if not os.path.exists(path):
            print(f"Warning: Video path {path} does not exist")
            continue
            
        relative_path = path.split(f"{test_videos_root_path}/")[-1]
        response = predict_video_label(
            path, model, processor, 
            path_text_dict[relative_path], 
            prompt_flag, device
        )

        print(f"Video path: {path} - Predicted label: {response}")
        all_video_responses.append({
            "video_path": path,
            "response": response
        })

    # Process results
    video_paths = [video["video_path"] for video in all_video_responses]
    video_responses = [video["response"] for video in all_video_responses]
    video_responses = [response.split(":")[-1].strip() for response in video_responses]
    video_responses_mapped = [0 if response == "Non-Ambivalent" else 1 for response in video_responses]

    # Calculate metrics
    accuracy = accuracy_score(video_labels, video_responses_mapped)
    f1 = f1_score(video_labels, video_responses_mapped, average='macro')

    # Baseline metrics (all zeros)
    all_zeros = [0] * len(video_labels)
    accuracy_all_zeros = accuracy_score(video_labels, all_zeros)    
    f1_all_zeros = f1_score(video_labels, all_zeros, average='macro')

    print(f"\nBaseline (all zeros) - Accuracy: {accuracy_all_zeros:.4f}, F1: {f1_all_zeros:.4f}")
    print(f"Model Performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    video_responses_df = pd.DataFrame({
        "video_path": video_paths,
        "response": video_responses_mapped
    })
    predictions_file = os.path.join(output_dir, f'predictions_{prompt_flag}.csv')
    video_responses_df.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved to: {predictions_file}")

    video_labels_df = pd.DataFrame({
        "video_path": video_paths,
        "label": video_labels
    })
    labels_file = os.path.join(output_dir, "ground_truth_labels.csv")
    video_labels_df.to_csv(labels_file, index=False)
    print(f"Ground truth labels saved to: {labels_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Video Ambivalence Classification using VideoLLaVA'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        required=True,
        help='Path to test file (CSV format: video_path,label,transcript)'
    )
    parser.add_argument(
        '--videos_root',
        type=str,
        required=True,
        help='Root directory containing video files'
    )
    parser.add_argument(
        '--prompt_flag',
        type=str,
        choices=['s', 'd1', 'd2', 'td1', 'td2'],
        default='td1',
        help='Prompt type: s (simple), d1/d2 (with definition), td1/td2 (transcript+definition)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Directory to save output files'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='GPU device ID to use'
    )
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    run_all_predictions(
        args.test_file,
        args.videos_root,
        args.prompt_flag,
        args.output_dir,
        device
    )


if __name__ == '__main__':
    main()