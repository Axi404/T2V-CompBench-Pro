import cv2
import numpy as np
import torch
import os
from torchvision.io import write_video
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_frames(video_path, num_frames=16):
    frames = []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("total frames", total_frames)
    if total_frames <= num_frames:
        frame_indices = np.arange(total_frames)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def rgb_to_yuv(frame):
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return yuv_frame


def frames_to_video(frames, output_path, fps=8):
    yuv_frames = [rgb_to_yuv(frame) for frame in frames]
    video_tensor = torch.from_numpy(np.array(yuv_frames)).to(torch.uint8)
    write_video(
        output_path, video_tensor, fps, video_codec="h264", options={"crf": "18"}
    )


def convert_video(input_path, output_path, num_frames):
    frames = extract_frames(input_path, num_frames=num_frames)
    frames_to_video(frames, output_path)


def merge_grid(image_list):
    # Concatenate the images horizontally
    row1 = np.concatenate((image_list[0], image_list[1]), axis=1)
    row2 = np.concatenate((image_list[2], image_list[3]), axis=1)
    row3 = np.concatenate((image_list[4], image_list[5]), axis=1)
    # Concatenate the rows vertically
    grid = np.concatenate((row1, row2, row3), axis=0)
    return grid


def read_video_path(video_path):
    if os.path.isdir(video_path):  # if video_path is a list of videos
        video = os.listdir(video_path)
        video = [
            video
            for video in video
            if not os.path.isdir(os.path.join(video_path, video))
        ]
    elif os.path.isfile(video_path):  # else if video_path is a single video
        video = [os.path.basename(video_path)]
        video_path = os.path.dirname(video_path)
    video.sort()
    return video, video_path


def _process_single_video_to_frames(
    v: str, video_path: str, output_path: str, num_frames: int
) -> str:
    """
    Process a single video to frames.

    Args:
        v: video filename
        video_path: directory containing the video
        output_path: directory to save the frames
        num_frames: number of frames to extract

    Returns:
        vid_id: the video id that was processed
    """
    vid_id = v.split(".")[0]
    frames_dir = os.path.join(output_path, vid_id)
    os.makedirs(frames_dir, exist_ok=True)
    vid_path = os.path.join(video_path, v)
    frames = extract_frames(vid_path, num_frames=num_frames)
    for frame_count, frame in enumerate(frames):
        frame_filename = os.path.join(frames_dir, f"{vid_id}_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
    return vid_id


def convert_video_to_frames(
    video_path: str, num_frames: int = 16, max_workers: int = 8
) -> str:
    """
    Convert videos to frames using multi-threading.

    Args:
        video_path: path to video file or directory containing videos
        num_frames: number of frames to extract per video
        max_workers: maximum number of threads to use

    Returns:
        output_path: directory where frames are saved
    """
    video, video_path = read_video_path(video_path)
    print(f"start converting video to {num_frames} frames from path:", video_path)
    print(f"using {max_workers} threads")

    output_path = os.path.join(
        os.path.dirname(video_path), "frames", os.path.basename(video_path)
    )
    os.makedirs(output_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_video_to_frames, v, video_path, output_path, num_frames
            ): v
            for v in video
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                v = futures[future]
                print(f"Error processing {v}: {e}")

    print("finish converting from path: ", video_path)
    print("video frames stored in: ", output_path)
    return output_path


def _process_single_video_to_standard(
    v: str, video_path: str, output_path: str, num_frames: int
) -> str:
    """
    Process a single video to standard format.

    Args:
        v: video filename
        video_path: directory containing the video
        output_path: directory to save the standard video
        num_frames: number of frames to extract

    Returns:
        v_mp4: the output video filename
    """
    v_mp4 = v.split(".")[0] + ".mp4"
    convert_video(
        os.path.join(video_path, f"{v}"),
        os.path.join(output_path, f"{v_mp4}"),
        num_frames,
    )
    return v_mp4


def convert_video_to_standard_video(
    video_path: str, num_frames: int, max_workers: int = 8
) -> str:
    """
    Convert videos to standard format using multi-threading.

    Args:
        video_path: path to video file or directory containing videos
        num_frames: number of frames to extract per video
        max_workers: maximum number of threads to use

    Returns:
        output_path: directory where standard videos are saved
    """
    video, video_path = read_video_path(video_path)
    print("start converting video to video with 16 frames from path:", video_path)
    print(f"using {max_workers} threads")

    output_path = os.path.join(
        os.path.dirname(video_path), "video_standard", os.path.basename(video_path)
    )
    os.makedirs(output_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_video_to_standard,
                v,
                video_path,
                output_path,
                num_frames,
            ): v
            for v in video
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                v = futures[future]
                print(f"Error processing {v}: {e}")

    print("finish converting from path: ", video_path)
    print("standard video stored in: ", output_path)
    return output_path


def _process_single_video_to_grid(
    v: str, video_path: str, output_path: str, num_image: int
) -> str:
    """
    Process a single video to image grid.

    Args:
        v: video filename
        video_path: directory containing the video
        output_path: directory to save the grid image
        num_image: number of frames to extract

    Returns:
        vid_id: the video id that was processed
    """
    vid_id = v.split(".")[0]
    vid_path = os.path.join(video_path, v)
    frames = extract_frames(vid_path)
    frame_indices = np.linspace(
        0, len(frames) - 1, num_image, dtype=int
    )  # take 6 from 16 evenly, 1st & last included
    grid = [frames[i] for i in frame_indices]
    grid_image = merge_grid(grid)
    grid_filename = os.path.join(output_path, f"{vid_id}.png")
    cv2.imwrite(grid_filename, grid_image)
    return vid_id


def convert_video_to_grid(
    video_path: str, num_image: int = 6, max_workers: int = 8
) -> str:
    """
    Convert videos to image grids using multi-threading.

    Args:
        video_path: path to video file or directory containing videos
        num_image: number of frames to extract per video
        max_workers: maximum number of threads to use

    Returns:
        output_path: directory where grid images are saved
    """
    video, video_path = read_video_path(video_path)
    print(
        f"start converting video to image grid with {num_image} frames from path:",
        video_path,
    )
    print(f"using {max_workers} threads")

    output_path = os.path.join(
        os.path.dirname(video_path), "image_grid", os.path.basename(video_path)
    )
    os.makedirs(output_path, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_video_to_grid, v, video_path, output_path, num_image
            ): v
            for v in video
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                v = futures[future]
                print(f"Error processing {v}: {e}")

    print("finish converting from path: ", video_path)
    print("image grid stored in: ", output_path)
    return output_path
