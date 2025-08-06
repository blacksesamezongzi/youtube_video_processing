import os
import subprocess

def download_videos():
    video_urls = "video_urls.txt"
    output_dir = "downloaded_videos"
    os.makedirs(output_dir, exist_ok = True)

    # Read URLs from txt file and download
    with open(video_urls, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        print(f"Downloading {url}")
        cmd = ["yt-dlp",
            "-f", "bestvideo[ext=mp4]",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            url]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    print("Starting video download...")
    download_videos()
    print("All videos downloaded successfully.")