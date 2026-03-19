import subprocess

IMAGE_NAME = "lecture-rag-api"

def image_exists():
    result = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME],
        capture_output=True,
        text=True
    )
    return result.stdout.strip() != ""

def build_image():
    print("Building Docker image...")
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."])

def gpu_available():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except:
        return False

def run_container(gpu=False):
    cmd = ["docker", "run"]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += [
        "-p", "8000:8000",
        "-v", "rag_data:/app/data",
        IMAGE_NAME
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    if not image_exists():
        build_image()

    if gpu_available():
        print("Starting with GPU...")
        run_container(gpu=True)
    else:
        print("Starting with CPU...")
        run_container(gpu=False)