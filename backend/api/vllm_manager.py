import docker
import time
import requests
import threading

class VLLMManager:
    def __init__(
        self,
        model_name: str,
        container_name: str = "vllm-server",
        network_name: str = "rag-network",
        port: int = 30001,
        **startup_kwargs
    ):
        self.model_name = model_name
        self.container_name = container_name
        self.network_name = network_name
        self.port = port

        self.client = docker.from_env()
        self.startup_kwargs = startup_kwargs

    def ensure_network(self):
        networks = self.client.networks.list(names=[self.network_name])
        if not networks:
            self.client.networks.create(self.network_name)

    def connect_api_container(self):
        try:
            network = self.client.networks.get(self.network_name)
            network.connect("lecture-rag-api")
        except docker.errors.APIError as e:
            if "already exists" in str(e):
                pass  #already connected
            else:
                print("You need to run your docker container with '--name lecture-rag-api'")
                raise

    def start_container(self):
        try:
            self.container = self.client.containers.get(self.container_name)
            if self.container.status == "running":
                return
            else:
                self.container.start()
                return
        except docker.errors.NotFound:
            pass

        full_command = ["serve", self.model_name, "--port", f"{self.port}"]
        for key, value in self.startup_kwargs.items():
            if value is not None:
                full_command += [f"--{key}", f"{value}"]

        print(f"VLLM called with: '{full_command}'")
        self.container = self.client.containers.run(
            "vllm/vllm-openai",
            name=self.container_name,
            entrypoint=["vllm"],
            command=full_command,
            detach=True,
            network=self.network_name,
            ports={f"{self.port}/tcp": self.port},  # optional (host access)
            device_requests=[
                docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            ],
            volumes={
                "hf_cache": {
                    "bind": "/root/.cache/huggingface",
                    "mode": "rw"
                },
                "vllm_cache": {
                    "bind": "/root/.cache/vllm",
                    "mode": "rw"
                }
            }
        )

    def wait_until_ready(self, timeout: int = 900):
        url = f"http://{self.container_name}:{self.port}/health"

        for i in range(timeout):
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    print("vLLM ready", flush=True)
                    return
            except:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM server did not become ready in time.")

    def stream_startup_logs(self, stop_event: threading.Event):
        try:
            since_start = time.time()
            for line in self.container.logs(stream=True,follow=True, since=since_start):
                if stop_event.is_set():
                    break
                decoded = line.decode("utf-8", errors="ignore").rstrip()
                print(f"[vLLM] {decoded}", flush=True)

        except Exception as e:
            print(f"[vLLM log stream ended: {e}]", flush=True)

    def get_url(self):
        return f"http://{self.container_name}:{self.port}/v1/chat/completions"

    def start(self):
        self.ensure_network()
        self.connect_api_container()
        self.start_container()

        stop_event = threading.Event()
        log_thread = threading.Thread(
            target=self.stream_startup_logs,
            args=(stop_event,),
            daemon=True
        )
        log_thread.start()
        try:
            self.wait_until_ready()
        finally:
            stop_event.set()  # signals the log thread to exit
            log_thread.join(timeout=5)

    def stop(self):
        try:
            self.container = self.client.containers.get(self.container_name)
            print("[VLLM] Stopping container...")
            self.container.stop()
            self.container.remove()
        except docker.errors.NotFound:
            pass