import docker
import time
import requests


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
        except docker.errors.APIError:
            # already connected → ignore
            pass

    def start_container(self):
        try:
            container = self.client.containers.get(self.container_name)
            if container.status == "running":
                return
            else:
                container.start()
                return
        except docker.errors.NotFound:
            pass

        full_command = ["serve", self.model_name, "--port", f"{self.port}"]

        for key, value in self.startup_kwargs.items():
            if value is not None:
                full_command += [f"--{key}", f"{value}"]
        print(f"VLLM called with: '{full_command}'")
        self.client.containers.run(
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
        )

    def wait_until_ready(self, timeout: int = 900):
        url = f"http://{self.container_name}:{self.port}/v1/models"

        for i in range(timeout):
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
            if i % 10 == 0:
                print(f"[VLLM] Still waiting... ({i}s)")

        raise RuntimeError("vLLM server did not become ready in time.")

    def get_url(self):
        return f"http://{self.container_name}:{self.port}/v1/chat/completions"

    def start(self):
        self.ensure_network()
        self.connect_api_container()
        self.start_container()
        self.wait_until_ready()

    def stop(self):
        try:
            container = self.client.containers.get(self.container_name)
            print("[VLLM] Stopping container...")
            container.stop()
            container.remove()
        except docker.errors.NotFound:
            pass