<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Flux Image Service</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.6; }
    h1, h2, h3 { color: #2c3e50; }
    pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; }
    code { background: #eee; padding: 2px 4px; }
    a { color: #3498db; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .section { margin-bottom: 2rem; }
  </style>
</head>
<body>

  <h1>Flux Image Service</h1>

  <p>A lightweight gRPC-powered image-generation service backed by a stable Diffusers FLUX 4-bit pipeline, plus an HTTP metrics endpoint and a Streamlit front-end.</p>

  <div class="section">
    <h2>🎯 Features</h2>
    <ul>
      <li><strong>gRPC API</strong>
        <ul>
          <li><code>Ping</code> — health check</li>
          <li><code>Generate</code> — text-to-image inference (returns PNG + timing)</li>
        </ul>
      </li>
      <li><strong>HTTP Metrics</strong> (<code>/metrics</code>)
        <ul>
          <li>CPU, RAM, GPU usage</li>
          <li>Worker &amp; gRPC health</li>
        </ul>
      </li>
      <li><strong>Worker Auto-scaling</strong>
        <ul>
          <li>Lazy-loads the FLUX 4-bit model on first request</li>
          <li>Unloads after configurable idle timeout</li>
        </ul>
      </li>
      <li><strong>Streamlit Front-End</strong>
        <ul>
          <li>Live metrics dashboard</li>
          <li>“Generate” form with prompt, height/width, steps, guidance sliders</li>
        </ul>
      </li>
      <li><strong>Single-client concurrency</strong>
        <ul>
          <li>Only one inference at a time; others immediately receive <code>RESOURCE_EXHAUSTED</code></li>
        </ul>
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>🏗️ Architecture</h2>
    <pre><code>
┌───────────┐    gRPC    ┌────────────┐    multiprocessing    ┌───────────┐
│  Client   │ ─────────> │  Server    │ ────────────────────>│  Worker   │
│ (Streamlit│            │ (server.py)│                       │ (worker.py)│
└───────────┘            └────────────┘                       └───────────┘
                                │
                                │ HTTP
                                ▼
                           ┌───────────┐
                           │  Metrics  │
                           │ (Flask)   │
                           └───────────┘
    </code></pre>
    <p>
      <strong>Server</strong> spins up a background <code>worker_main</code> process on first request.<br>
      <strong>Worker</strong> loads &amp; warms the FLUX 4-bit pipeline, handles inference, then auto-exits after <code>WORKER_IDLE_SEC</code> of idle.<br>
      <strong>Server</strong> enforces a one-at-a-time lock so concurrent generate calls fail fast.<br>
      <strong>Metrics</strong> are exposed over a simple HTTP GET for easy polling.
    </p>
  </div>

  <div class="section">
    <h2>🚀 Getting Started</h2>

    <h3>Prerequisites</h3>
    <ul>
      <li>Python 3.9+</li>
      <li>NVIDIA GPU + CUDA (for 4-bit offload)</li>
      <li><a href="https://docker.com">Docker</a> (optional, for containerization)</li>
    </ul>

    <h3>Clone &amp; Setup</h3>
    <pre><code>
git clone https://github.com/&lt;your-username&gt;/flux-image-service.git
cd flux-image-service
    </code></pre>
  </div>

  <div class="section">
    <h3>1) Install &amp; Run the Server</h3>
    <pre><code>
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install \
    grpcio grpcio-tools \
    diffusers bitsandbytes torch \
    flask psutil pynvml
# Generate Python gRPC code:
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. image_gen.proto

# (Optional) override via ENV:
export API_KEYS="client1,otherkey"
export WORKER_IDLE_SEC=60

python server.py
    </code></pre>
    <p>
      <strong>gRPC</strong> listens on <code>0.0.0.0:50051</code><br>
      <strong>Metrics</strong> HTTP on <code>0.0.0.0:8000/metrics</code>
    </p>
  </div>

  <div class="section">
    <h3>2) Install &amp; Run the Client (Streamlit)</h3>
    <pre><code>
cd client
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install grpcio protobuf streamlit pillow requests streamlit-autorefresh

streamlit run app.py --server.address &lt;SERVER_IP_OR_HOSTNAME&gt;
    </code></pre>
    <p>Use the sidebar to enter your API key (e.g. <code>client1</code>), prompt, and generation parameters. The live metrics panel polls <code>/metrics</code> every 5 s.</p>
  </div>

  <div class="section">
    <h2>🔌 API Reference</h2>

    <h3>1) Ping (gRPC health check)</h3>
    <pre><code>
grpcurl -plaintext \
  -import-path server/ \
  -proto    server/image_gen.proto \
  -rpc-header "api-key: client1" \
  -d '{}' \
  &lt;SERVER&gt;:50051 \
  imagegen.ImageGen/Ping
    </code></pre>
    <p><strong>Response:</strong></p>
    <pre><code>
{ "message": "Pong" }
    </code></pre>

    <h3>2) Generate (gRPC inference)</h3>
    <pre><code>
grpcurl -plaintext \
  -import-path server/ \
  -proto    server/image_gen.proto \
  -rpc-header "api-key: client1" \
  -d '{
    "prompt":"A cyberpunk city at night",
    "height":512,
    "width":512,
    "num_inference_steps":10,
    "guidance_scale":3.5
  }' \
  &lt;SERVER&gt;:50051 \
  imagegen.ImageGen/Generate > response.bin
    </code></pre>
    <p>
      The <code>response.bin</code> file contains a <code>GenerateResponse</code> message:
      <ul>
        <li>First field: raw PNG bytes</li>
        <li>Second field: <code>inference_time</code> in seconds</li>
      </ul>
    </p>

    <h3>3) Metrics (HTTP)</h3>
    <pre><code>
curl http://&lt;SERVER&gt;:8000/metrics
    </code></pre>
    <p><strong>Response:</strong></p>
    <pre><code>
{
  "cpu_percent": 12.3,
  "ram_used_mb": 14321,
  "gpu_used_mb": 8123,
  "worker_alive": true,
  "grpc_alive": true
}
    </code></pre>
  </div>

  <div class="section">
    <h2>🐳 Docker (optional)</h2>
    <p>Build and run the server in a container:</p>
    <pre><code>
# server/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      grpcio grpcio-tools \
      diffusers bitsandbytes torch \
      flask psutil pynvml
CMD ["python","server.py"]
    </code></pre>
    <pre><code>
docker build -t flux-image-server server/
docker run --gpus all -p 50051:50051 -p 8000:8000 flux-image-server
    </code></pre>
  </div>

  <div class="section">
    <h2>📄 License</h2>
    <p>MIT © Your Name</p>
    <p>Feel free to file issues or PRs if you run into any problems!</p>
  </div>

</body>
</html>
