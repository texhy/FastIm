#!/usr/bin/env python3
import os
import time
import logging
import threading
import multiprocessing as mp
import signal
import json

from flask import Flask, jsonify           # HTTP metrics
import psutil, pynvml, torch, grpc         # monitoring + gRPC
from concurrent import futures

import image_gen_pb2, image_gen_pb2_grpc
from worker import worker_main

# -------- Configuration --------
MAX_HEIGHT       = 1024
MAX_WIDTH        = 1024
MAX_STEPS        = 20
MAX_GUIDANCE     = 10.0
ALLOWED_API_KEYS = os.getenv("API_KEYS", "client1").split(",")
WORKER_IDLE_SEC  = int(os.getenv("WORKER_IDLE_SEC", "60"))
GRPC_PORT        = 50051
HTTP_PORT        = 8000

# -------- NVML init for GPU metrics --------
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# -------- Shared Queues & Locks --------
task_queue   = mp.Queue()
result_queue = mp.Queue()
worker_proc  = None
worker_lock  = threading.Lock()
corr_counter = 0
grpc_server  = None

# -------- Helpers --------
def validate_api_key(ctx):
    md = dict(ctx.invocation_metadata() or [])
    if md.get("api-key") not in ALLOWED_API_KEYS:
        ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

def validate_params(req, ctx):
    if not (1 <= req.height <= MAX_HEIGHT):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT,
                  f"height must be 1–{MAX_HEIGHT}")
    if not (1 <= req.width  <= MAX_WIDTH):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT,
                  f"width must be 1–{MAX_WIDTH}")
    if not (1 <= req.num_inference_steps <= MAX_STEPS):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT,
                  f"num_inference_steps must be 1–{MAX_STEPS}")
    if not (0.1 <= req.guidance_scale <= MAX_GUIDANCE):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT,
                  f"guidance_scale must be 0.1–{MAX_GUIDANCE}")

def ensure_worker():
    """Spawn a worker process if none is alive."""
    global worker_proc
    if worker_proc is None or not worker_proc.is_alive():
        p = mp.Process(
            target=worker_main,
            args=(task_queue, result_queue, WORKER_IDLE_SEC),
            daemon=False
        )
        p.start()
        worker_proc = p
        logging.info(f"[Server] Spawned worker PID={p.pid}")

# -------- gRPC Servicer --------
class ImageGenServicer(image_gen_pb2_grpc.ImageGenServicer):
    def Ping(self, request, context):
        validate_api_key(context)
        return image_gen_pb2.PingResponse(message="Pong")

    def Generate(self, request, context):
        validate_api_key(context)
        validate_params(request, context)

        ensure_worker()

        # Try lock _without_ blocking
        if not worker_lock.acquire(blocking=False):
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Server busy—only one inference at a time. Please retry shortly."
            )

        try:
            global corr_counter
            cid = corr_counter
            corr_counter += 1

            task_queue.put((
                request.prompt,
                request.height,
                request.width,
                request.num_inference_steps,
                request.guidance_scale,
                cid
            ))

            start = time.time()
            while True:
                rid, img_bytes = result_queue.get()
                if rid == cid:
                    duration = time.time() - start
                    return image_gen_pb2.GenerateResponse(
                        image_png      = img_bytes or b"",
                        inference_time = duration
                    )
                else:
                    result_queue.put((rid, img_bytes))
        finally:
            worker_lock.release()

# -------- HTTP Metrics App --------
http_app = Flask("metrics")

@http_app.route("/metrics", methods=["GET"])
def metrics():
    # Note: do NOT error if worker_proc is None
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    data = {
        "cpu_percent":  psutil.cpu_percent(),
        "ram_used_mb":  psutil.virtual_memory().used // (1024*1024),
        "gpu_used_mb":  mem.used // (1024*1024),
        "worker_alive": bool(worker_proc and worker_proc.is_alive()),
        "grpc_alive":   grpc_server is not None,
    }
    return jsonify(data), 200

def run_http():
    http_app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)

# -------- Graceful Shutdown --------
def shutdown(sig, frame):
    logging.info("Signal received—shutting down")
    if grpc_server: grpc_server.stop(0)
    if worker_proc and worker_proc.is_alive():
        worker_proc.terminate()
    os._exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

# -------- Server Startup --------
def serve():
    global grpc_server
    logging.basicConfig(level=logging.INFO)

    # 1) Launch HTTP metrics in background
    threading.Thread(target=run_http, daemon=True).start()
    logging.info(f"HTTP metrics listening on 0.0.0.0:{HTTP_PORT}/metrics")

    # 2) Start gRPC
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_gen_pb2_grpc.add_ImageGenServicer_to_server(ImageGenServicer(), grpc_server)
    grpc_server.add_insecure_port(f"[::]:{GRPC_PORT}")
    logging.info(f"gRPC server listening on port {GRPC_PORT}")
    grpc_server.start()
    grpc_server.wait_for_termination()

if __name__ == "__main__":
    serve()
