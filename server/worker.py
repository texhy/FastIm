# worker.py

import time
import io
import queue
import torch
from diffusers import FluxPipeline

def worker_main(task_queue: queue.Queue, result_queue: queue.Queue, idle_timeout: int = 60):
    """
    - Lazy‐loads & warms up the pipeline on first job.
    - Shuts itself down (exits) if no new job arrives within `idle_timeout` seconds.
    - Each job is (prompt, height, width, steps, guidance, corr_id).
    - Each result is (corr_id, png_bytes or None).
    """
    # 1) On first job, we'll load & warm up
    pipe = None
    last_time = time.time()

    while True:
        try:
            prompt, h, w, steps, guidance, cid = task_queue.get(timeout=idle_timeout)
        except queue.Empty:
            # idle for too long → shutdown
            print(f"[Worker] Idle for {idle_timeout}s; exiting.")
            break

        # 2) load & warm up if needed
        if pipe is None:
            print("[Worker] Loading pipeline…")
            device_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            dtype = torch.bfloat16 if (device_props and device_props.major >= 8) else torch.float16

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=dtype,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                quantization_repo="HighCWu/FLUX.1-dev-4bit",
                device_map="balanced",
                low_cpu_mem_usage=True
            )
            # quick warmup
            _ = pipe(
                "warmup",
                height=64, width=64,
                num_inference_steps=2,
                guidance_scale=1.0,
                generator=torch.Generator("cuda").manual_seed(0)
            )
            print("[Worker] Pipeline loaded & warmed up.")

        # 3) Run inference
        try:
            gen = torch.Generator("cuda").manual_seed(0)
            image = pipe(
                prompt=prompt,
                height=h, width=w,
                num_inference_steps=steps,
                guidance_scale=guidance,
                output_type="pil",
                generator=gen
            ).images[0]

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            result_queue.put((cid, buf.getvalue()))
            print(f"[Worker] Served job {cid}")
        except Exception as e:
            print(f"[Worker] Error in job {cid}: {e}")
            result_queue.put((cid, None))

        last_time = time.time()
