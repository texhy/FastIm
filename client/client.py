#!/usr/bin/env python3
import sys, grpc, argparse

import image_gen_pb2, image_gen_pb2_grpc

# Match server caps
MAX_HEIGHT = 1024
MAX_WIDTH  = 1024
MAX_STEPS  = 20
MAX_GUIDANCE = 10.0

def validate_args(args):
    if not (1 <= args.height <= MAX_HEIGHT):
        raise ValueError(f"height must be 1–{MAX_HEIGHT}")
    if not (1 <= args.width  <= MAX_WIDTH):
        raise ValueError(f"width must be 1–{MAX_WIDTH}")
    if not (1 <= args.steps  <= MAX_STEPS):
        raise ValueError(f"steps must be 1–{MAX_STEPS}")
    if not (0.1 <= args.guidance <= MAX_GUIDANCE):
        raise ValueError(f"guidance must be 0.1–{MAX_GUIDANCE}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--out", default="out.png")
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print("Invalid parameter:", e, file=sys.stderr)
        sys.exit(1)

    # Connect
    channel = grpc.insecure_channel(args.server)
    stub = image_gen_pb2_grpc.ImageGenStub(channel)
    metadata = [("api-key", args.api_key)]

    # Ping
    try:
        resp = stub.Ping(image_gen_pb2.PingRequest(), metadata=metadata)
        print("Server response:", resp.message)
    except grpc.RpcError as e:
        print("Ping failed:", e, file=sys.stderr)
        sys.exit(1)

    # Generate
    req = image_gen_pb2.GenerateRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance
    )
    try:
        resp = stub.Generate(req, metadata=metadata)
    except grpc.RpcError as e:
        print("Generate RPC error:", e.details(), file=sys.stderr)
        sys.exit(1)

    # Save image
    with open(args.out, "wb") as f:
        f.write(resp.image_png)
    print("Image saved to", args.out)

if __name__ == "__main__":
    main()
