# streamlit_app.py

import streamlit as st
import grpc
import io
from PIL import Image

import image_gen_pb2, image_gen_pb2_grpc

# ─── 1) Sidebar: API key + prompt/settings ────────────────────
with st.sidebar:
    st.header("🔑 API Key")
    API_KEY = st.text_input(
        "Enter your API key",
        value="client1",
        type="password",
    )

    st.markdown("---")
    st.header("✏️ Prompt & Settings")
    prompt   = st.text_area(
        "Prompt",
        "A blue cat holding a sign that says hello world"
    )
    height   = st.slider("Height", 64, 1024, 512)
    width    = st.slider("Width",  64, 1024, 512)
    steps    = st.slider("Steps", 1, 20, 10)
    guidance = st.slider("Guidance Scale", 0.1, 10.0, 4.07)

    st.markdown("---")
    generate_clicked = st.button(
        "Generate Image",
        disabled=st.session_state.get("loading", False)
    )

# ─── 2) gRPC stub (public endpoint) ───────────────────────────
channel = grpc.insecure_channel("fastim.duckdns.org:50051")
stub    = image_gen_pb2_grpc.ImageGenStub(channel)

# ─── 3) Initialize session state ───────────────────────────────
if "image" not in st.session_state:
    st.session_state.image = None
if "inference_time" not in st.session_state:
    st.session_state.inference_time = 0.0
if "loading" not in st.session_state:
    st.session_state.loading = False

# ─── 4) Handle Generate click ─────────────────────────────────
if generate_clicked:
    st.session_state.loading = True
    st.session_state.image   = None
    st.session_state.inference_time = 0.0

    req = image_gen_pb2.GenerateRequest(
        prompt               = prompt,
        height               = height,
        width                = width,
        num_inference_steps  = steps,
        guidance_scale       = guidance,
    )
    try:
        resp = stub.Generate(req, metadata=[("api-key", API_KEY)])
        byte_len = len(resp.image_png)
        st.write(f"🔍 Received {byte_len} bytes of image data.")

        with open("debug_output.png", "wb") as f:
            f.write(resp.image_png)

        pil_img = Image.open(io.BytesIO(resp.image_png))
        st.session_state.image = pil_img
        st.session_state.inference_time = resp.inference_time

    except grpc.RpcError as e:
        st.error(f"Generation failed: {e.details()}")
    finally:
        st.session_state.loading = False

# ─── 5) Big preview box ────────────────────────────────────────
st.markdown("---")
preview = st.empty()
if st.session_state.loading:
    preview.markdown(
        "<div style='border:2px dashed #444; height:400px; "
        "display:flex; align-items:center; justify-content:center;'>"
        "<h3>⏳ Generating...</h3></div>",
        unsafe_allow_html=True
    )
elif st.session_state.image:
    preview.image(st.session_state.image, width=width)
else:
    preview.markdown(
        "<div style='border:2px dashed #444; height:400px; "
        "display:flex; align-items:center; justify-content:center;'>"
        "<h3>No image yet</h3></div>",
        unsafe_allow_html=True
    )

# ─── 6) Inference time below preview ──────────────────────────
st.markdown("---")
c1, _ = st.columns(2)
c1.metric("🕒 Inference Time", f"{st.session_state.inference_time:.2f}s")
