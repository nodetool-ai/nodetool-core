"""
Quick ComfyUI WebSocket test

- Connects to ComfyUI at `127.0.0.1:8188` (override with COMFYUI_ADDR)
- Submits a small prompt graph
- Option A: save via HTTP history/view (default)
- Option B: stream image bytes via SaveImageWebsocket

Usage:
  python scripts/test_comfyui.py

Requirements:
  pip install websocket-client pillow
"""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

try:
    import websocket  # websocket-client
except ImportError as e:
    print("Missing dependency: websocket-client. Install with `pip install websocket-client`.")
    raise


SERVER_ADDRESS = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")
CLIENT_ID = str(uuid.uuid4())
OUT_DIR = Path(os.environ.get("COMFYUI_OUT", "comfyui_outputs"))
MODE = os.environ.get("COMFYUI_MODE", "txt2img")  # txt2img | img2img
IMG_PATH = os.environ.get("COMFYUI_IMG", "")  # path to input image for img2img
IMG_STRENGTH = float(os.environ.get("COMFYUI_IMG_STRENGTH", "0.65"))  # denoise strength
# Always use WebSocket image streaming
WS_IMAGES = True


def queue_prompt(prompt: dict, prompt_id: str) -> None:
    payload = {"prompt": prompt, "client_id": CLIENT_ID, "prompt_id": prompt_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    with urllib.request.urlopen(req) as resp:
        resp.read()


def get_image(filename: str, subfolder: str, folder_type: str) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/view?{url_values}") as response:
        return response.read()


def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as response:
        return json.loads(response.read())


def upload_image(path: Path, *, folder_type: str = "input", subfolder: str = "", overwrite: bool = True) -> dict:
    """Upload an image to ComfyUI via /upload/image and return server JSON.

    Uses urllib to avoid adding requests dependency.
    """
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }

    def part(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode()
        )

    def file_part(field: str, filename: str, content: bytes, mime: str = "application/octet-stream") -> bytes:
        return (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n".encode()
            + content
            + b"\r\n"
        )

    with open(path, "rb") as fh:
        content = fh.read()

    body = b"".join(
        [
            file_part("image", path.name, content, "image/png" if path.suffix.lower() == ".png" else "image/jpeg"),
            part("type", folder_type),
            part("overwrite", "true" if overwrite else "false"),
            part("subfolder", subfolder),
            f"--{boundary}--\r\n".encode(),
        ]
    )

    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/upload/image", data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_images(ws: websocket.WebSocket, prompt: dict) -> dict[str, list[bytes]]:
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt, prompt_id)
    output_images: dict[str, list[bytes]] = {}
    current_node: str = ""
    ws_node_key = "save_image_websocket_node"

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    if data.get("node") is None:
                        break
                    else:
                        current_node = data.get("node") or ""
        else:
            # Binary frames – if using SaveImageWebsocket, the image bytes follow an 8-byte header
            if WS_IMAGES and current_node == ws_node_key:
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output
            else:
                # Ignore previews or unrelated binary frames
                continue

    # No HTTP fallback; we always stream via WS

    return output_images


def build_prompt() -> dict:
    if MODE == "img2img":
        if not IMG_PATH:
            raise SystemExit("COMFYUI_IMG is required for img2img mode")
        # Upload image via API and use returned name/subfolder/type
        up = upload_image(Path(IMG_PATH), folder_type="input", subfolder="", overwrite=True)
        image_name = up.get("name") or Path(IMG_PATH).name
        image_sub = up.get("subfolder", "")
        image_type = up.get("type", "input")
        prompt = {
            # Load image from ComfyUI input folder
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name, "subfolder": image_sub, "type": image_type},
            },
            # Encode VAE to latent
            "2": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["4", 2]},
            },
            # Sampler
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 7,
                    "denoise": IMG_STRENGTH,
                    "latent_image": ["2", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": 12345,
                    "steps": 20,
                },
            },
            # Checkpoint
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "placeholder"},
            },
            # Positive/Negative prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": ""},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": ""},
            },
            # Decode and save
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "save_image_websocket_node": (
                {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}}
                if WS_IMAGES
                else
                {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI_img2img", "images": ["8", 0]}}
            ),
        }
    else:
        # Build txt2img prompt as dict (avoid conditionals inside JSON strings)
        prompt_dict = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": 8566257,
                    "steps": 20,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": 512, "width": 512},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": "masterpiece best quality girl"},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": "bad hands"},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
        }
        if WS_IMAGES:
            # Use a fixed node key matching ComfyUI example so WS frames can be matched
            prompt_dict["save_image_websocket_node"] = {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}}
        else:
            prompt_dict["9"] = {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
            }
        prompt = prompt_dict
    # Customize prompt
    positive = os.environ.get("COMFYUI_PROMPT", "masterpiece best quality man")
    negative = os.environ.get("COMFYUI_NEG", "bad hands")
    seed = int(os.environ.get("COMFYUI_SEED", 5))
    ckpt = os.environ.get("COMFYUI_CKPT") or "dreamshaper_8.safetensors"

    prompt["6"]["inputs"]["text"] = positive
    prompt["7"]["inputs"]["text"] = negative
    if "3" in prompt:
        prompt["3"]["inputs"]["seed"] = seed
    prompt["4"]["inputs"]["ckpt_name"] = ckpt

    # If img2img, ensure image is available in ComfyUI input folder
    if MODE == "img2img":
        src = Path(IMG_PATH)
        dest = Path("/root/ComfyUI/input") / src.name
        try:
            if not dest.exists():
                dest.write_bytes(Path(IMG_PATH).read_bytes())
        except Exception as e:
            print(f"Warning: failed to copy input image to ComfyUI input folder: {e}")
    return prompt


def main() -> int:
    print(f"Connecting to ComfyUI at http://{SERVER_ADDRESS} ({MODE})")
    ws = websocket.WebSocket()
    ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")

    try:
        prompt = build_prompt()
        images = get_images(ws, prompt)
    finally:
        ws.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for node_id, imgs in images.items():
        for idx, data in enumerate(imgs):
            fname = OUT_DIR / f"{node_id}_{idx}.png"
            with open(fname, "wb") as f:
                f.write(data)
            saved += 1
            print(f"Saved: {fname}")

    if saved == 0:
        print("No images found in outputs. Check model/ckpt availability and graph.")
        return 2

    print(f"Done. Saved {saved} image(s) to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
