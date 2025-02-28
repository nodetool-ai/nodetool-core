import asyncio
from nodetool.providers.replicate.code_generation import (
    create_replicate_namespace,
)
from nodetool.metadata.types import AudioRef, ImageRef, SVGRef, VideoRef
import argparse
import dotenv
import os

replicate_nodes_folder = os.path.dirname(os.path.abspath(__file__))


"""
This script generates source code for all replicate nodes 
using information from Replicate's model API.
"""

replicate_nodes = [
    {
        "node_name": "AdInpaint",
        "namespace": "image.generate",
        "model_id": "logerzhu/ad-inpaint",
        "return_type": ImageRef,
        "overrides": {"image_path": ImageRef},
    },
    {
        "node_name": "ConsistentCharacter",
        "model_id": "fofr/consistent-character",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"subject": ImageRef},
    },
    {
        "node_name": "PulidBase",
        "model_id": "fofr/pulid-base",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"face_image": ImageRef},
    },
    # {
    #     "model_id": "lucataco/proteus-v0.4",
    #     "node_name": "Proteus",
    #     "namespace": "image.generate",
    #     "return_type": ImageRef,
    #     "overrides": {"image": ImageRef, "mask": ImageRef},
    # },
    {
        "node_name": "SDXLClipInterrogator",
        "namespace": "image.analyze",
        "model_id": "lucataco/sdxl-clip-interrogator",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "methexis-inc/img2prompt",
        "node_name": "Img2Prompt",
        "namespace": "image.analyze",
        "overrides": {"image": ImageRef},
        "return_type": "str",
    },
    {
        "model_id": "lucataco/moondream2",
        "namespace": "image.analyze",
        "node_name": "Moondream2",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "abiruyt/text-extract-ocr",
        "node_name": "TextExtractOCR",
        "namespace": "image.ocr",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "mickeybeurskens/latex-ocr",
        "node_name": "LatexOCR",
        "namespace": "image.ocr",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "fofr/face-to-many",
        "node_name": "FaceToMany",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "BecomeImage",
        "namespace": "image.face",
        "model_id": "fofr/become-image",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "image_to_become": ImageRef},
    },
    {
        "model_id": "tencentarc/photomaker",
        "node_name": "PhotoMaker",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "tencentarc/photomaker-style",
        "node_name": "PhotoMakerStyle",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {
            "input_image": ImageRef,
            "input_image2": ImageRef,
            "input_image3": ImageRef,
            "input_image4": ImageRef,
        },
    },
    {
        "model_id": "fofr/face-to-sticker",
        "node_name": "FaceToSticker",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "zsxkib/instant-id",
        "node_name": "InstantId",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "pose_image": ImageRef},
    },
    {
        "model_id": "grandlineai/instant-id-photorealistic",
        "node_name": "Instant_ID_Photorealistic",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "grandlineai/instant-id-artistic",
        "node_name": "Instant_ID_Artistic",
        "namespace": "image.face",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "RealEsrGan",
        "namespace": "image.upscale",
        "model_id": "daanelson/real-esrgan-a100",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "tencentarc/gfpgan",
        "node_name": "GFPGAN",
        "namespace": "image.upscale",
        "return_type": ImageRef,
        "overrides": {"img": ImageRef},
    },
    {
        "node_name": "ClarityUpscaler",
        "namespace": "image.upscale",
        "model_id": "philz1337x/clarity-upscaler",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "MagicImageRefiner",
        "namespace": "image.upscale",
        "model_id": "batouresearch/magic-image-refiner",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "node_name": "ruDallE_SR",
        "namespace": "image.upscale",
        "model_id": "cjwbw/rudalle-sr",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "HighResolutionControlNetTile",
        "namespace": "image.upscale",
        "model_id": "batouresearch/high-resolution-controlnet-tile",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "UltimateSDUpscale",
        "namespace": "image.upscale",
        "model_id": "fewjative/ultimate-sd-upscale",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "CodeFormer",
        "namespace": "image.enhance",
        "model_id": "lucataco/codeformer",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "cjwbw/night-enhancement",
        "node_name": "Night_Enhancement",
        "namespace": "image.enhance",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "cjwbw/supir-v0q",
        "node_name": "Supir_V0Q",
        "namespace": "image.enhance",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "cjwbw/supir-v0f",
        "node_name": "Supir_V0F",
        "namespace": "image.enhance",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "Maxim",
        "model_id": "google-research/maxim",
        "namespace": "image.enhance",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "SwinIR",
        "namespace": "image.upscale",
        "model_id": "jingyunliang/swinir",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "Swin2SR",
        "namespace": "image.upscale",
        "model_id": "mv-lab/swin2sr",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "stability-ai/stable-diffusion",
        "node_name": "StableDiffusion",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "stability-ai/stable-diffusion-3.5-medium",
        "node_name": "StableDiffusion3_5_Medium",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "stability-ai/stable-diffusion-3.5-large",
        "node_name": "StableDiffusion3_5_Large",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "stability-ai/stable-diffusion-3.5-large-turbo",
        "node_name": "StableDiffusion3_5_Large_Turbo",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "luma/ray",
        "node_name": "Ray",
        "namespace": "video.generate",
        "return_type": VideoRef,
        "overrides": {"start_image_url": ImageRef, "end_image_url": ImageRef},
    },
    {
        "model_id": "luma/photon-flash",
        "node_name": "Photon_Flash",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {
            "image_reference_url": ImageRef,
            "style_reference_url": ImageRef,
            "character_reference_url": ImageRef,
        },
    },
    {
        "model_id": "stability-ai/sdxl",
        "node_name": "StableDiffusionXL",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "swartype/sdxl-pixar",
        "node_name": "SDXL_Pixar",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "fofr/sdxl-emoji",
        "node_name": "SDXL_Emoji",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "stability-ai/stable-diffusion-inpainting",
        "node_name": "StableDiffusionInpainting",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "ai-forever/kandinsky-2.2",
        "node_name": "Kandinsky_2_2",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-schnell",
        "node_name": "Flux_Schnell",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-dev",
        "node_name": "Flux_Dev",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-pro",
        "node_name": "Flux_Pro",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-1.1-pro-ultra",
        "node_name": "Flux_1_1_Pro_Ultra",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-dev-lora",
        "node_name": "Flux_Dev_Lora",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-schnell-lora",
        "node_name": "Flux_Schnell_Lora",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-depth-pro",
        "node_name": "Flux_Depth_Pro",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-canny-pro",
        "node_name": "Flux_Canny_Pro",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-fill-pro",
        "node_name": "Flux_Fill_Pro",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-1.1-pro-ultra",
        "node_name": "Flux_1_1_Pro_Ultra",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-pro",
        "node_name": "Flux_Pro",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "black-forest-labs/flux-depth-dev",
        "node_name": "Flux_Depth_Dev",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "bytedance/hyper-flux-8step",
        "node_name": "Hyper_Flux_8Step",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "fofr/flux-mona-lisa",
        "node_name": "Flux_Mona_Lisa",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "adirik/flux-cinestill",
        "node_name": "Flux_Cinestill",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "fofr/flux-black-light",
        "node_name": "Flux_Black_Light",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "igorriti/flux-360",
        "node_name": "Flux_360",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "recraft-ai/recraft-v3",
        "node_name": "Recraft_V3",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "recraft-ai/recraft-20b",
        "node_name": "Recraft_20B",
        "namespace": "image.generate",
        "return_type": ImageRef,
    },
    {
        "model_id": "recraft-ai/recraft-20b-svg",
        "node_name": "Recraft_20B_SVG",
        "namespace": "image.generate",
        "return_type": SVGRef,
    },
    {
        "model_id": "recraft-ai/recraft-v3-svg",
        "node_name": "Recraft_V3_SVG",
        "namespace": "image.generate",
        "return_type": SVGRef,
    },
    {
        "model_id": "black-forest-labs/flux-canny-dev",
        "node_name": "Flux_Canny_Dev",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-fill-dev",
        "node_name": "Flux_Fill_Dev",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"control_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-redux-schnell",
        "node_name": "Flux_Redux_Schnell",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"redux_image": ImageRef},
    },
    {
        "model_id": "black-forest-labs/flux-redux-dev",
        "node_name": "Flux_Redux_Dev",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"redux_image": ImageRef},
    },
    {
        "model_id": "lucataco/sdxl-controlnet",
        "node_name": "SDXL_Controlnet",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "catacolabs/sdxl-ad-inpaint",
        "node_name": "SDXL_Ad_Inpaint",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "OldPhotosRestoration",
        "namespace": "image.enhance",
        "model_id": "microsoft/bringing-old-photos-back-to-life",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "Kandinsky",
        "namespace": "image.generate",
        "model_id": "ai-forever/kandinsky-2.2",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "StableDiffusionXLLightning",
        "namespace": "image.generate",
        "model_id": "bytedance/sdxl-lightning-4step",
        "return_type": ImageRef,
    },
    {
        "node_name": "PlaygroundV2",
        "namespace": "image.generate",
        "model_id": "playgroundai/playground-v2.5-1024px-aesthetic",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "datacte/proteus-v0.2",
        "node_name": "Proteus_V_02",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "datacte/proteus-v0.3",
        "node_name": "Proteus_V_03",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "fofr/sticker-maker",
        "node_name": "StickerMaker",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "StyleTransfer",
        "model_id": "fofr/style-transfer",
        "return_type": ImageRef,
        "namespace": "image.generate",
        "overrides": {"structure_image": ImageRef, "style_image": ImageRef},
    },
    {
        "node_name": "HotshotXL",
        "namespace": "video.generate",
        "model_id": "lucataco/hotshot-xl",
        "return_type": VideoRef,
    },
    {
        "node_name": "Zeroscope_V2_XL",
        "namespace": "video.generate",
        "model_id": "anotherjesse/zeroscope-v2-xl",
        "return_type": VideoRef,
    },
    {
        "node_name": "RobustVideoMatting",
        "namespace": "video.generate",
        "model_id": "arielreplicate/robust_video_matting",
        "return_type": VideoRef,
        "overrides": {"input_video": VideoRef},
    },
    # {
    #     "node_name": "StableDiffusionInfiniteZoom",
    #     "namespace": "video.generate",
    #     "model_id": "arielreplicate/stable_diffusion_infinite_zoom",
    #     "return_type": VideoRef,
    # },
    {
        "node_name": "Illusions",
        "namespace": "image.generate",
        "model_id": "fofr/illusions",
        "return_type": ImageRef,
        "overrides": {
            "image": ImageRef,
            "control_image": ImageRef,
            "mask_image": ImageRef,
        },
    },
    {
        "model_id": "lucataco/nsfw_image_detection",
        "node_name": "NSFWImageDetection",
        "namespace": "image.analyze",
        "return_type": "str",
    },
    {
        "model_id": "salesforce/blip",
        "node_name": "Blip",
        "namespace": "image.analyze",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "andreasjansson/blip-2",
        "node_name": "Blip2",
        "namespace": "image.analyze",
        "return_type": "str",
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "pharmapsychotic/clip-interrogator",
        "node_name": "ClipInterrogator",
        "namespace": "image.analyze",
        "return_type": str,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "Llava13b",
        "namespace": "image.analyze",
        "model_id": "yorickvp/llava-13b",
        "return_type": str,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "andreasjansson/clip-features",
        "node_name": "ClipFeatures",
        "namespace": "image.analyze",
        "return_type": list[dict],
    },
    {
        "model_id": "meta/meta-llama-3-8b",
        "node_name": "Llama3_8B",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/meta-llama-3-8b-instruct",
        "node_name": "Llama3_8B_Instruct",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/meta-llama-3-70b",
        "node_name": "Llama3_70B",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/meta-llama-3-8b-instruct",
        "node_name": "Llama3_8B_Instruct",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/meta-llama-3-70b-instruct",
        "node_name": "Llama3_70B_Instruct",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/meta-llama-3.1-405b-instruct",
        "node_name": "Llama3_1_405B_Instruct",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/llama-guard-3-11b-vision",
        "node_name": "LlamaGuard_3_11B_Vision",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "meta/llama-guard-3-8b",
        "node_name": "LlamaGuard_3_8B",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "snowflake/snowflake-arctic-instruct",
        "node_name": "Snowflake_Arctic_Instruct",
        "namespace": "text.generate",
        "return_type": str,
    },
    {
        "model_id": "ryan5453/demucs",
        "node_name": "Demucs",
        "namespace": "audio.separate",
        "overrides": {"audio": AudioRef},
        "return_type": {
            "vocals": AudioRef,
            "drums": AudioRef,
            "bass": AudioRef,
            "other": AudioRef,
        },
    },
    # {
    #     "model_id": "openai/whisper",
    #     "node_name": "Whisper",
    #     "namespace": "audio.transcribe",
    #     "return_type": str,
    #     "overrides": {"audio": AudioRef},
    # },
    {
        "model_id": "vaibhavs10/incredibly-fast-whisper",
        "node_name": "IncrediblyFastWhisper",
        "namespace": "audio.transcribe",
        "return_type": str,
        "overrides": {"audio": AudioRef},
    },
    {
        "node_name": "AudioSuperResolution",
        "namespace": "audio.enhance",
        "model_id": "nateraw/audio-super-resolution",
        "return_type": AudioRef,
        "overrides": {"input_file": AudioRef},
    },
    {
        "node_name": "RemoveBackground",
        "namespace": "image.process",
        "model_id": "cjwbw/rembg",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "node_name": "ModNet",
        "namespace": "image.process",
        "model_id": "pollinations/modnet",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef},
    },
    {
        "model_id": "piddnad/ddcolor",
        "node_name": "DD_Color",
        "namespace": "image.process",
        "return_type": ImageRef,
    },
    {
        "model_id": "batouresearch/magic-style-transfer",
        "node_name": "Magic_Style_Transfer",
        "namespace": "image.process",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "ip_image": ImageRef},
    },
    {
        "model_id": "codeplugtech/object_remover",
        "node_name": "ObjectRemover",
        "namespace": "image.process",
        "return_type": ImageRef,
        "overrides": {"org_image": ImageRef, "mask_image": ImageRef},
    },
    {
        "node_name": "RealisticVoiceCloning",
        "namespace": "audio.generate",
        "model_id": "zsxkib/realistic-voice-cloning",
        "return_type": AudioRef,
        "overrides": {"song_input": AudioRef},
    },
    {
        "node_name": "TortoiseTTS",
        "model_id": "afiaka87/tortoise-tts",
        "namespace": "audio.generate",
        "return_type": AudioRef,
        "overrides": {"custom_voice": AudioRef},
    },
    {
        "model_id": "adirik/styletts2",
        "node_name": "StyleTTS2",
        "namespace": "audio.generate",
        "return_type": AudioRef,
        "overrides": {"reference": AudioRef},
    },
    {
        "node_name": "Riffusion",
        "namespace": "audio.generate",
        "model_id": "riffusion/riffusion",
        "return_type": AudioRef,
        "output_key": "audio",
        "overrides": {"song_input": AudioRef},
    },
    # {
    #     "node_name": "Bark",
    #     "namespace": "audio.generate",
    #     "model_id": "suno-ai/bark",
    #     "return_type": AudioRef,
    #     "output_key": "audio_out",
    # },
    {
        "node_name": "MusicGen",
        "namespace": "audio.generate",
        "model_id": "meta/musicgen",
        "return_type": AudioRef,
    },
    # No latest version
    # {
    #     "node_name": "VideoLlava",
    #     "namespace": "video.analyze",
    #     "model_id": "nateraw/video-llava",
    #     "return_type": str,
    #     "overrides": {"video_path": VideoRef, "image_path": ImageRef},
    # },
    {
        "node_name": "AudioToWaveform",
        "namespace": "video.generate",
        "model_id": "fofr/audio-to-waveform",
        "return_type": VideoRef,
        "overrides": {"audio": AudioRef},
    },
    {
        "model_id": "tencent/hunyuan-video",
        "node_name": "Hunyuan_Video",
        "namespace": "video.generate",
        "return_type": VideoRef,
    },
    {
        "model_id": "minimax/video-01-live",
        "node_name": "Video_01_Live",
        "namespace": "video.generate",
        "return_type": VideoRef,
    },
    {
        "model_id": "zsxkib/mmaudio",
        "node_name": "MMAudio",
        "namespace": "audio.generate",
        "return_type": AudioRef,
    },
    {
        "model_id": "minimax/video-01",
        "node_name": "Video_01",
        "namespace": "video.generate",
        "return_type": VideoRef,
    },
    {
        "model_id": "minimax/music-01",
        "node_name": "Music_01",
        "namespace": "video.generate",
        "return_type": AudioRef,
        "overrides": {
            "voice_file": AudioRef,
            "song_file": AudioRef,
            "instumental_file": AudioRef,
        },
    },
    {
        "model_id": "ideogram-ai/ideogram-v2",
        "node_name": "Ideogram_V2",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "ideogram-ai/ideogram-v2-turbo",
        "node_name": "Ideogram_V2_Turbo",
        "namespace": "image.generate",
        "return_type": ImageRef,
        "overrides": {"image": ImageRef, "mask": ImageRef},
    },
    {
        "model_id": "lightricks/ltx-video",
        "node_name": "LTX_Video",
        "namespace": "video.generate",
        "return_type": VideoRef,
        "overrides": {"image": ImageRef},
    },
]

if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", help="Specify the namespace argument")
    args = parser.parse_args()

    if args.namespace:
        nodes = []
        for node in replicate_nodes:
            if node["namespace"] == args.namespace:
                nodes.append(node)

        print(f"Creating namespace: {args.namespace}")
        asyncio.run(
            create_replicate_namespace(replicate_nodes_folder, args.namespace, nodes)
        )
    else:
        nodes_by_namespace = {}
        for node in replicate_nodes:
            if node["namespace"] not in nodes_by_namespace:
                nodes_by_namespace[node["namespace"]] = []
            nodes_by_namespace[node["namespace"]].append(node)

        for namespace, nodes in nodes_by_namespace.items():
            print(f"Creating namespace: {namespace}")
            asyncio.run(
                create_replicate_namespace(replicate_nodes_folder, namespace, nodes)
            )
