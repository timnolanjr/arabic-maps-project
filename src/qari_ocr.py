#!/usr/bin/env python3
# src/qari_ocr.py
# ------------------------------------------------------------
# Full-page Arabic OCR using NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct
#
# Usage:
#   pip install transformers qwen_vl_utils accelerate bitsandbytes
#   python src/qari_ocr.py \
#       /path/to/image_or_folder \
#       --output_folder results/ \
#       --device cuda \
#       --max_new_tokens 2000
# ------------------------------------------------------------

import os
import sys
import argparse

from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def ocr_image(model, processor, image_path, prompt, device, max_new_tokens):
    # Load and prepare the image
    image = Image.open(image_path).convert("RGB")
    # Build the multimodal chat message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    # Apply chat template and extract vision inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    vision_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=vision_inputs["image_inputs"],
        videos=vision_inputs["video_inputs"],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate the OCR output
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )
    # Trim off the prompt tokens
    trimmed = [
        out_ids[input_len:]
        for input_len, out_ids in zip(inputs.input_ids, generated)
    ]
    # Decode to plain text
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text

def main():
    parser = argparse.ArgumentParser(
        description="Run QARI-OCR (Arabic full-page OCR) on an image or folder"
    )
    parser.add_argument(
        "input",
        help="Path to an image file or a folder of images"
    )
    parser.add_argument(
        "--model_name",
        default="NAMAA-Space/Qari-OCR-0.2.2.1-Arabic-2B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., cuda or cpu)"
    )
    parser.add_argument(
        "--output_folder",
        default="./result/qari_ocr/",
        help="Where to save the .txt outputs"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Below is the image of one page of a document. "
            "Extract and return the plain Arabic text exactly as it appears, "
            "including diacritics. Do not hallucinate."
        ),
        help="Instruction prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2000,
        help="Maximum number of tokens to generate"
    )
    args = parser.parse_args()

    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Load model & processor
    print(f"Loading model {args.model_name} on {args.device} …")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if args.device.startswith("cuda") else None
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    # Gather input files
    if os.path.isdir(args.input):
        exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        images = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f.lower())[1] in exts
        ]
    elif os.path.isfile(args.input):
        images = [args.input]
    else:
        print(f"ERROR: {args.input} is not a file or folder.")
        sys.exit(1)

    # Process each image
    for img_path in images:
        print(f"→ OCR on {img_path} …")
        try:
            text = ocr_image(
                model, processor,
                img_path,
                args.prompt,
                args.device,
                args.max_new_tokens
            )
        except Exception as e:
            print(f"  ✗ Error on {img_path}: {e}")
            continue

        # Save the result
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_folder, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  ✓ Saved OCR to {out_path}\n")

if __name__ == "__main__":
    main()
