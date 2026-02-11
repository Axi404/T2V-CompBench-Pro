import os
import torch


def load_qwen3_model(model_path: str):
    """Load a Qwen3-VL model and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[INFO] Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    except (ValueError, ImportError):
        print("[WARN] flash_attention_2 unavailable, using sdpa")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="sdpa",
            device_map="auto",
        )

    cls_name = type(model).__name__.lower()
    model_type = getattr(model.config, "model_type", "").lower()
    if "qwen3" not in cls_name and "qwen3" not in model_type:
        raise ValueError(
            f"Only Qwen3-VL models are supported, got class={type(model).__name__}, model_type={model_type}"
        )

    print(f"[INFO] Model loaded. dtype={model.dtype}")
    return model, processor


def generate_with_messages(
    model,
    processor,
    messages: list,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float | None = None,
    num_beams: int | None = 1,
) -> str:
    """Run a single generation step for chat messages."""
    do_sample = temperature > 0
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if num_beams is not None:
        gen_kwargs["num_beams"] = num_beams

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def image_message(image_path: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": os.path.abspath(image_path)},
            {"type": "text", "text": text},
        ],
    }


def text_message(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_message(text: str) -> dict:
    return {"role": "assistant", "content": text}
