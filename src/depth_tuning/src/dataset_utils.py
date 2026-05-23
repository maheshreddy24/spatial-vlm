from PIL import Image

# System message
SYSYTEM_MESSAGE = """You are an expert food and drink image extractor.
You provide structured data to visual inputs classifying them as edible food/drink or not.
As well as titling the image with a simple food/drink related caption.
Finally you extract any and all visible food/drink items to lists.
"""

# User prompt with image input as well as desired output
USER_PROMPT = """Classify the given input image into food or not and if edible food or drink items are present, extract those to a list. If no food/drink items are visible, return empty lists.

Only return valid JSON in the following form:

```json
{
  'is_food': 0, # int - 0 or 1 based on whether food/drinks are present (0 = no foods visible, 1 = foods visible)
  'image_title': '', # str - short food-related title for what foods/drinks are visible in the image, leave blank if no foods present
  'food_items': [], # list[str] - list of visible edible food item nouns
  'drink_items': [] # list[str] - list of visible edible drink item nouns
}
```
}"""

def format_data(sample):
    return {
        "messages": [

            # Message 0 - [SYSTEM] System Prompt (setting the scene)
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSYTEM_MESSAGE}]
            },

            # Message 1 - [USER] User input (image + prompt pair)
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT # Note: In a future extension, you might train the model to not require any text input and just go straight from image -> text output
                    }
                ],
            },

            # Message 2 - [MODEL] Ideal model output (e.g. our structured data format)
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output_label_json"]}]
            }
        ]
    }


# from all the samples this will return a list of images.@@@@@
def convert_message_to_list_of_images(messages: list[dict]) -> list[Image.Image]:
    image_inputs: list[Image.Image] = []

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            if not isinstance(element, dict):
                continue

            is_image_block = (
                "image" in element
                or element.get("type") == "image"
            )
            if not is_image_block:
                continue

            raw = element.get("image", element)
            if isinstance(raw, Image.Image):
                image_inputs.append(raw.convert("RGB"))

    return image_inputs


# Resolve image token ID once, robustly
# image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
# print(f"[INFO] SmolVLM2 uses the following for the image_token_id: {image_token_id}, we mask this token as it is only a placeholder in our sequence of tokens (we don't need the model to learn to predict it).")
