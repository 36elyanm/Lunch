import os
import io
import json
import base64
import anthropic
import PIL.Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a nutritionist and meal-prep expert. Your job is to:
1. Identify all visible ingredients in the fridge photo
2. Assess the lunchbox size and capacity from the lunchbox photo
3. Generate practical, balanced lunch meal ideas using ONLY the available ingredients

NUTRITION REQUIREMENTS (mandatory for every meal suggestion):
- PROTEIN: Must include a clear protein source (meat, eggs, legumes, dairy, tofu, etc.)
- CARBOHYDRATES: Must include a carb source (bread, rice, pasta, vegetables, fruit, etc.)
- HEALTHY FAT: Must include healthy fat (olive oil, avocado, nuts, cheese, etc.)
- NO TRANS FAT: Absolutely no partially hydrogenated oils, margarine, or processed foods containing trans fats

PERSONAL PREFERENCES: You MUST strictly respect the user's stated allergies, dislikes, and preferences.
- ALLERGIES are hard constraints — never include an allergen even as a trace ingredient
- DISLIKES should be avoided entirely
- LIKES should be prioritised where possible

FORMAT your response as JSON with this exact structure:
{
  "fridge_ingredients": ["list", "of", "identified", "ingredients"],
  "lunchbox_capacity": "description of lunchbox size (e.g., small bento, large container, standard sandwich box)",
  "meals": [
    {
      "name": "Meal Name",
      "description": "Brief appealing description",
      "ingredients": ["ingredient 1", "ingredient 2"],
      "assembly": "Simple step-by-step instructions",
      "nutrition": {
        "protein": "what provides protein and estimated amount",
        "carbs": "what provides carbs and estimated amount",
        "fat": "what provides healthy fat and estimated amount",
        "calories": "estimated total calories"
      },
      "fits_lunchbox": true,
      "prep_time": "X minutes"
    }
  ],
  "tips": ["optional nutrition or prep tip 1", "optional tip 2"]
}

Provide 3 meal suggestions. Be creative but realistic with what's visible in the fridge.
If you cannot clearly see certain ingredients, only use ones you are confident about.
Make sure every meal genuinely fits the assessed lunchbox size.
Return JSON only — no markdown, no explanation outside the JSON."""


def encode_image(image_file) -> tuple[str, str]:
    """Encode uploaded image to base64, resizing to stay under Claude's 5MB limit."""
    data = image_file.read()
    img = PIL.Image.open(io.BytesIO(data))

    # Convert to RGB for JPEG compatibility
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    # Resize so longest side is at most 1568px (Claude recommended max)
    max_side = 1568
    if img.width > max_side or img.height > max_side:
        img.thumbnail((max_side, max_side), PIL.Image.LANCZOS)

    # Compress to JPEG, reducing quality until under 3.9MB raw (~5MB base64)
    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= 3_900_000 or quality <= 30:
            break
        quality -= 10

    b64 = base64.standard_b64encode(data).decode("utf-8")
    return b64, "image/jpeg"


def build_preferences_text(allergies: str, likes: str, dislikes: str, brands: str, country: str) -> str:
    """Build a preferences block to inject into the user message."""
    lines = []
    if country.strip():
        lines.append(
            f"CUISINE STYLE: I am from / eating in the style of {country.strip()}. "
            "Please draw inspiration from that country's traditional ingredients, spice profiles, "
            "and typical lunch dishes. Adapt the meal names, seasoning, and presentation to reflect "
            "local culinary culture while still using the ingredients visible in my fridge."
        )
    if brands.strip():
        lines.append(
            f"BRANDS IN MY FRIDGE: {brands.strip()}. "
            "Use these specific brand names when referencing those products in ingredient lists "
            "and use their known nutritional profiles for more accurate calorie and macro estimates."
        )
    if allergies.strip():
        lines.append(f"ALLERGIES (MUST avoid completely): {allergies.strip()}")
    if dislikes.strip():
        lines.append(f"DISLIKES (please avoid): {dislikes.strip()}")
    if likes.strip():
        lines.append(f"LIKES (prioritise if possible): {likes.strip()}")
    if not lines:
        return ""
    return "\n\nMy personal preferences:\n" + "\n".join(lines)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "fridge" not in request.files or "lunchbox" not in request.files:
        return jsonify({"error": "Both fridge and lunchbox images are required."}), 400

    fridge_file = request.files["fridge"]
    lunchbox_file = request.files["lunchbox"]

    if fridge_file.filename == "" or lunchbox_file.filename == "":
        return jsonify({"error": "Please select both images before submitting."}), 400

    # Preferences (all optional)
    allergies = request.form.get("allergies", "")
    likes = request.form.get("likes", "")
    dislikes = request.form.get("dislikes", "")
    brands = request.form.get("brands", "")
    country = request.form.get("country", "").strip()
    prefs_text = build_preferences_text(allergies, likes, dislikes, brands, country)

    try:
        fridge_b64, fridge_media = encode_image(fridge_file)
        lunchbox_b64, lunchbox_media = encode_image(lunchbox_file)
    except Exception:
        return jsonify({"error": "Failed to process images. Please try again."}), 400

    user_message = (
        "Please analyze both images and generate 3 balanced lunch meal ideas "
        "that I can make with the ingredients in my fridge and that will fit "
        "in my lunchbox. Each meal MUST include protein, carbohydrates, and "
        "healthy fat (no trans fat)."
        + prefs_text
        + "\n\nReturn JSON only."
    )

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 10000},
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is a photo of my fridge:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": fridge_media,
                                "data": fridge_b64,
                            },
                        },
                        {"type": "text", "text": "And here is my lunchbox:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": lunchbox_media,
                                "data": lunchbox_b64,
                            },
                        },
                        {"type": "text", "text": user_message},
                    ],
                }
            ],
        )

        # Extract text from response (skip thinking blocks)
        result_text = ""
        for block in response.content:
            if block.type == "text":
                result_text = block.text.strip()
                break

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        meal_data = json.loads(result_text)
        return jsonify({"success": True, "data": meal_data})

    except Exception as e:
        err = str(e)
        if "API_KEY" in err or "api key" in err.lower() or "authentication" in err.lower():
            return jsonify({"error": "Invalid API key. Please check your ANTHROPIC_API_KEY."}), 401
        if "quota" in err.lower() or "rate" in err.lower() or "overloaded" in err.lower():
            return jsonify({"error": "Rate limit reached. Please wait a moment and try again."}), 429
        if "credit" in err.lower() or "billing" in err.lower():
            return jsonify({"error": "Insufficient credits. Please check your Anthropic account."}), 402
        return jsonify({"error": f"Analysis failed: {err}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
