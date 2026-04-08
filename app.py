import os
import json
import io
import PIL.Image
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

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


def decode_image(image_file) -> PIL.Image.Image:
    """Decode uploaded image file to PIL Image."""
    data = image_file.read()
    return PIL.Image.open(io.BytesIO(data))


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
        fridge_img = decode_image(fridge_file)
        lunchbox_img = decode_image(lunchbox_file)
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
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )

        response = model.generate_content([
            "Here is a photo of my fridge:",
            fridge_img,
            "And here is my lunchbox:",
            lunchbox_img,
            user_message,
        ])

        result_text = response.text.strip()

        # Strip markdown code fences if present
        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        meal_data = json.loads(result_text)
        return jsonify({"success": True, "data": meal_data})

    except Exception as e:
        err = str(e)
        if "API_KEY" in err or "api key" in err.lower():
            return jsonify({"error": "Invalid API key. Please check your GEMINI_API_KEY."}), 401
        if "quota" in err.lower() or "rate" in err.lower():
            return jsonify({"error": "Rate limit reached. Please wait a moment and try again."}), 429
        return jsonify({"error": f"Analysis failed: {err}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
