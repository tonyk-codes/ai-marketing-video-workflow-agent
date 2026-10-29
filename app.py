import base64, os, re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import fal_client as fal, streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import yaml

# App setup
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ICON_PATH = ASSETS_DIR / "icon" / "nike_icon.png"
CONFIG_PATH = BASE_DIR / "config.yml"
LOCAL_CONFIG_PATH = BASE_DIR / "config.local.yml"

DEFAULT_CONFIG = {
    "models": {
        "slogan": "erichflam-hkust/Qwen2.5-VL-7B-Instruct-NIKE-Finetuned",
        "slogan_fallback": "Qwen/Qwen2.5-7B-Instruct:together",
        "slogan_endpoint": "https://atm0kc5pzw8g9pck.us-east-1.aws.endpoints.huggingface.cloud",
        "script": "deepseek-ai/DeepSeek-V4.1-Flash",
        "script_fallback": "zai-org/GLM-4.7-Flash:novita",
        "video": "fal-ai/sora-2/image-to-video/pro",
    },
    "generation": {
        "script_temperature": 0.45,
        "script_max_tokens": 4000,
        "video_duration": 8,
        "video_resolution": "720p",
        "video_aspect_ratio": "16:9",
    },
    "credentials": {
        "HF_TOKEN": "",
        "FAL_KEY": "",
    },
}


def merge_config(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value
    return base


def load_config() -> dict:
    config = merge_config({}, deepcopy(DEFAULT_CONFIG))
    for path in (CONFIG_PATH, LOCAL_CONFIG_PATH):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Configuration file must contain a YAML mapping: {path.name}")
        merge_config(config, loaded)
    return config


CONFIG = load_config()

SLOGAN_MODEL = CONFIG["models"]["slogan"]
SLOGAN_FALLBACK_MODEL = CONFIG["models"]["slogan_fallback"]
SLOGAN_ENDPOINT = CONFIG["models"]["slogan_endpoint"]
SCRIPT_MODEL = CONFIG["models"]["script"]
SCRIPT_FALLBACK_MODEL = CONFIG["models"]["script_fallback"]
VIDEO_MODEL = CONFIG["models"]["video"]

NATIONALITIES = ["Chinese", "American", "Indian", "Indonesian", "Pakistani", "Nigerian", "Brazilian", "Bangladeshi", "Russian", "Mexican"]
NEGATIVE_DEFAULT = "blurry, low quality, artifacts, deformed, static, watermark, ugly, distorted, overexposed"
CATALOG = [
    ("air-force-1", "Nike Air Force 1 '07 LV8", "Casual Shoe"), ("acg-ultrafly", "Nike ACG Ultrafly Trail", "Trail Shoe"),
    ("vomero-plus", "Nike Vomero Plus", "Running Shoe"), ("kobe-3-protro", "Kobe III Protro", "Basketball Shoe"),
    ("tiempo-maestro", "Nike Tiempo Maestro Elite LV8", "Football Shoe"), ("sb-dunk-low", "Nike SB Dunk Low Pro Premium", "Skateboarding Shoe"),
]

try:
    st.set_page_config(page_title="AI Personalized Marketing Nike Video Advertisement Agent", layout="wide", page_icon=str(ICON_PATH) if ICON_PATH.exists() else None)
except Exception:
    st.set_page_config(page_title="AI Personalized Marketing Nike Video Advertisement Agent", layout="wide")

@dataclass(frozen=True)
class Customer:
    name: str; age: int; gender: str; nationality: str; location: str

@dataclass(frozen=True)
class Product:
    id: str; name: str; shoe_type: str

PRODUCTS = [Product(*p) for p in CATALOG]
PRODUCT_MAP = {p.name: p for p in PRODUCTS}

def secret(name: str) -> str:

    try: value = st.secrets.get(name, "")
    except Exception: value = ""
    return str(value or os.getenv(name, "") or CONFIG["credentials"].get(name, "")).strip()

HF_TOKEN = secret("HF_TOKEN")
FAL_KEY = secret("FAL_KEY")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

def data_uri(image_path: str | None) -> str | None:
    if not image_path or not (path := Path(image_path)).exists(): return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"

def first_existing(*paths: Path) -> str | None:
    return next((str(p) for p in paths if p.exists()), None)

def get_product_image(product: Product) -> str | None:
    return first_existing(
        ASSETS_DIR / f"{product.id}.png",
        ASSETS_DIR / f"{product.name}.png",
        ASSETS_DIR / f"{product.name.replace(' ', '_')}.png",
    )

def extract_text(chunk_content) -> str:
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        bits = []
        for item in chunk_content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    bits.append(text)
        return "".join(bits)
    return ""

def hf_chat_stream(model: str, messages: list[dict], max_tokens: int, *, base_url: str | None = None) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")
    client = InferenceClient(api_key=HF_TOKEN, base_url=base_url) if base_url else InferenceClient(api_key=HF_TOKEN)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )
    parts: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if not delta:
            continue
        content = getattr(delta, "content", None)
        text = extract_text(content)
        if text:
            parts.append(text)
    result = "".join(parts).strip()
    if not result:
        raise RuntimeError(f"No text returned by {model}.")
    return result

def hf_chat_once(
    model: str,
    messages: list[dict],
    max_tokens: int,
    *,
    base_url: str | None = None,
    temperature: float = 0.7,
) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing.")
    client = InferenceClient(api_key=HF_TOKEN, base_url=base_url) if base_url else InferenceClient(api_key=HF_TOKEN)
    result = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    choices = getattr(result, "choices", None) or []
    if not choices: raise RuntimeError(f"No text returned by {model}.")
    text = extract_text(getattr(choices[0].message, "content", None))
    if not text: raise RuntimeError(f"No text returned by {model}.")
    return text

def clean_slogan(text: str, customer_name: str) -> str:
    """Light post-processing to enforce the requested slogan format."""
    text = re.sub(r"[.!?]+$", "", re.sub(r"\bNike\b", "", re.sub(r"\s+", " ", text.replace("\n", " ")).strip().strip('"\''), flags=re.I).strip(" ,")).strip()
    head = re.sub(rf",?\s*{re.escape(customer_name)}$", "", text, flags=re.I).strip(" ,") if text.lower().endswith(customer_name.lower()) else text
    head = " ".join(head.split()[:10])
    return f"{head}, {customer_name}".strip(" ,") + ("" if head.endswith(customer_name) else "")

def generate_slogan_and_description(
    customer: Customer,
    product: Product,
    negative_prompt: str,
    product_image_path: str | None,
) -> tuple[str, str]:
    image = data_uri(product_image_path)

    # Slogan Generation
    slogan_prompt = f"""
You are a senior Nike global copywriter and personalization expert who creates hyper-targeted campaigns and product stories. You adapt language, tone, energy, and references to match the individual customer's profile for maximum relevance and inspiration — just like Nike does in regional campaigns, athlete spotlights, and Nike By You experiences.

Customer profile:
- Name: {customer.name} (use their first name naturally in the slogan/description when it feels empowering/motivational)
- Age: {customer.age} (tailor energy: youthful & rebellious for teens/20s, mature & disciplined for 30s+, wise & enduring for 40+)
- Gender: {customer.gender} (adapt phrasing, body references, and vibe — e.g., strength & power for men, grace & resilience for women, inclusive/empowering for non-binary)
- Nationality: {customer.nationality} (respectfully reflect cultural pride/roots when relevant)
- Location / City: {customer.location} (weave in local flavor — urban hustle in Taipei, rainy runs in Seattle, street energy in New York, tropical vibes in Miami, etc. Reference weather, culture, or city landmarks subtly if it fits naturally)

Product context:
- Shoe type: {product.shoe_type}

Core Nike style rules:
- Bold, motivational, empowering, concise — short, punchy sentences with rhythmic flow.
- Focus on performance, innovation, attitude, and personal triumph.
- Tie product features directly to how they unlock {customer.name}'s potential in their life/city/age/gender context.
- Inspirational without cheese; authentic, direct, gritty confidence.
- Avoid generic buzzwords; be real and athlete-minded.
- Do NOT use the words "Nike", "Just do it", or any specific model names.

When given a product image:
1. Analyze every visual detail deeply: colorway, materials, Swoosh placement, sole tech, silhouette, vibe (performance, street, retro, etc.), inferred sport/use-case.
2. Personalize everything to the customer's profile above — make {customer.name} feel this product was made for their journey, their city, their stage of life, their identity.

TASK:
Generate exactly ONE line that satisfies ALL of the following:

1) Slogan constraints (for {product.shoe_type}):
- One short, natural-sounding ad line.
- Maximum 5 words total BEFORE the comma (excluding the customer name).
- The line must end exactly with: , {customer.name}
- Do NOT add a period at the end.
- Do NOT use any periods, ellipses, exclamation marks, question marks, colons, semicolons, or dashes anywhere in the line.
- The only punctuation allowed in the entire line is the single comma immediately before the customer name.
- Do NOT include labels like "Slogan:", "-", "—", "." or quotes.
- Do NOT include any variables literally (use real words only in the generated text).
- Bold, motivational, empowering, concise — focus on performance, innovation, attitude, and personal triumph.
- There must be exactly one comma in the line, followed by a space and {customer.name}, with no other punctuation anywhere.

2) Chant-worthy / pre-workout vibe:
- The line should feel iconic, chant-worthy, and memorable — something {customer.name} could say to themselves before a run or workout.
- Maintain Nike-like authenticity and athlete-minded tone.

EXAMPLE 1:
Walk with power every day, {customer.name}

EXAMPLE 2:
Your daily run starts stronger, {customer.name}

Output only the final slogan line that meets all rules above (no labels, no explanations, no extra text).
""".strip()

    slogan_messages = [{"role": "user", "content": [{"type": "text", "text": slogan_prompt}]}]
    if image:
        slogan_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
    
    # Generate the slogan text from the stream
    try:
        raw_slogan = hf_chat_stream(SLOGAN_MODEL, slogan_messages, 80, base_url=SLOGAN_ENDPOINT)
    except Exception:
        slogan_messages_fallback = [{"role": "user", "content": slogan_prompt}]
        raw_slogan = hf_chat_once(SLOGAN_FALLBACK_MODEL, slogan_messages_fallback, 80)
    
    # Ensure any stray whitespace or newlines are stripped only at the very end
    # Assuming clean_slogan() is your custom function that formats the name suffix
    slogan = clean_slogan(raw_slogan.strip(), customer.name)

    # Description Generation
    description_prompt = f"""
You are a senior global copywriter and personalization expert who creates hyper-targeted campaigns and product stories. You adapt language, tone, energy, and references to match the individual customer's profile for maximum relevance and inspiration.

Your task is to write EXACTLY TWO vivid marketing sentences for a {product.shoe_type}.

Customer Profile
Name: {customer.name} (Use their first name naturally in the description to make it feel empowering and personal)

Age: {customer.age} (Tailor energy: youthful/rebellious for teens/20s, mature/disciplined for 30s+, wise/enduring for 40+)

Gender: {customer.gender} (Adapt phrasing and vibe: strength/power for men, grace/resilience for women, inclusive/empowering for non-binary)

Location: {customer.location} (Weave in local flavor, weather, city landmarks, or street energy natively)

Background: {customer.nationality} (Respectfully reflect cultural pride, heritage strength, or community roots)

Product Details
Product Name: {product.name} (Mention this exactly once in the text)

Product Type: {product.shoe_type} (Describe the performance, design, and fit based on this type)

Copywriting & Style Rules
Write in normal English with regular spaces between words. No numbered lists, hyphens, or dashes.

Output EXACTLY TWO sentences total, forming a single short paragraph without the numbering.

Tie product features directly to how they unlock {customer.name}'s potential in their specific life, city, age, and gender context. Highlight features → benefits → emotional payoff.

Keep it bold, motivational, empowering, and concise. Be authentic and athlete-minded; avoid generic buzzwords or cheesy phrases.

Strict Negative Constraints
- DO NOT use specific brand names (like Nike, Adidas) or product model names other than the provided {product.name}.
- DO NOT output any introductory text, explanations, or meta-commentary.
- DO NOT append any tone guidelines, instructions, or trailing text at the end of your response.
- DO NOT end the description with "- Use at least one benefit of the product in your response."

Example format:
Designed for your relentless pace, {customer.name}, the {product.name} {product.shoe_type} offers unmatched breathability and a responsive midsole for those humid Miami runs. Whether you are sprinting through the city streets or pushing your limits, this sleek design ensures a locked-in fit that matches your unyielding strength.
""".strip()

    description_messages = [{"role": "user", "content": [{"type": "text", "text": description_prompt}]}]
    if image:
        description_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
    
    # Generate the description text from the stream
    try:
        raw_description = hf_chat_stream(SLOGAN_MODEL, description_messages, 180, base_url=SLOGAN_ENDPOINT)
    except Exception:
        description_messages_fallback = [{"role": "user", "content": description_prompt}]
        raw_description = hf_chat_once("Qwen/Qwen2.5-7B-Instruct:together", description_messages_fallback, 180)
    
    # Strip whitespace only on the final assembled string
    description = raw_description.strip()
    
    return slogan, description

def generate_cinematic_script(
    customer: Customer,
    product: Product,
    product_description: str,
    slogan: str,
    negative_prompt: str,
) -> str:
    system_prompt = f"""
You are a commercial film director, sports cinematographer, and expert image-to-video prompt engineer.
Write one production-ready prompt for an 8-second premium sports advertisement. The video model will receive
the product image separately as its visual reference, so describe motion and cinematography precisely without
inventing a different shoe. The output must be vivid, physically plausible, and directly executable.

Use exactly this structure and no introduction or closing commentary:
[Creative Direction]:
[0.0-1.0s Opening Detail]:
[1.0-2.5s Hero Introduction]:
[2.5-4.5s Performance Action]:
[4.5-6.5s Hero Moment]:
[6.5-8.0s Product End Frame]:
[Camera and Lens]:
[Lighting and Color]:
[Environment and Sound]:
[On-Screen Text]:
[Negative Constraints]:

Creative requirements:
- Build one continuous visual story with motivated cuts and clear time progression.
- Feature a believable {customer.age}-year-old {customer.gender} hero from {customer.nationality} in {customer.location}; use respectful, subtle cultural and environmental cues rather than stereotypes.
- Keep the hero in most shots, show a clear face at least once, and show the {product.shoe_type} clearly on-foot during movement, contact, and landing.
- Preserve the same hero, wardrobe, shoe identity, proportions, lighting direction, and environment across every shot.
- Specify shot size, camera movement, lens character, focus pull, shutter-like motion feel, and transitions where useful.
- Use tactile details: foot strike, fabric movement, breath, surface texture, dust or rain only when motivated by the setting.
- Make the action achievable in real physics. Avoid impossible body movement, floating feet, warped soles, extra limbs, or sudden wardrobe changes.
- Create a strong visual crescendo, then leave enough stillness for the final product frame and slogan to read.
- The final frame must display exactly this slogan and no other text: {slogan}
- Do not mention the product model name or invent product specifications.
- Avoid: {negative_prompt}
""".strip()

    user_prompt = f"""
Target customer: {customer.age}, {customer.gender}, {customer.nationality}, located in {customer.location}
Shoe type: {product.shoe_type}
Product description: {product_description}
Generate the full script now.
""".strip()

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        return hf_chat_once(
            SCRIPT_MODEL,
            messages,
            int(CONFIG["generation"]["script_max_tokens"]),
            temperature=float(CONFIG["generation"]["script_temperature"]),
        )
    except Exception:
        return hf_chat_once(
            SCRIPT_FALLBACK_MODEL,
            messages,
            int(CONFIG["generation"]["script_max_tokens"]),
            temperature=float(CONFIG["generation"]["script_temperature"]),
        )

def normalize_video_output(output) -> str | None:
    if output is None:
        return None
    if isinstance(output, (str, Path)):
        value = str(output)
        return value if value.startswith(("http://", "https://")) or Path(value).exists() else None
    if hasattr(output, "url") and getattr(output, "url"):
        return output.url
    if isinstance(output, dict):
        for scope in (output, output.get("data", {}), output.get("video", {}), output.get("data", {}).get("video", {})):
            if isinstance(scope, dict):
                for key in ("url", "video_url", "file", "path"):
                    value = scope.get(key)
                    if isinstance(value, str) and value:
                        return value
    return None

def generate_video(product_image_path: str | None, cinematic_script: str, slogan: str) -> str:
    if not product_image_path or not Path(product_image_path).exists():
        raise RuntimeError("Product image not found for Pipeline 3.")
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing.")

    image_url = fal.upload_file(Path(product_image_path))
    prompt = (
        "Use the provided product image as the visual reference.\n"
        "Follow this script closely and keep the shoe prominent.\n"
        f"{cinematic_script}\n"
        f'End the video with the exact on-screen slogan "{slogan}", presented elegantly in a stylish, cinematic composition.\n'
    )

    result = fal.subscribe(
        VIDEO_MODEL,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "duration": int(CONFIG["generation"]["video_duration"]),
            "resolution": CONFIG["generation"]["video_resolution"],
            "aspect_ratio": CONFIG["generation"]["video_aspect_ratio"],
        },
        with_logs=True,
    )
    video = normalize_video_output(result)
    if not video:
        raise RuntimeError(f"No usable video source returned by {VIDEO_MODEL}.")
    return video

def playable(video_source: str | None) -> bool:
    return bool(video_source and (video_source.startswith(("http://", "https://")) or Path(video_source).exists()))

def app_style() -> None:
    st.markdown("""<style>
        .stApp {background:#0e1117;color:#fff;} .stApp header {background:transparent;}
        [data-testid="stSidebar"] {background:#262730;color:#fff;}
        h1,h2,h3,h4,h5,h6,p,label,.stMarkdown,span {color:#fff !important;}
        .stTextInput>div>div>input,.stNumberInput>div>div>input,.stTextArea textarea,div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {background:#1e1e1e !important;color:#fff !important;border-color:#444 !important;}
        [data-testid="stSidebar"] .stNumberInput input,[data-testid="stSidebar"] .stTextArea textarea {background:#000 !important;color:#fff !important;border:1px solid #444 !important;}
        .stButton>button {background:#ff4b4b !important;color:#fff !important;border:none !important;} .stButton>button:hover {background:#ff6b6b !important;}
        [data-testid="stAlert"] {background:#1e1e1e !important;color:#fff !important;border:1px solid #444 !important;}
        </style>""", unsafe_allow_html=True)

def main() -> None:
    app_style()
    st.title("AI Marketing Video Workflow Agent")
    st.caption("Create personalized campaign copy, a cinematic shot plan, and an image-to-video advertisement locally.")

    with st.sidebar:
        st.header("Customer Profile")
        name = st.text_input("Name", "Alex")
        age = st.number_input("Age", 18, 90, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        nationality = st.selectbox("Nationality", NATIONALITIES, index=0)
        location = st.selectbox("Location", ["Hong Kong", "Los Angeles", "London"], index=0)

        st.header("Generation Config")
        negative_prompt = st.text_area("Negative Prompt for Video", NEGATIVE_DEFAULT, height=100)
        
        st.header("Product Selection")
        product_name = st.selectbox("Product", list(PRODUCT_MAP))
        product = PRODUCT_MAP[str(product_name)] if product_name else list(PRODUCT_MAP.values())[0]
        product_image = get_product_image(product)
        if product_image:
            st.image(product_image, use_container_width=True)
        else:
            st.info(f"No image found for {product.name}")

        st.caption(f"Script model: `{SCRIPT_MODEL}`")
        run = st.button("Generate Campaign", type="primary", use_container_width=True)

    if not (run and name):
        st.info("Complete the profile, choose a product, and start the campaign workflow from the sidebar.")
        return

    customer = Customer(str(name), int(age), str(gender), str(nationality), str(location))
    progress = st.progress(0, "Initiating pipeline...")
    try:
        with st.spinner("Writing personalized campaign copy..."):
            slogan, description = generate_slogan_and_description(customer, product, negative_prompt, product_image)
        progress.progress(25, "Stage 1 of 3 complete: campaign copy")

        with st.spinner("Directing the cinematic shot plan..."):
            script = generate_cinematic_script(customer, product, description, slogan, negative_prompt)
        progress.progress(50, "Stage 2 of 3 complete: cinematic script")

        with st.spinner("Rendering the image-to-video advertisement..."):
            video = generate_video(product_image, script, slogan)
        progress.progress(100, "Workflow complete")
    except Exception as error:
        progress.empty()
        st.error(f"Workflow stopped: {error}")
        st.caption("Check your configured API keys, model identifiers, product image, and provider availability.")
        return

    copy_tab, script_tab, video_tab = st.tabs(["Campaign Copy", "Cinematic Script", "Video"])
    with copy_tab:
        st.subheader(slogan)
        st.write(description)
        st.download_button("Download copy", f"{slogan}\n\n{description}\n", "campaign_copy.txt", "text/plain")
        st.caption(f"Copy model: {SLOGAN_MODEL}")
    with script_tab:
        st.code(script, language="text")
        st.download_button("Download script", script, "cinematic_script.txt", "text/plain")
        st.caption(f"Script model: {SCRIPT_MODEL}")
    with video_tab:
        if playable(video):
            st.video(video)
            if video.startswith(("http://", "https://")):
                st.link_button("Open video source", video)
        else:
            st.warning("The provider returned no playable video source.")
        st.caption(f"Video model: {VIDEO_MODEL}")

if __name__ == "__main__":
    main()
