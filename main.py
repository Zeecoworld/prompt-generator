from flask import Flask, request, jsonify, render_template, url_for, Response
import redis
import replicate
from datetime import datetime
# import secrets
from difflib import SequenceMatcher
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
# from .check import validate_input
from flask_cors import CORS
import re
import os

from dotenv import load_dotenv

load_dotenv() 


app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

redis_client = redis.Redis.from_url(os.getenv('REDIS_URL'))

client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["5 per minute"],
    storage_uri=os.getenv('REDIS_URL'),
)

limiter.init_app(app) 

CORS(app, 
     origins=['https://promptgenerator.ink','https://www.promptgenerator.ink'],  # Specific origins only
     methods=['GET', 'POST'],
     allow_headers=['Content-Type', 'Authorization'])




@app.after_request
def add_csp_permissive(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self' *; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' *; "
        "style-src 'self' 'unsafe-inline' *; "
        "img-src 'self' data: *; "
        "font-src 'self' data: *; "
        "connect-src 'self' *; "
        "frame-src *; "
        "object-src 'none';"
    )
    return response


def calculate_prompt_similarity(prompt1, prompt2):
    """Calculate similarity between two prompts"""
    if prompt1.strip().lower() == prompt2.strip().lower():
        return 1.0
    
    if not prompt1 or not prompt2:
        return 0.0
    
    p1 = prompt1.lower().strip()
    p2 = prompt2.lower().strip()
    
    # Character similarity
    char_sim = SequenceMatcher(None, p1, p2).ratio()
    
    # Word overlap
    words1 = set(re.findall(r'\b\w+\b', p1))
    words2 = set(re.findall(r'\b\w+\b', p2))
    
    if not words1 and not words2:
        return char_sim
    
    word_sim = len(words1 & words2) / len(words1 | words2) if (words1 or words2) else 0.0
    
    return (char_sim + word_sim) / 2

@app.after_request
def set_security_headers(response):
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # XSS Protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Force HTTPS (only in production)
    if not app.debug:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Content Security Policy
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    # Referrer Policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response

@app.route('/ads.txt')
def ads_txt():
    ads_content = "google.com, pub-7354227353904441, DIRECT, f08c47fec0942fa0"
    return Response(ads_content, mimetype='text/plain')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/generate", methods=["POST"])
@limiter.limit("10 per minute")
def generate():
    data = request.get_json()
    required = ["topic", "aiModel", "promptStyle", "tone", "length"]

    if not data or not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    topic = data["topic"].strip()
    ai_model = data["aiModel"]
    prompt_style = data["promptStyle"]
    context = data.get("context", "").strip()
    tone = data["tone"]
    length = data["length"]

    full_prompt = f"{topic} {prompt_style} {tone} {length} {context}"

    SIMILARITY_THRESHOLD = 0.60  
    try:
        prompt_keys = redis_client.lrange("prompt_keys", 0, -1)
        
        # Check each cached prompt for similarity
        for past_prompt in prompt_keys:
            # Handle bytes from Redis
            past_prompt_str = past_prompt.decode() if isinstance(past_prompt, bytes) else past_prompt
            
            # Calculate similarity
            similarity_score = calculate_prompt_similarity(full_prompt, past_prompt_str)
            
            # Debug logging (optional - remove in production)
            print(f"Comparing:\nNew: {full_prompt}\nOld: {past_prompt_str}\nSimilarity: {similarity_score:.3f}")
            
            # Check if similarity meets threshold
            if similarity_score >= SIMILARITY_THRESHOLD:
                # Found a similar prompt, try to get cached response
                cached_response = redis_client.get(past_prompt)
                
                if cached_response:
                    # Decode bytes if necessary
                    cached_text = cached_response.decode() if isinstance(cached_response, bytes) else cached_response
                    
                    print(f"✅ Cache HIT! Using cached response (similarity: {similarity_score:.3f})")
                    return jsonify({
                        "response": cached_text,
                        "cached": True,
                        "similarity": similarity_score
                    }), 200
                else:
                    print(f"⚠️ Similar prompt found but no cached response")
        
        print("❌ No similar prompts found, generating new response...")
                    
    except Exception as e:
        print(f"Redis error during similarity check: {e}")

    # No cache hit - generate new response
    system_prompt = create_model_specific_prompt(ai_model, prompt_style, tone, length)
    main_prompt = f"{system_prompt}\n\nUser Request: Create a prompt for {topic}"
    if context:
        main_prompt += f"\n\nAdditional Context: {context}"

    max_tokens = get_max_tokens_for_length(length)
    temperature = get_temperature_for_style(prompt_style)

    try:
        output = client.run(
            "meta/meta-llama-3-8b-instruct",
            input={
                "prompt": main_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }
        )
        response_text = "".join(output).strip()
        cleaned_text = re.sub(r'\s+', ' ', response_text)

        # Cache the new response
        redis_client.set(full_prompt, cleaned_text)
        redis_client.rpush("prompt_keys", full_prompt)
        
        print(f"✅ Generated and cached new response")

        return jsonify({
            "response": cleaned_text,
            "cached": False
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Replicate error: {e}"}), 502



def create_model_specific_prompt(ai_model, prompt_style, tone, length):
    """Create a system prompt that makes the model act as the chosen AI model"""
    
    model_personas = {
        "chatgpt": {
            "identity": "You are ChatGPT, a helpful AI assistant created by OpenAI. You specialize in conversational AI and can help with a wide variety of tasks including writing, analysis, coding, and creative projects.",
            "characteristics": "You provide balanced, informative responses and ask clarifying questions when needed. You're known for being helpful, harmless, and honest."
        },
        "claude": {
            "identity": "You are Claude, an AI assistant created by Anthropic. You're thoughtful, nuanced, and excel at complex reasoning tasks, creative writing, and detailed analysis.",
            "characteristics": "You provide thorough, well-structured responses and are particularly good at understanding context and nuance. You're helpful while being honest about your limitations."
        },
        "gemini": {
            "identity": "You are Gemini, Google's multimodal AI model. You excel at reasoning, coding, creative tasks, and can work with various types of content.",
            "characteristics": "You provide comprehensive responses and are particularly strong at technical and analytical tasks. You can handle complex multi-step problems effectively."
        },
        "midjourney": {
            "identity": "You are an expert at creating Midjourney prompts. You understand the specific syntax, parameters, and artistic language that work best with Midjourney's image generation.",
            "characteristics": "You create detailed, artistic prompts with specific style references, lighting descriptions, camera angles, and technical parameters. You use artistic terminology and understand visual composition."
        },
        "stable-diffusion": {
            "identity": "You are specialized in creating Stable Diffusion prompts. You understand the model's strengths, prompt structure, and how to achieve specific artistic styles.",
            "characteristics": "You create detailed prompts with emphasis on artistic style, technical photography terms, and negative prompts when beneficial. You understand how different prompt weights and structures affect output."
        },
        "dalle": {
            "identity": "You are expert at creating DALL-E prompts. You understand OpenAI's image generation model and how to craft prompts that work well with its capabilities and safety guidelines.",
            "characteristics": "You create clear, descriptive prompts that work within DALL-E's guidelines while maximizing creative potential. You understand the model's artistic strengths and limitations."
        },
        "generic": {
            "identity": "You are a versatile AI assistant that can adapt to work with various AI models and platforms.",
            "characteristics": "You create flexible, well-structured prompts that can work across different AI systems while being clear and effective."
        }
    }
    
    # Get the model persona
    persona = model_personas.get(ai_model, model_personas["generic"])
    
    # Style-specific instructions
    style_instructions = {
        "detailed": "Create highly detailed, comprehensive prompts with specific instructions and examples.",
        "creative": "Focus on imaginative, open-ended prompts that encourage creative exploration and unique outputs.",
        "technical": "Create precise, technical prompts with specific parameters and clear, measurable objectives.",
        "conversational": "Create prompts that encourage natural, engaging dialogue and interactive responses.",
        "step-by-step": "Structure prompts to guide the AI through a logical sequence of steps or reasoning process."
    }
    
    # Tone-specific instructions
    tone_instructions = {
        "professional": "Use formal, business-appropriate language",
        "casual": "Use relaxed, friendly language",
        "formal": "Use precise, academic language",
        "friendly": "Use warm, approachable language",
        "expert": "Use specialized, authoritative language",
        "beginner": "Use simple, easy-to-understand language"
    }
    
    # Length-specific instructions
    length_instructions = {
        "short": "Create concise, focused prompts",
        "medium": "Create moderately detailed prompts",
        "long": "Create comprehensive, detailed prompts",
        "comprehensive": "Create extremely thorough, multi-faceted prompts"
    }
    
    # Build the system prompt
    system_prompt = f"""
        {persona['identity']}

        {persona['characteristics']}

        Your task is to generate a high-quality prompt based on the user's requirements. Follow these specific guidelines:

        Style: {style_instructions.get(prompt_style, 'Create effective prompts')}
        Tone: {tone_instructions.get(tone, 'Use appropriate language')}
        Length: {length_instructions.get(length, 'Provide suitable detail level')}

        When creating prompts for image generation models (Midjourney, Stable Diffusion, DALL-E), include:
        - Detailed visual descriptions
        - Artistic style references
        - Technical parameters (aspect ratios, quality settings)
        - Lighting and composition details

        When creating prompts for text-based models (ChatGPT, Claude, Gemini), include:
        - Clear task definition
        - Context and background information
        - Expected output format
        - Any specific requirements or constraints

        Generate only the prompt itself, without additional explanation unless specifically requested.
        """
    
    return system_prompt


def get_max_tokens_for_length(length):
    """Return appropriate max_tokens based on desired output length"""
    length_mapping = {
        "short": 150,
        "medium": 300,
        "long": 500,
        "comprehensive": 800
    }
    return length_mapping.get(length, 300)


def get_temperature_for_style(prompt_style):
    """Return appropriate temperature based on prompt style"""
    style_mapping = {
        "detailed": 0.3,
        "creative": 0.8,
        "technical": 0.2,
        "conversational": 0.7,
        "step-by-step": 0.4
    }
    return style_mapping.get(prompt_style, 0.7)


@app.route('/about')
def about():
    return render_template('about-us.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/press')
def press():
    return render_template('press.html')

@app.route('/security')
def security():
    return render_template('security.html')


@app.route('/help-center')
def help_center():
    return render_template('help-center.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/term-of-service')
def term_of_service():
    return render_template('term-of-service.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded", "retry_after": str(e.retry_after)}), 429


@app.route('/sitemap.xml', methods=['GET'])
def sitemap():
    pages = []
    ten_days_ago = (datetime.now()).date().isoformat()

    # Static routes (manually listed)
    static_urls = [
        {'loc': url_for('index', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('about', _external=True), 'lastmod': ten_days_ago}, 
        {'loc': url_for('privacy', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('press', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('security', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('help_center', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('contact', _external=True), 'lastmod': ten_days_ago},
        {'loc': url_for('term_of_service', _external=True), 'lastmod': ten_days_ago},
    ]
    pages.extend(static_urls)

    sitemap_xml = render_template("sitemap_template.xml", pages=pages)
    return Response(sitemap_xml, mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    return app.send_static_file('robots.txt')


if __name__ == "__main__":
    # Run the Flask app (for development only; use a WSGI server in production)
    app.run(debug=True)
