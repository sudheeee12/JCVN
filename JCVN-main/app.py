from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import json
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output
import io
import base64
import re
import zipfile
import logging
from werkzeug.utils import secure_filename
from typing import Dict, List, Tuple, Optional, Any
import mimetypes
import os
import hashlib
from web3 import Web3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-super-secret-key-for-development')
app.static_folder = 'static'

def parse_env_set(key, default=None):
    val = os.environ.get(key)
    if val is None:
        return default
    return set([v.strip() if v.strip().startswith('.') else '.'+v.strip() if not v.strip().startswith('.') else v.strip() for v in val.split(',')])

class Config:
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))
    ALLOWED_IMAGE_EXTENSIONS = parse_env_set('ALLOWED_IMAGE_EXTENSIONS', {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'})
    ALLOWED_TEMPLATE_EXTENSIONS = parse_env_set('ALLOWED_TEMPLATE_EXTENSIONS', {'.json'})
    DEFAULT_PSM_MODE = int(os.environ.get('DEFAULT_PSM_MODE', 3))
    DEFAULT_SCALE_FACTOR = int(os.environ.get('DEFAULT_SCALE_FACTOR', 2))
    DEFAULT_OEM_MODE = int(os.environ.get('DEFAULT_OEM_MODE', 3))
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 4))
    WORD_INTERSECTION_THRESHOLD = float(os.environ.get('WORD_INTERSECTION_THRESHOLD', 0.1))
    TEMPLATE_STORAGE_PATH = os.environ.get('TEMPLATE_STORAGE_PATH', 'stored_templates')

app.config.from_object(Config)
os.makedirs(app.config['TEMPLATE_STORAGE_PATH'], exist_ok=True)


class BlockchainConfig:
    RPC_URL = os.environ.get('RPC_URL')
    CONTRACT_ADDRESS = os.environ.get('CONTRACT_ADDRESS')
    SIGNER_PRIVATE_KEY = os.environ.get('SIGNER_PRIVATE_KEY')
    ABI_PATH = os.environ.get('ABI_PATH')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- User Management & RBAC ---
# In a real app, use a database. For this demo, a dictionary is sufficient.
USERS = {
    'university_user': {
        'pw_hash': generate_password_hash('uni123'),
        'role': 'university'
    },
    'verifier_user': {
        'pw_hash': generate_password_hash('ver123'),
        'role': 'verifier'
    }
}

def login_required(role="ANY"):
    """Decorator to require login and optionally a specific role."""
    def wrapper(fn):
        @wraps(fn)
        def decorated_view(*args, **kwargs):
            if 'username' not in session:
                return redirect(url_for('login'))
            if role != "ANY" and session.get('role') != role:
                return "<h1>Forbidden</h1><p>You don't have permission to access this page.</p>", 403
            return fn(*args, **kwargs)
        return decorated_view
    return wrapper

try:
    w3 = Web3(Web3.HTTPProvider(BlockchainConfig.RPC_URL))
    if not w3.is_connected():
        logger.error("Could not connect to the blockchain node.")
        w3 = None
    with open(BlockchainConfig.ABI_PATH) as f:
        contract_abi = json.load(f)['abi']
    contract_address = Web3.to_checksum_address(BlockchainConfig.CONTRACT_ADDRESS)
    merkle_contract = w3.eth.contract(address=contract_address, abi=contract_abi)
    signer_account = w3.eth.account.from_key(BlockchainConfig.SIGNER_PRIVATE_KEY)
except FileNotFoundError:
    logger.error(f"Contract ABI file not found at: {BlockchainConfig.ABI_PATH}")
    merkle_contract = None
except Exception as e:
    logger.error(f"Error setting up blockchain connection: {e}")
    merkle_contract = None


# --- Validation Functions ---

def validate_file_size(file_storage):
    """Validates file size against configured maximum."""
    if hasattr(file_storage, 'content_length') and file_storage.content_length:
        if file_storage.content_length > app.config['MAX_FILE_SIZE']:
            return False, f"File size exceeds maximum allowed size of {app.config['MAX_FILE_SIZE'] // (1024*1024)}MB"
    return True, None

def validate_file_extension(filename: str, allowed_extensions: set) -> Tuple[bool, Optional[str]]:
    """Validates file extension against allowed extensions."""
    if not filename:
        return False, "No filename provided"
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File type '{ext}' not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, None

def validate_image_file(file_storage) -> Tuple[bool, Optional[str]]:
    """Comprehensive validation for image files."""
    if not file_storage or not file_storage.filename:
        return False, "No file provided"
    
    valid_ext, ext_error = validate_file_extension(file_storage.filename, app.config['ALLOWED_IMAGE_EXTENSIONS'])
    if not valid_ext:
        return False, ext_error
    
    valid_size, size_error = validate_file_size(file_storage)
    if not valid_size:
        return False, size_error
    
    file_storage.seek(0)
    try:
        Image.open(file_storage).verify()
        file_storage.seek(0)
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def validate_template_data(template_data: Dict) -> Tuple[bool, Optional[str]]:
    """Validates template data from a dictionary."""
    try:
        if not isinstance(template_data, dict):
            return False, "Template must be a JSON object"
        
        if 'fields' not in template_data:
            return False, "Template must contain 'fields' array"
        
        fields = template_data['fields']
        if not isinstance(fields, list):
            return False, "'fields' must be an array"
        
        valid_types = {'multi', 'single_line', 'number', 'date', 'single_word'}
        for i, field in enumerate(fields):
            if not isinstance(field, dict):
                return False, f"Field {i} must be an object"
            
            if 'name' not in field or not isinstance(field['name'], str):
                return False, f"Field {i} must have a string 'name'"
            
            if 'coords' not in field or not isinstance(field['coords'], list) or len(field['coords']) != 4:
                return False, f"Field {i} must have 'coords' as array of 4 numbers [x1, y1, x2, y2]"
            
            coords = field['coords']
            try:
                coords = [float(c) for c in coords]
                if coords[0] >= coords[2] or coords[1] >= coords[3]:
                    return False, f"Field {i} coordinates invalid: x1 < x2 and y1 < y2 required"
            except (ValueError, TypeError):
                return False, f"Field {i} coordinates must be numbers"
            
            field_type = field.get('type', 'multi')
            if field_type not in valid_types:
                return False, f"Field {i} type '{field_type}' invalid. Valid types: {valid_types}"
        
        return True, None
    except Exception as e:
        return False, f"Error validating template data: {str(e)}"

def validate_template_file(file_storage) -> Tuple[bool, Optional[str], Optional[Dict]]:
    """Validates and parses template JSON file."""
    if not file_storage or not file_storage.filename:
        return False, "No template file provided", None
    
    valid_ext, ext_error = validate_file_extension(file_storage.filename, app.config['ALLOWED_TEMPLATE_EXTENSIONS'])
    if not valid_ext:
        return False, ext_error, None
    
    valid_size, size_error = validate_file_size(file_storage)
    if not valid_size:
        return False, size_error, None
    
    try:
        file_storage.seek(0)
        template_data = json.load(file_storage)
        valid, error = validate_template_data(template_data)
        if not valid:
            return False, error, None
        return True, None, template_data
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}", None
    except Exception as e:
        return False, f"Error validating template: {str(e)}", None

# --- Helper Functions for OCR ---

def clean_ocr_text(text: str, field_type: str = 'multi') -> str:
    """Cleans the extracted OCR text based on the field type."""
    if not text:
        return ''
    
    text = text.replace("'", "'").replace("'", "'").replace("\"", '"').replace("\"", '"').strip()
    text = re.sub(r'[\|\©\s]+$', '', text)
    
    if field_type == 'number':
        text = re.sub(r'[^0-9.\-]', '', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\-+', '-', text)
    elif field_type == 'date':
        text = re.sub(r'[^0-9\-/.]', '', text)
    elif field_type == 'single_word':
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
    elif field_type == 'single_line':
        text = re.sub(r'\s+', ' ', text)
    else:  # multi
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def preprocess_for_ocr(image: Image.Image, scale_factor: int = None) -> Image.Image:
    """Pre-processes an image for better OCR results."""
    if scale_factor is None:
        scale_factor = app.config['DEFAULT_SCALE_FACTOR']
    
    processed_image = image.convert('L')
    processed_image = ImageOps.invert(processed_image)
    
    width, height = processed_image.size
    new_size = (width * scale_factor, height * scale_factor)
    processed_image = processed_image.resize(new_size, Image.LANCZOS)
    
    return processed_image

def calculate_intersection_ratio(word_box: Dict, field_rect: Dict) -> float:
    """Calculate the ratio of intersection between word box and field rectangle."""
    left = max(word_box['left'], field_rect['left'])
    top = max(word_box['top'], field_rect['top'])
    right = min(word_box['left'] + word_box['width'], field_rect['left'] + field_rect['width'])
    bottom = min(word_box['top'] + word_box['height'], field_rect['top'] + field_rect['height'])
    
    if left >= right or top >= bottom:
        return 0.0
    
    intersection_area = (right - left) * (bottom - top)
    word_area = word_box['width'] * word_box['height']
    
    return intersection_area / word_area if word_area > 0 else 0.0

def words_intersect_flexible(word_box: Dict, field_rect: Dict, threshold: float = None) -> bool:
    """Checks if a word 'touches' or intersects with a field rectangle."""
    if threshold is None:
        threshold = app.config['WORD_INTERSECTION_THRESHOLD']
    
    intersection_ratio = calculate_intersection_ratio(word_box, field_rect)
    return intersection_ratio >= threshold

TYPE_CONFIG = {
    'multi': {'cleaner': 'multi'},
    'single_line': {'cleaner': 'single_line'},
    'number': {'cleaner': 'number'},
    'date': {'cleaner': 'date'},
    'single_word': {'cleaner': 'single_word'}
}

def process_image(image_file, template_data: Dict, debug: bool, ocr_config: Dict) -> Tuple[Dict, List]:
    """Processes a single image for OCR extraction."""
    try:
        logger.info(f"Processing image: {image_file.filename}")
        
        valid_image, error_msg = validate_image_file(image_file)
        if not valid_image:
            return {'image_name': image_file.filename, 'error': error_msg}, []
        
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
        fields = template_data.get('fields', [])

        scale_factor = ocr_config.get('scale_factor', app.config['DEFAULT_SCALE_FACTOR'])
        psm_mode = ocr_config.get('psm_mode', app.config['DEFAULT_PSM_MODE'])
        oem_mode = ocr_config.get('oem_mode', app.config['DEFAULT_OEM_MODE'])

        processed_image = preprocess_for_ocr(image, scale_factor)
        custom_config = f'--oem {oem_mode} --psm {psm_mode}'

        logger.info(f"Running OCR with config: {custom_config}")
        data = pytesseract.image_to_data(processed_image, output_type=Output.DICT, config=custom_config)

        words = []
        for i in range(len(data['level'])):
            if data['level'][i] == 5 and data['text'][i].strip():
                confidence = data['conf'][i]
                if confidence > 0:
                    words.append({
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'text': data['text'][i],
                        'confidence': confidence
                    })

        logger.info(f"Extracted {len(words)} words from {image_file.filename}")

        image_result = {}
        image_debug_files = []

        for field in fields:
            name = field['name']
            coords = field['coords']
            field_type = field.get('type', 'multi')

            logger.info(f"Processing field: {name} ({field_type})")

            scaled_coords = [c * scale_factor for c in coords]
            field_rect = {
                'left': scaled_coords[0],
                'top': scaled_coords[1],
                'width': scaled_coords[2] - scaled_coords[0],
                'height': scaled_coords[3] - scaled_coords[1]
            }

            intersecting_words = [w for w in words if words_intersect_flexible(w, field_rect)]
            logger.info(f"Found {len(intersecting_words)} intersecting words for field {name}")

            if not intersecting_words:
                image_result[name] = ''
                continue

            intersecting_words.sort(key=lambda w: (w['top'], w['left']))

            lines = []
            if intersecting_words:
                current_line = [intersecting_words[0]]
                avg_height = sum(w['height'] for w in intersecting_words) / len(intersecting_words)
                for w in intersecting_words[1:]:
                    if abs(w['top'] - current_line[-1]['top']) < avg_height * 0.5:
                        current_line.append(w)
                    else:
                        lines.append(current_line)
                        current_line = [w]
                lines.append(current_line)

            text_lines = []
            for line in lines:
                line.sort(key=lambda w: w['left'])
                line_text = ' '.join(w['text'] for w in line)
                text_lines.append(line_text)
            
            full_text = '\n'.join(text_lines) if field_type == 'multi' else ' '.join(text_lines)

            config = TYPE_CONFIG.get(field_type, TYPE_CONFIG['multi'])
            cleaned_text = clean_ocr_text(full_text, config['cleaner'])
            image_result[name] = cleaned_text

            logger.info(f"Field {name} result: '{cleaned_text[:50]}...'")

            if debug:
                margin = 20 * scale_factor
                crop_left = max(0, int(field_rect['left'] - margin))
                crop_top = max(0, int(field_rect['top'] - margin))
                crop_right = min(processed_image.width, int(field_rect['left'] + field_rect['width'] + margin))
                crop_bottom = min(processed_image.height, int(field_rect['top'] + field_rect['height'] + margin))
                
                debug_crop = processed_image.crop((crop_left, crop_top, crop_right, crop_bottom))
                img_io = io.BytesIO()
                debug_crop.save(img_io, 'PNG')
                img_io.seek(0)
                
                safe_filename = secure_filename(image_file.filename)
                debug_filename = f"{safe_filename}_{name}_processed.png"
                image_debug_files.append((debug_filename, img_io.getvalue()))

        return {'image_name': image_file.filename, 'data': image_result}, image_debug_files
    
    except Exception as e:
        logger.error(f"Error processing image {image_file.filename}: {str(e)}")
        return {'image_name': image_file.filename, 'error': f"Processing error: {str(e)}"}, []

# --- Merkle Tree and Blockchain Helper Functions ---

def generate_merkle_tree(data: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Generates a Merkle tree from a dictionary of data."""
    if not data:
        return "", {}

    sorted_items = sorted(data.items())
    leaves = [hashlib.sha256(f"{k}{v}".encode('utf-8')).digest() for k, v in sorted_items]
    
    leaf_hashes = {k: hashlib.sha256(f"{k}{v}".encode('utf-8')).hexdigest() for k, v in data.items()}

    if not leaves:
        return "", {}

    nodes = leaves
    while len(nodes) > 1:
        if len(nodes) % 2 != 0:
            nodes.append(nodes[-1])
        
        next_level = []
        for i in range(0, len(nodes), 2):
            combined_hash = nodes[i] + nodes[i+1]
            new_node = hashlib.sha256(combined_hash).digest()
            next_level.append(new_node)
        nodes = next_level

    merkle_root = nodes[0].hex()
    return merkle_root, leaf_hashes

def store_merkle_root_on_chain(root_hash: str) -> str:
    """Stores a given Merkle root hash on the blockchain."""
    if not merkle_contract or not w3:
        raise ConnectionError("Blockchain is not configured or connected.")

    try:
        root_hash_bytes = bytes.fromhex(root_hash)
        
        tx_data = {
            'from': signer_account.address,
            'nonce': w3.eth.get_transaction_count(signer_account.address),
            'gas': 200000,
            'gasPrice': w3.eth.gas_price,
        }
        
        transaction = merkle_contract.functions.storeRoot(root_hash_bytes).build_transaction(tx_data)
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=BlockchainConfig.SIGNER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hash_hex = tx_receipt.transactionHash.hex() if 'transactionHash' in tx_receipt else tx_hash.hex()
        logger.info(f"Merkle root stored on-chain. Tx hash: {tx_hash_hex}")
        return tx_hash_hex

    except Exception as e:
        logger.error(f"Failed to store Merkle root on-chain: {e}")
        raise

def verify_merkle_root_on_chain(root_hash: str) -> bool:
    """Checks if a Merkle root exists on the blockchain."""
    if not merkle_contract or not w3:
        raise ConnectionError("Blockchain is not configured or connected.")
    try:
        root_hash_bytes = bytes.fromhex(root_hash)
        exists = merkle_contract.functions.hasRoot(root_hash_bytes).call()
        return exists
    except Exception as e:
        logger.error(f"Failed to verify Merkle root on-chain: {e}")
        raise

# --- Flask Page Routes ---

@app.route('/')
def index():
    """Redirects to login page or role-specific dashboard."""
    if 'username' in session:
        if session.get('role') == 'university':
            return redirect(url_for('extract_data_page'))
        elif session.get('role') == 'verifier':
            return redirect(url_for('verify_page'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = USERS.get(username)

        if user and check_password_hash(user['pw_hash'], password):
            session['username'] = username
            session['role'] = user['role']
            logger.info(f"User '{username}' logged in with role '{user['role']}'.")
            if user['role'] == 'university':
                return redirect(url_for('build_template_page'))
            elif user['role'] == 'verifier':
                return redirect(url_for('verify_page'))
        else:
            logger.warning(f"Failed login attempt for username: '{username}'.")
            return render_template('login.html', error="Invalid username or password.")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the user out."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/build')
@login_required(role='university')
def build_template_page():
    """Renders the page for building a new template."""
    return render_template('build_template.html')

@app.route('/extract')
@login_required(role='university')
def extract_data_page():
    """Renders the page for extracting data using a template."""
    config_info = {
        'max_file_size_mb': app.config['MAX_FILE_SIZE'] // (1024 * 1024),
        'allowed_image_types': list(app.config['ALLOWED_IMAGE_EXTENSIONS']),
        'default_psm_mode': app.config['DEFAULT_PSM_MODE'],
        'default_scale_factor': app.config['DEFAULT_SCALE_FACTOR']
    }
    return render_template('extract_data.html', config=config_info)

@app.route('/verify_ocr', methods=['GET'])
@login_required(role='university')
def verify_ocr_page():
    return render_template('verify_ocr.html')

@app.route('/verify')
@login_required(role='verifier')
def verify_page():
    """Renders the data verification page for verifiers."""
    return render_template('verify.html')

# --- API Routes ---

@app.route('/extract_ocr', methods=['POST'])
@login_required(role='university')
def extract_ocr():
    """Batch OCR extraction, stores results in session for later verification."""
    try:
        if 'template' not in request.files:
            return jsonify({'error': 'Missing template file'}), 400
        template_file = request.files['template']
        valid_template, template_error, template_data = validate_template_file(template_file)
        if not valid_template:
            return jsonify({'error': f'Template validation failed: {template_error}'}), 400
        images = request.files.getlist('image')
        if not images or (len(images) == 1 and not images[0].filename):
            return jsonify({'error': 'No image files provided'}), 400
        valid_images = []
        validation_errors = []
        for img in images:
            if img and img.filename:
                valid_img, img_error = validate_image_file(img)
                if valid_img:
                    img.seek(0)
                    image_data = img.read()
                    valid_images.append({'filename': img.filename, 'data': image_data})
                else:
                    validation_errors.append(f"{img.filename}: {img_error}")
        if not valid_images:
            return jsonify({'error': f'No valid images found. Errors: {"; ".join(validation_errors)}'}), 400
        if validation_errors:
            logger.warning(f"Some images were invalid: {validation_errors}")
        debug = request.form.get('debug', 'false').lower() == 'true'
        ocr_config = {
            'psm_mode': int(request.form.get('psm_mode', app.config['DEFAULT_PSM_MODE'])),
            'scale_factor': int(request.form.get('scale_factor', app.config['DEFAULT_SCALE_FACTOR'])),
            'oem_mode': int(request.form.get('oem_mode', app.config['DEFAULT_OEM_MODE']))
        }
        logger.info(f"Processing {len(valid_images)} images with OCR config: {ocr_config}")
        results = [None] * len(valid_images)
        debug_files = []
        with ThreadPoolExecutor(max_workers=app.config['MAX_WORKERS']) as executor:
            future_to_idx = {}
            for i, img in enumerate(valid_images):
                img_file = io.BytesIO(img['data'])
                img_file.filename = img['filename']
                future = executor.submit(process_image, img_file, template_data, debug, ocr_config)
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    img_res, img_debug = future.result()
                    results[idx] = img_res
                    debug_files.extend(img_debug)
                    logger.info(f"Completed processing: {valid_images[idx]['filename']}")
                except Exception as e:
                    logger.error(f"Error processing {valid_images[idx]['filename']}: {str(e)}")
                    results[idx] = {'image_name': valid_images[idx]['filename'], 'error': str(e)}
        
        for i, res in enumerate(results):
            if 'error' not in res:
                image_bytes = valid_images[i]['data']
                res['image_base64'] = base64.b64encode(image_bytes).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(res['image_name'])
                res['image_mime'] = mime_type or 'image/jpeg'
        
        response = {
            'results': results,
            'processed_count': len([r for r in results if 'error' not in r]),
            'error_count': len([r for r in results if 'error' in r]),
            'validation_errors': validation_errors if validation_errors else None
        }
        if debug and debug_files:
            try:
                zip_io = io.BytesIO()
                with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename, content in debug_files:
                        zipf.writestr(filename, content)
                zip_io.seek(0)
                zip_base64 = base64.b64encode(zip_io.getvalue()).decode('utf-8')
                response['debug_zip'] = f'data:application/zip;base64,{zip_base64}'
                logger.info(f"Generated debug ZIP with {len(debug_files)} files")
            except Exception as e:
                logger.error(f"Error creating debug ZIP: {str(e)}")
                response['debug_error'] = str(e)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Unexpected error in extract_ocr: {str(e)}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/validate_template', methods=['POST'])
@login_required(role='university')
def validate_template():
    """API endpoint to validate template files."""
    if 'template' not in request.files:
        return jsonify({'valid': False, 'error': 'No template file provided'}), 400
    
    template_file = request.files['template']
    valid, error, template_data = validate_template_file(template_file)
    
    if valid:
        return jsonify({
            'valid': True,
            'field_count': len(template_data['fields']),
            'fields': [{'name': f['name'], 'type': f.get('type', 'multi')} for f in template_data['fields']]
        })
    else:
        return jsonify({'valid': False, 'error': error}), 400

@app.route('/generate_and_store_merkle', methods=['POST'])
@login_required(role='university')
def generate_and_store_merkle():
    """Receives verified data, generates Merkle tree(s), and stores root(s) on-chain."""
    try:
        req_json = request.get_json()
        if not req_json:
            return jsonify({'error': 'No data provided'}), 400

        if isinstance(req_json, dict) and 'bulk' in req_json and isinstance(req_json['bulk'], list):
            bulk_data = req_json['bulk']
            results, roots = [], []
            for idx, data in enumerate(bulk_data):
                if not isinstance(data, dict) or not data:
                    results.append({'error': f'Item {idx+1} is invalid or empty.'})
                    continue
                try:
                    logger.info(f"Generating Merkle tree for bulk item {idx+1}.")
                    merkle_root, leaf_hashes = generate_merkle_tree(data)
                    if not merkle_root:
                        results.append({'error': f'Item {idx+1} produced empty Merkle root.'})
                        continue
                    results.append({'merkle_root': merkle_root, 'leaf_hashes': leaf_hashes})
                    roots.append(merkle_root)
                except Exception as e:
                    logger.error(f"Error in Merkle generation (bulk item {idx+1}): {e}")
                    results.append({'error': f'Item {idx+1}: {str(e)}'})
            
            tx_hash_hex = None
            if roots:
                try:
                    root_bytes_list = [bytes.fromhex(r) for r in roots]
                    tx_data = {
                        'from': signer_account.address,
                        'nonce': w3.eth.get_transaction_count(signer_account.address),
                        'gas': 200000 + (len(roots) * 50000),
                        'gasPrice': w3.eth.gas_price,
                    }
                    transaction = merkle_contract.functions.storeRoots(root_bytes_list).build_transaction(tx_data)
                    signed_txn = w3.eth.account.sign_transaction(transaction, private_key=BlockchainConfig.SIGNER_PRIVATE_KEY)
                    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
                    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                    tx_hash_hex = tx_receipt.transactionHash.hex() if 'transactionHash' in tx_receipt else tx_hash.hex()
                    logger.info(f"Bulk Merkle roots stored on-chain. Tx hash: {tx_hash_hex}")
                except Exception as e:
                    logger.error(f"Failed to store bulk Merkle roots on-chain: {e}")
                    for res in results:
                        if 'error' not in res:
                            res['error'] = f'Blockchain storage failed: {str(e)}'
            
            for res in results:
                if 'error' not in res:
                    res['transaction_hash'] = tx_hash_hex
            return jsonify({'results': results})

        verified_data = req_json
        if not isinstance(verified_data, dict) or not verified_data:
            return jsonify({'error': 'Invalid or empty data format.'}), 400
        logger.info("Generating Merkle tree from verified data.")
        merkle_root, leaf_hashes = generate_merkle_tree(verified_data)
        if not merkle_root:
            return jsonify({'error': 'Empty data for Merkle tree.'}), 400
        tx_hash = store_merkle_root_on_chain(merkle_root)
        return jsonify({
            'success': True,
            'message': 'Merkle root successfully generated and stored on-chain.',
            'merkle_root': merkle_root,
            'leaf_hashes': leaf_hashes,
            'transaction_hash': tx_hash
        })

    except ConnectionError as e:
        logger.error(f"Blockchain connection error: {e}")
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(f"Error in Merkle generation/storage: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# --- Server-side Template Storage API ---

@app.route('/api/templates', methods=['GET'])
@login_required(role="ANY")
def list_templates():
    """Lists all saved templates."""
    try:
        path = app.config['TEMPLATE_STORAGE_PATH']
        templates = [f.replace('.json', '') for f in os.listdir(path) if f.endswith('.json')]
        return jsonify(sorted(templates))
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        return jsonify({"error": "Could not retrieve template list."}), 500

@app.route('/api/templates/<filename>', methods=['GET'])
@login_required(role="ANY")
def get_template(filename):
    """Retrieves a specific template file."""
    try:
        safe_filename = secure_filename(f"{filename}.json")
        return send_from_directory(app.config['TEMPLATE_STORAGE_PATH'], safe_filename)
    except FileNotFoundError:
        return jsonify({"error": "Template not found."}), 404
    except Exception as e:
        logger.error(f"Error getting template {filename}: {e}")
        return jsonify({"error": "Could not retrieve template."}), 500

@app.route('/api/templates', methods=['POST'])
@login_required(role="university")
def save_template():
    """Saves a new template to the server."""
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'template_data' not in data:
            return jsonify({'error': 'Request must contain name and template_data'}), 400
        
        template_name = data['name']
        template_data = data['template_data']

        safe_name = secure_filename(template_name)
        if not safe_name:
            return jsonify({'error': 'Invalid template name specified.'}), 400

        valid, error = validate_template_data(template_data)
        if not valid:
            return jsonify({'error': f'Template data is invalid: {error}'}), 400

        filepath = os.path.join(app.config['TEMPLATE_STORAGE_PATH'], f"{safe_name}.json")
        if os.path.exists(filepath):
            return jsonify({'error': f'Template "{safe_name}" already exists.'}), 409

        with open(filepath, 'w') as f:
            json.dump(template_data, f, indent=4)
        
        logger.info(f"Saved new template: {safe_name}.json")
        return jsonify({'success': True, 'message': f'Template "{safe_name}" saved successfully.'})

    except Exception as e:
        logger.error(f"Error saving template: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# --- New Verifier API Endpoints ---

@app.route('/api/ocr_for_verification', methods=['POST'])
@login_required(role='verifier')
def ocr_for_verification():
    """Runs OCR for the verifier, returning data for manual verification and editing."""
    try:
        if 'image' not in request.files or 'template' not in request.files:
            return jsonify({'error': 'Missing certificate image or template file'}), 400

        image_file = request.files['image']
        template_file = request.files['template']

        valid_template, template_error, template_data = validate_template_file(template_file)
        if not valid_template:
            return jsonify({'error': f'Template validation failed: {template_error}'}), 400

        ocr_config = {
            'psm_mode': app.config['DEFAULT_PSM_MODE'],
            'scale_factor': app.config['DEFAULT_SCALE_FACTOR'],
            'oem_mode': app.config['DEFAULT_OEM_MODE']
        }
        ocr_result, _ = process_image(image_file, template_data, debug=False, ocr_config=ocr_config)

        if 'error' in ocr_result:
            return jsonify(ocr_result), 400
        
        image_file.seek(0)
        image_bytes = image_file.read()
        ocr_result['image_base64'] = base64.b64encode(image_bytes).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(ocr_result['image_name'])
        ocr_result['image_mime'] = mime_type or 'image/jpeg'

        return jsonify(ocr_result)

    except Exception as e:
        logger.error(f"Unexpected error during verifier OCR: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/api/check_data_on_chain', methods=['POST'])
@login_required(role='verifier')
def check_data_on_chain():
    """Generates a Merkle root from verifier-provided data and checks if it exists on-chain."""
    try:
        data_to_verify = request.get_json()
        if not data_to_verify or not isinstance(data_to_verify, dict):
            return jsonify({'error': 'Data to verify must be a non-empty JSON object.'}), 400
        
        generated_merkle_root, _ = generate_merkle_tree(data_to_verify)
        if not generated_merkle_root:
            return jsonify({
                'is_on_chain': False,
                'message': "Could not generate a Merkle root from the provided data.",
                'generated_merkle_root': None
            })

        logger.info(f"Checking verifier-generated Merkle root '{generated_merkle_root}' on-chain.")
        is_on_chain = verify_merkle_root_on_chain(generated_merkle_root)

        if is_on_chain:
            message = "Success! The certificate data is authentic and its record was found on the blockchain."
        else:
            message = "Verification Failed. The data from this certificate has not been recorded on the blockchain."

        return jsonify({
            'is_on_chain': is_on_chain,
            'message': message,
            'generated_merkle_root': generated_merkle_root
        })

    except ConnectionError as e:
        logger.error(f"Blockchain connection error during final verification: {e}")
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(f"Error during on-chain check: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# --- Error Handlers ---

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size is {app.config["MAX_FILE_SIZE"] // (1024*1024)}MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)