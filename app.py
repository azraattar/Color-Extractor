from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import os
from werkzeug.utils import secure_filename
import colorsys
import pandas as pd
import math
import random

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load XKCD color dataset from CSV
color_data = pd.read_csv('xkcd_colors_cleaned.csv')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def closest_color(rgb_tuple):
    """Find the closest color name from the XKCD dataset using Euclidean distance."""
    min_distance = float('inf')
    closest_name = "Unknown"
    
    for index, row in color_data.iterrows():
        hex_color = row['hex']
        name = row['name']
        
        # Convert hex to RGB
        hex_r = int(hex_color[1:3], 16)
        hex_g = int(hex_color[3:5], 16)
        hex_b = int(hex_color[5:7], 16)
        
        # Calculate Euclidean distance
        distance = math.sqrt(
            (rgb_tuple[0] - hex_r) ** 2 +
            (rgb_tuple[1] - hex_g) ** 2 +
            (rgb_tuple[2] - hex_b) ** 2
        )
        
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    
    return closest_name

def get_dominant_color(image_path):
    """Get the single most dominant color in an image."""
    img = Image.open(image_path).convert('RGB')
    
    # Reduce colors to improve performance and reduce noise
    img = img.quantize(colors=256).convert('RGB')
    
    # Get all pixels
    pixels = img.getdata()
    
    # Count occurrences of each color
    color_count = Counter(pixels)
    
    # Get the most common color
    dominant_color = color_count.most_common(1)[0][0]
    
    # Convert to hex
    hex_color = '#%02x%02x%02x' % dominant_color
    
    # Get the color name
    color_name = closest_color(dominant_color)
    
    return {"hex": hex_color, "name": color_name}

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color code."""
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hex_to_rgb(hex_color):
    """Convert hex color code to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_complementary_color(rgb):
    """Generate the complementary color of a given RGB color."""
    comp_rgb = (255 - rgb[0], 255 - rgb[1], 255 - rgb[2])
    return rgb_to_hex(comp_rgb)

def extract_colors(image_path, num_colors=5):
    """Extract the most representative colors from an image using improved method."""
    img = Image.open(image_path).convert('RGB')
    
    # Resize image to speed up processing
    img.thumbnail((200, 200))
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Reshape the array to a list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # Check if this is a solid or nearly solid color image
    unique_colors = np.unique(pixels, axis=0)
    if len(unique_colors) <= 5:
        result = []
        for color in unique_colors:
            hex_color = rgb_to_hex(color)
            color_name = closest_color(color)
            complementary = get_complementary_color(color)
            result.append({"hex": hex_color, "name": color_name, "complementary": complementary})
        return result
    
    # For complex images, use K-means with better parameters
    kmeans = KMeans(
        n_clusters=num_colors,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    # Fit K-means to the pixel data
    kmeans.fit(pixels)
    
    # Get the colors
    colors = kmeans.cluster_centers_
    
    # Count pixels in each cluster to determine importance
    labels = kmeans.labels_
    color_counts = np.bincount(labels)
    
    # Sort colors by count (most common first)
    sorted_indices = np.argsort(color_counts)[::-1]
    sorted_colors = colors[sorted_indices]
    
    result = []
    for color in sorted_colors:
        hex_color = rgb_to_hex(color)
        color_name = closest_color(color)
        complementary = get_complementary_color(color)
        result.append({"hex": hex_color, "name": color_name, "complementary": complementary})
    
    return result

# New functions for palette generation
def rgb_to_hsv(rgb):
    """Convert RGB to HSV."""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h, s, v)

def hsv_to_rgb(hsv):
    """Convert HSV to RGB."""
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

def generate_analogous_palette(hex_color, count=5):
    """Generate analogous color palette from a base color."""
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb)
    
    palette = []
    # Get colors that are adjacent on the color wheel
    for i in range(count):
        # Calculate angle on the color wheel
        # Analogous colors are within about 30° of the base color
        new_h = (h + (i - count//2) * 0.05) % 1.0
        new_rgb = hsv_to_rgb((new_h, s, v))
        
        new_hex = rgb_to_hex(new_rgb)
        name = closest_color(new_rgb)
        palette.append({"hex": new_hex, "name": name})
    
    return palette

def generate_monochromatic_palette(hex_color, count=5):
    """Generate monochromatic color palette from a base color."""
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb)
    
    palette = []
    for i in range(count):
        # Keep the same hue, vary saturation and value
        new_s = 0.3 + (0.7 * i / (count - 1)) if count > 1 else s
        new_v = 0.7 + (0.3 * i / (count - 1)) if count > 1 else v
        
        new_rgb = hsv_to_rgb((h, new_s, new_v))
        new_hex = rgb_to_hex(new_rgb)
        name = closest_color(new_rgb)
        palette.append({"hex": new_hex, "name": name})
    
    return palette

def generate_triadic_palette(hex_color, count=5):
    """Generate triadic color palette from a base color."""
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb)
    
    palette = []
    # Base triadic colors (120° apart on the color wheel)
    base_triadic = [h, (h + 1/3) % 1.0, (h + 2/3) % 1.0]
    
    # Fill in the first 3 slots with our triadic colors
    for i in range(min(3, count)):
        triadic_rgb = hsv_to_rgb((base_triadic[i], s, v))
        triadic_hex = rgb_to_hex(triadic_rgb)
        name = closest_color(triadic_rgb)
        palette.append({"hex": triadic_hex, "name": name})
    
    # If we need more colors, fill in with variations
    for i in range(3, count):
        # Choose a base triadic color to vary
        base_h = base_triadic[i % 3]
        # Vary the saturation and value slightly
        new_s = max(0.2, min(s * (0.8 + (i/count) * 0.4), 1.0))
        new_v = max(0.2, min(v * (0.8 + (i/count) * 0.4), 1.0))
        
        new_rgb = hsv_to_rgb((base_h, new_s, new_v))
        new_hex = rgb_to_hex(new_rgb)
        name = closest_color(new_rgb)
        palette.append({"hex": new_hex, "name": name})
    
    return palette

def generate_complementary_palette(hex_color, count=5):
    """Generate complementary color palette from a base color."""
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb)
    
    # Complementary color (opposite on the color wheel)
    comp_h = (h + 0.5) % 1.0
    
    palette = []
    # Add the base color
    palette.append({"hex": hex_color, "name": closest_color(rgb)})
    
    # Add the complementary color
    comp_rgb = hsv_to_rgb((comp_h, s, v))
    comp_hex = rgb_to_hex(comp_rgb)
    comp_name = closest_color(comp_rgb)
    palette.append({"hex": comp_hex, "name": comp_name})
    
    # Fill in remaining slots with variations of both colors
    for i in range(2, count):
        # Alternate between the base and complementary color
        base_h = h if i % 2 == 0 else comp_h
        
        # Create variations by adjusting saturation and value
        new_s = max(0.3, min(s * (0.7 + (i/count) * 0.5), 1.0))
        new_v = max(0.3, min(v * (0.7 + (i/count) * 0.5), 1.0))
        
        new_rgb = hsv_to_rgb((base_h, new_s, new_v))
        new_hex = rgb_to_hex(new_rgb)
        name = closest_color(new_rgb)
        palette.append({"hex": new_hex, "name": name})
    
    return palette

def generate_split_complementary_palette(hex_color, count=5):
    """Generate split-complementary color palette from a base color."""
    rgb = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(rgb)
    
    # Split-complementary colors (the color + two colors adjacent to its complement)
    comp_h = (h + 0.5) % 1.0
    split1_h = (comp_h - 0.05) % 1.0
    split2_h = (comp_h + 0.05) % 1.0
    
    palette = []
    
    # Add the original color
    palette.append({"hex": hex_color, "name": closest_color(rgb)})
    
    # Add the two split-complementary colors
    split1_rgb = hsv_to_rgb((split1_h, s, v))
    split1_hex = rgb_to_hex(split1_rgb)
    split1_name = closest_color(split1_rgb)
    palette.append({"hex": split1_hex, "name": split1_name})
    
    split2_rgb = hsv_to_rgb((split2_h, s, v))
    split2_hex = rgb_to_hex(split2_rgb)
    split2_name = closest_color(split2_rgb)
    palette.append({"hex": split2_hex, "name": split2_name})
    
    # Fill in remaining slots with variations
    for i in range(3, count):
        # Cycle through our three base colors
        if i % 3 == 0:
            base_h = h
        elif i % 3 == 1:
            base_h = split1_h
        else:
            base_h = split2_h
        
        # Create slight variations
        new_s = max(0.2, min(s * (0.8 + (i/count) * 0.4), 1.0))
        new_v = max(0.2, min(v * (0.8 + (i/count) * 0.4), 1.0))
        
        new_rgb = hsv_to_rgb((base_h, new_s, new_v))
        new_hex = rgb_to_hex(new_rgb)
        name = closest_color(new_rgb)
        palette.append({"hex": new_hex, "name": name})
    
    return palette

def generate_random_palette(base_colors, count=5):
    """Generate random palette based on extracted colors with some level of harmony."""
    palette = []
    
    # If we have base colors, use them as inspiration
    if base_colors:
        # First, add a couple of the base colors directly
        num_base_to_use = min(2, len(base_colors))
        for i in range(num_base_to_use):
            palette.append({"hex": base_colors[i]["hex"], "name": base_colors[i]["name"]})
        
        # Then create some variations and random colors that still have harmony
        for i in range(num_base_to_use, count):
            # Randomly select a base color as reference
            ref_color = random.choice(base_colors)
            rgb = hex_to_rgb(ref_color["hex"])
            h, s, v = rgb_to_hsv(rgb)
            
            # Create a somewhat random variation
            new_h = (h + random.uniform(-0.2, 0.2)) % 1.0
            new_s = max(0.1, min(random.uniform(0.6, 1.0), 1.0))
            new_v = max(0.1, min(random.uniform(0.6, 1.0), 1.0))
            
            new_rgb = hsv_to_rgb((new_h, new_s, new_v))
            new_hex = rgb_to_hex(new_rgb)
            name = closest_color(new_rgb)
            palette.append({"hex": new_hex, "name": name})
    else:
        # If no base colors, create truly random colors
        for i in range(count):
            h = random.random()
            s = random.uniform(0.6, 1.0)
            v = random.uniform(0.6, 1.0)
            
            rgb = hsv_to_rgb((h, s, v))
            hex_color = rgb_to_hex(rgb)
            name = closest_color(rgb)
            palette.append({"hex": hex_color, "name": name})
    
    return palette

def generate_multi_color_complementary(base_colors, count=5):
    """Generate complementary palette using multiple source colors."""
    palette = []
    
    # Add original colors
    for color in base_colors:
        palette.append({"hex": color["hex"], "name": color["name"]})
    
    # Add complementary colors for each base color
    for color in base_colors:
        rgb = hex_to_rgb(color["hex"])
        h, s, v = rgb_to_hsv(rgb)
        
        # Get complementary color
        comp_h = (h + 0.5) % 1.0
        comp_rgb = hsv_to_rgb((comp_h, s, v))
        comp_hex = rgb_to_hex(comp_rgb)
        comp_name = closest_color(comp_rgb)
        
        palette.append({"hex": comp_hex, "name": comp_name})
    
    # Trim or extend to the requested count
    if len(palette) > count:
        palette = palette[:count]
    elif len(palette) < count:
        # Fill remaining slots with variations
        for i in range(count - len(palette)):
            base_idx = i % len(base_colors)
            rgb = hex_to_rgb(base_colors[base_idx]["hex"])
            h, s, v = rgb_to_hsv(rgb)
            
            # Create a variation
            new_s = max(0.2, min(s * (0.8 + (i/count) * 0.4), 1.0))
            new_v = max(0.2, min(v * (0.8 + (i/count) * 0.4), 1.0))
            new_rgb = hsv_to_rgb((h, new_s, new_v))
            
            hex_color = rgb_to_hex(new_rgb)
            name = closest_color(new_rgb)
            palette.append({"hex": hex_color, "name": name})
    
    return palette

def generate_multi_color_split(base_colors, count=5):
    """Generate split-complementary palette using multiple source colors."""
    palette = []
    
    # Add the original colors
    for color in base_colors:
        palette.append({"hex": color["hex"], "name": color["name"]})
    
    # For each base color, add split-complementary colors
    for color in base_colors:
        rgb = hex_to_rgb(color["hex"])
        h, s, v = rgb_to_hsv(rgb)
        
        # Get complementary base
        comp_h = (h + 0.5) % 1.0
        
        # Get two colors adjacent to the complementary color (split-complementary)
        split_h1 = (comp_h - 0.05) % 1.0
        split_h2 = (comp_h + 0.05) % 1.0
        
        # Only add one split color per base to avoid too many colors
        split_rgb = hsv_to_rgb((split_h1 if len(palette) % 2 == 0 else split_h2, s, v))
        split_hex = rgb_to_hex(split_rgb)
        split_name = closest_color(split_rgb)
        
        palette.append({"hex": split_hex, "name": split_name})
    
    # Trim or extend to the requested count
    if len(palette) > count:
        palette = palette[:count]
    elif len(palette) < count:
        # Fill remaining slots with variations
        for i in range(count - len(palette)):
            base_idx = i % len(base_colors)
            rgb = hex_to_rgb(base_colors[base_idx]["hex"])
            h, s, v = rgb_to_hsv(rgb)
            
            # Create a variation
            new_h = (h + (i * 0.1)) % 1.0
            new_rgb = hsv_to_rgb((new_h, s, v))
            
            hex_color = rgb_to_hex(new_rgb)
            name = closest_color(new_rgb)
            palette.append({"hex": hex_color, "name": name})
    
    return palette

def generate_extracted_palette(base_colors, count=5):
    """Generate a palette that directly uses the extracted colors with minimal modification."""
    palette = []
    
    # Add all extracted colors first
    for color in base_colors:
        palette.append({"hex": color["hex"], "name": color["name"]})
    
    # If we need more colors, create slight variations
    if len(palette) < count:
        for i in range(count - len(palette)):
            # Choose a base color to modify
            base_idx = i % len(base_colors)
            rgb = hex_to_rgb(base_colors[base_idx]["hex"])
            h, s, v = rgb_to_hsv(rgb)
            
            # Subtle variation to get a coherent palette
            new_s = max(0.1, min(s * (0.9 + (i/10)), 1.0))
            new_v = max(0.1, min(v * (0.9 + (i/10)), 1.0))
            new_rgb = hsv_to_rgb((h, new_s, new_v))
            
            hex_color = rgb_to_hex(new_rgb)
            name = closest_color(new_rgb)
            palette.append({"hex": hex_color, "name": name})
    else:
        # If we have too many colors, just use the first 'count' colors
        palette = palette[:count]
    
    return palette

def generate_palettes(base_colors):
    """Generate multiple palettes based on extracted colors, 
    using more of the extracted colors rather than just the dominant one."""
    if not base_colors:
        return {}
    
    # Use multiple colors instead of just the dominant one
    palettes = {}
    
    # For palettes that traditionally use one color as base, rotate through our extracted colors
    for i, color in enumerate(base_colors[:min(3, len(base_colors))]):
        primary_color = color["hex"]
        palette_type = ""
        
        # Assign different palette types to different extracted colors
        if i == 0:
            palette_type = "analogous"
            palettes[palette_type] = generate_analogous_palette(primary_color)
        elif i == 1:
            palette_type = "monochromatic"
            palettes[palette_type] = generate_monochromatic_palette(primary_color)
        elif i == 2:
            palette_type = "triadic"
            palettes[palette_type] = generate_triadic_palette(primary_color)
    
    # For palettes that can incorporate multiple colors, use combinations of extracted colors
    if len(base_colors) >= 2:
        # Use the two most prominent colors for complementary palette
        palettes["complementary"] = generate_multi_color_complementary(base_colors[:2])
        
        # Use three most prominent colors for split complementary
        colors_for_split = base_colors[:min(3, len(base_colors))]
        palettes["split_complementary"] = generate_multi_color_split(colors_for_split)
    else:
        # Fallback to original methods if we don't have enough colors
        primary_color = base_colors[0]["hex"]
        palettes["complementary"] = generate_complementary_palette(primary_color)
        palettes["split_complementary"] = generate_split_complementary_palette(primary_color)
    
    # Use all extracted colors for random palette generation
    palettes["random"] = generate_random_palette(base_colors, count=5)
    
    # Add a new palette type that explicitly uses all extracted colors
    palettes["extracted"] = generate_extracted_palette(base_colors)
    
    return palettes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract-colors', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # For solid colors, get the dominant color
            dominant_color = get_dominant_color(filepath)
            
            # Get the clustered colors
            clustered_colors = extract_colors(filepath)
            
            return jsonify({
                'dominant_color': dominant_color,
                'clustered_colors': clustered_colors,
                'image_url': f"/{filepath}"
            })
        except Exception as e:
            return jsonify({'error': f"Processing error: {str(e)}"}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/generate-palettes', methods=['POST'])
def generate_color_palettes():
    data = request.json
    if not data or 'colors' not in data:
        return jsonify({'error': 'No colors provided'}), 400
    
    try:
        # Generate various palettes based on the provided colors
        palettes = generate_palettes(data['colors'])
        
        return jsonify({
            'palettes': palettes
        })
    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)