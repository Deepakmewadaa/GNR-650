import os
import csv
import argparse
from PIL import Image, ImageDraw, ImageFont

def build_visual_grid(images_dir: str, prompts_csv: str, output_path: str, tile_size: int = 256):
    """
    Stitches generated images into a high-resolution grid and creates an HTML gallery.
    """
    if not os.path.exists(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return

    # Load prompts
    prompts = {}
    if os.path.exists(prompts_csv):
        with open(prompts_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    prompts[str(row[0]).strip()] = str(row[1]).strip()

    # Collect images
    img_files = []
    for f in sorted(os.listdir(images_dir)):
        if f.startswith("image_") and f.endswith(".png"):
            item_id = f.replace("image_", "").replace(".png", "")
            img_files.append((item_id, os.path.join(images_dir, f)))

    if not img_files:
        print(f"No image_*.png found in {images_dir}")
        return

    # Sort numerically if possible
    try:
        img_files.sort(key=lambda x: int(x[0]))
    except ValueError:
        pass

    cols = 4
    rows = (len(img_files) + cols - 1) // cols
    margin = 12
    caption_h = 36
    cell_w = tile_size + margin
    cell_h = tile_size + caption_h + margin

    grid_w = cols * cell_w + margin
    grid_h = rows * cell_h + margin
    grid_img = Image.new("RGB", (grid_w, grid_h), color=(245, 245, 247))
    draw = ImageDraw.Draw(grid_img)

    for idx, (item_id, path) in enumerate(img_files):
        c = idx % cols
        r = idx // cols
        x = margin + c * cell_w
        y = margin + r * cell_h

        try:
            with Image.open(path) as im:
                im_resized = im.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                grid_img.paste(im_resized, (x, y))
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue

        label = f"ID {item_id}: {prompts.get(item_id, '')[:35]}..."
        draw.text((x + 2, y + tile_size + 6), label, fill=(30, 30, 30))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    grid_img.save(output_path, quality=95)
    print(f"Successfully saved high-resolution grid ({grid_w}x{grid_h}) to: {output_path}")

    # Also build an interactive HTML gallery for easy viewing in browser
    html_path = os.path.join(images_dir, "viewer.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Generated Face Evaluation Gallery</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
        h1 { text-align: center; font-size: 24px; font-weight: 600; margin-bottom: 24px; color: #38bdf8; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; max-width: 1400px; margin: 0 auto; }
        .card { background: #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.3); border: 1px solid #334155; }
        .card img { width: 100%; height: 260px; object-fit: cover; image-rendering: auto; }
        .card-body { padding: 12px 16px; }
        .badge { display: inline-block; background: #0284c7; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; margin-bottom: 6px; }
        .prompt { font-size: 13px; line-height: 1.4; color: #cbd5e1; }
    </style>
</head>
<body>
    <h1>Generated Face Gallery (Optimal Transport Flow Matching)</h1>
    <div class="gallery">
""")
        for item_id, path in img_files:
            fname = os.path.basename(path)
            prompt_str = prompts.get(item_id, "")
            f.write(f"""        <div class="card">
            <img src="{fname}" alt="Generated image {item_id}">
            <div class="card-body">
                <span class="badge">ID {item_id}</span>
                <div class="prompt">{prompt_str}</div>
            </div>
        </div>\n""")
        f.write("""    </div>
</body>
</html>\n""")
    print(f"Generated interactive browser gallery at: {html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Visual Grid and HTML viewer for generated images")
    parser.add_argument("--images-dir", type=str, default="outputs/grader_results", help="Directory containing image_*.png")
    parser.add_argument("--prompts-csv", type=str, default="data/test_sample_prompts.csv", help="CSV with id and prompt")
    parser.add_argument("--output", type=str, default="outputs/evaluation_grid.png", help="Path to save output grid image")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels")
    args = parser.parse_args()
    build_visual_grid(args.images_dir, args.prompts_csv, args.output, args.tile_size)
