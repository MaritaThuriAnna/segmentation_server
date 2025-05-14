import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import io
import base64
from ultralytics import YOLO

# Mapping of YOLOv8 class IDs (0 to 31, i.e., tooth_1 to tooth_32) to human-readable names
tooth_name_mapping = {
    0: "Upper Right Third Molar",
    1: "Upper Right Second Molar",
    2: "Upper Right First Molar",
    3: "Upper Right Second Premolar",
    4: "Upper Right First Premolar",
    5: "Upper Right Canine",
    6: "Upper Right Lateral Incisor",
    7: "Upper Right Central Incisor",
    8: "Upper Left Central Incisor",
    9: "Upper Left Lateral Incisor",
    10: "Upper Left Canine",
    11: "Upper Left First Premolar",
    12: "Upper Left Second Premolar",
    13: "Upper Left First Molar",
    14: "Upper Left Second Molar",
    15: "Upper Left Third Molar",
    16: "Lower Left Third Molar",
    17: "Lower Left Second Molar",
    18: "Lower Left First Molar",
    19: "Lower Left Second Premolar",
    20: "Lower Left First Premolar",
    21: "Lower Left Canine",
    22: "Lower Left Lateral Incisor",
    23: "Lower Left Central Incisor",
    24: "Lower Right Central Incisor",
    25: "Lower Right Lateral Incisor",
    26: "Lower Right Canine",
    27: "Lower Right First Premolar",
    28: "Lower Right Second Premolar",
    29: "Lower Right First Molar",
    30: "Lower Right Second Molar",
    31: "Lower Right Third Molar"
}


# Caries detection function (U-Net++)
def test_caries_detection_highlight_only(image, model_path='/app/models/best_model4.pth',
                                         threshold=0.05, crop=True):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mean, std = 0.5, 0.5
    transform = transforms.Compose([
        transforms.Resize((384, 768)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    try:
        # Store the original dimensions for mapping back
        original_size = image.size
        if crop:
            W, H = image.size
            left = int(W * 0.239)
            right = int(W * 0.761)
            top = int(H * 0.2325)
            bottom = int(H * 0.7675)
            image = image.crop((left, top, right, bottom))

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            preds = (probs > threshold).float()

        preds_np = preds.squeeze().cpu().numpy()
        mask_pil = Image.fromarray((preds_np * 255).astype(np.uint8))

        original = image.resize((768, 384)).convert('RGB')
        overlay = Image.new('RGBA', original.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for x in range(mask_pil.size[0]):
            for y in range(mask_pil.size[1]):
                if mask_pil.getpixel((x, y)) > 0:
                    draw.point((x, y), (255, 0, 0, 128))  # Red highlight
        overlaid = Image.alpha_composite(original.convert('RGBA'), overlay)

        mask_buffered = io.BytesIO()
        mask_pil.save(mask_buffered, format="PNG")
        mask_base64 = base64.b64encode(mask_buffered.getvalue()).decode()

        overlay_buffered = io.BytesIO()
        overlaid.save(overlay_buffered, format="PNG")
        overlay_base64 = base64.b64encode(overlay_buffered.getvalue()).decode()

        return {
            "mask": mask_base64,
            "overlay": overlay_base64,
            "caries_detected": bool(np.any(preds_np > 0)),
            "caries_area_ratio": float(np.mean(preds_np)),
            "mask_np": preds_np,
            "original_size": original_size,
            "cropped_image": image,
            "crop_coords": (left, top, right, bottom)
        }

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {"error": str(e)}


# Main function to detect teeth and caries
def detect_caries_and_teeth(image_input, yolo_model_path='/app/models/best.pt',
                            caries_model_path='/app/models/best_model4.pth', threshold=0.01):
    """
    Detect teeth and caries in an X-ray image.

    Args:
        image_input: Path to the image or numpy array of the image.
        yolo_model_path: Path to the YOLOv8 model weights.
        caries_model_path: Path to the U-Net++ model weights.
        threshold: Caries detection threshold.

    Returns:
        dict: {
            "output_image": Base64 string of the final image with bounding boxes and caries highlights,
            "caries_teeth": List of human-readable names of teeth with caries,
            "caries_area_ratio": Area ratio of detected caries
        }
    """
    # Load the YOLOv8 model
    model = YOLO(yolo_model_path)

    # Load the image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input  # Assume it's a numpy array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference with YOLOv8 to detect teeth
    results = model.predict(source=image_rgb, conf=0.5)

    # Store bounding boxes and class IDs
    boxes = []
    classes = []
    confidences = []
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf

    # Convert the original image to PIL for caries detection
    original_pil = Image.fromarray(image_rgb)

    # Run caries detection on the whole image
    caries_result = test_caries_detection_highlight_only(
        image=original_pil,
        model_path=caries_model_path,
        threshold=threshold,
        crop=True
    )

    # Process caries detection results and map to teeth
    caries_teeth = []
    caries_boxes = []
    caries_confidences = []
    caries_area_ratio = 0.0
    if "caries_detected" in caries_result and caries_result['caries_detected']:
        caries_area_ratio = caries_result['caries_area_ratio']
        print(f"Caries detected in the image with area ratio {caries_area_ratio:.4f}")

        # Extract crop coordinates
        left, top, right, bottom = caries_result['crop_coords']
        cropped_W = right - left
        cropped_H = bottom - top
        original_size = caries_result['original_size']
        W, H = original_size

        # Resize mask to cropped region size (768x384 after resize in caries detection)
        mask_np = caries_result['mask_np']
        mask_resized = cv2.resize(mask_np, (768, 384), interpolation=cv2.INTER_NEAREST)

        # Map the mask back to the original image resolution
        mask_original = np.zeros((H, W), dtype=np.uint8)
        mask_resized_cropped = cv2.resize(mask_resized, (cropped_W, cropped_H), interpolation=cv2.INTER_NEAREST)
        # Ensure dimensions match by clipping if necessary
        h, w = mask_resized_cropped.shape
        if h == cropped_H and w == cropped_W:
            mask_original[top:bottom, left:right] = mask_resized_cropped
        else:
            h_clip = min(cropped_H, h)
            w_clip = min(cropped_W, w)
            mask_original[top:top + h_clip, left:left + w_clip] = mask_resized_cropped[:h_clip, :w_clip]

        # Check overlap between caries mask and bounding boxes to identify affected teeth
        for box, cls, conf in zip(boxes, classes, confidences):
            x_min, y_min, x_max, y_max = map(int, box)
            class_id = int(cls)
            human_readable_name = tooth_name_mapping[class_id]

            # Create a mask for the bounding box region
            tooth_mask = np.zeros((H, W), dtype=np.uint8)
            tooth_mask[y_min:y_max, x_min:x_max] = 1

            # Compute overlap
            overlap = np.logical_and(mask_original > 0, tooth_mask).sum()
            tooth_area = (x_max - x_min) * (y_max - y_min)
            overlap_ratio = overlap / tooth_area if tooth_area > 0 else 0

            if overlap_ratio > 0.01:  # Threshold for significant overlap
                caries_teeth.append(human_readable_name)
                caries_boxes.append((x_min, y_min, x_max, y_max))
                caries_confidences.append(conf)
                print(f"Caries detected on {human_readable_name} with overlap ratio {overlap_ratio:.4f}")

    # Crop the original image to match the caries detection crop
    vis_pil = Image.fromarray(image_rgb)
    vis_cropped = vis_pil.crop((left, top, right, bottom)).resize((768, 384))
    vis_cropped_rgb = np.array(vis_cropped)

    # Draw bounding boxes only on teeth with caries
    for (x_min, y_min, x_max, y_max), human_readable_name, conf in zip(caries_boxes, caries_teeth, caries_confidences):
        # Map the bounding box coordinates to the cropped image
        x_min_cropped = int((x_min - left) * 768 / cropped_W)
        x_max_cropped = int((x_max - left) * 768 / cropped_W)
        y_min_cropped = int((y_min - top) * 384 / cropped_H)
        y_max_cropped = int((y_max - top) * 384 / cropped_H)

        # Draw the bounding box
        cv2.rectangle(vis_cropped_rgb, (x_min_cropped, y_min_cropped), (x_max_cropped, y_max_cropped), (0, 255, 0), 2)
        # Uncomment if you want to show labels
        # cv2.putText(vis_cropped_rgb, f"{human_readable_name} ({conf:.2f})", (x_min_cropped, y_min_cropped - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Convert the caries overlay to numpy for combining
    overlaid = Image.open(io.BytesIO(base64.b64decode(caries_result['overlay']))).convert('RGBA')
    overlaid_np = np.array(overlaid)

    # Convert the cropped image with bounding boxes to RGBA
    vis_cropped_rgba = Image.fromarray(vis_cropped_rgb).convert('RGBA')
    vis_cropped_np = np.array(vis_cropped_rgba)

    # Combine the images (blend bounding boxes with caries highlights)
    alpha = 0.5  # Transparency for bounding box overlay
    combined = (vis_cropped_np.astype(float) * alpha + overlaid_np.astype(float) * (1 - alpha)).astype(np.uint8)

    # Convert the final image to base64 for app usage
    combined_pil = Image.fromarray(combined)
    output_buffer = io.BytesIO()
    combined_pil.save(output_buffer, format="PNG")
    output_image_base64 = base64.b64encode(output_buffer.getvalue()).decode()

    return {
        "output_image": output_image_base64,
        "caries_teeth": caries_teeth,
        "caries_area_ratio": caries_area_ratio
    }