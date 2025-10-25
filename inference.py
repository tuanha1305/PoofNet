import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
import os
import cv2
import matplotlib.pyplot as plt

from model import PoofNet


class ObjectRemover:
    """
    Class để inference - remove objects từ ảnh
    """

    def __init__(
            self,
            checkpoint_path: str,
            num_classes: int = 80,
            num_styles: int = 3,
            device: str = 'cuda',
            img_size: int = 512
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            num_classes: Number of object classes
            num_styles: Number of image styles
            device: Device to run inference
            img_size: Image size for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        # Style mapping
        self.style_to_idx = {
            'human': 0,
            'art': 1,
            'graphic_design': 2
        }

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = PoofNet(
            in_ch=3,
            out_ch=1,
            num_classes=num_classes,
            num_styles=num_styles
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✅ Model loaded successfully on {self.device}")

        # Transform
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict_mask(
            self,
            image: np.ndarray,
            bboxes: list,
            classes: list,
            style: str = 'human'
    ) -> np.ndarray:
        """
        Predict segmentation mask

        Args:
            image: Input image (H, W, 3) numpy array
            bboxes: List of bounding boxes [[x1,y1,x2,y2], ...]
            classes: List of class indices [0, 2, ...]
            style: Image style ('human', 'art', 'graphic_design')

        Returns:
            mask: Binary mask (H, W) numpy array
        """
        orig_h, orig_w = image.shape[:2]

        # Convert to PIL Image
        pil_img = Image.fromarray(image)

        # Transform
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Normalize bboxes
        bboxes_norm = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1_norm = x1 / orig_w
            y1_norm = y1 / orig_h
            x2_norm = x2 / orig_w
            y2_norm = y2 / orig_h
            bboxes_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])

        # Padding to max 20 objects
        max_objects = 20
        while len(bboxes_norm) < max_objects:
            bboxes_norm.append([0.0, 0.0, 0.0, 0.0])
            classes.append(0)

        # Convert to tensors
        bboxes_tensor = torch.tensor([bboxes_norm], dtype=torch.float32).to(self.device)
        classes_tensor = torch.tensor([classes], dtype=torch.long).to(self.device)

        # Style index
        style_idx = self.style_to_idx.get(style, 0)
        style_tensor = torch.tensor([style_idx], dtype=torch.long).to(self.device)

        # Inference
        outputs = self.model(img_tensor, bboxes_tensor, classes_tensor, style_tensor)

        # Get final prediction (first output is the fused result)
        pred_mask = outputs[0][0, 0].cpu().numpy()

        # Resize to original size
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Threshold
        pred_mask = (pred_mask > 0.5).astype(np.uint8)

        return pred_mask

    def remove_objects(
            self,
            image: np.ndarray,
            bboxes: list,
            classes: list,
            style: str = 'human',
            inpaint_method: str = 'telea'
    ) -> np.ndarray:
        """
        Remove objects and inpaint

        Args:
            image: Input image (H, W, 3)
            bboxes: List of bounding boxes
            classes: List of class indices
            style: Image style ('human', 'art', 'graphic_design')
            inpaint_method: 'telea' or 'ns' (Navier-Stokes)

        Returns:
            result: Inpainted image (H, W, 3)
        """
        # Get segmentation mask
        mask = self.predict_mask(image, bboxes, classes, style)

        # Dilate mask slightly for better inpainting
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask * 255, kernel, iterations=1)

        # Inpaint
        if inpaint_method == 'telea':
            result = cv2.inpaint(image, mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        else:
            result = cv2.inpaint(image, mask_dilated, inpaintRadius=3, flags=cv2.INPAINT_NS)

        return result, mask

    def visualize(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            result: np.ndarray,
            bboxes: list = None,
            save_path: str = None
    ):
        """
        Visualize results
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image with bboxes
        ax = axes[0]
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if bboxes:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
        ax.set_title('Original Image with BBoxes')
        ax.axis('off')

        # Predicted mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')

        # Result
        axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Result (Objects Removed)')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()


def interactive_bbox_selector(image_path: str):
    """
    Interactive tool để user select bounding boxes

    Usage:
        - Click and drag to draw bounding box
        - Press 's' to save and continue
        - Press 'q' to quit
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image from {image_path}")

    clone = img.copy()
    bboxes = []
    current_bbox = []
    drawing = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, current_bbox, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_bbox = [x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_img = clone.copy()
                cv2.rectangle(temp_img, (current_bbox[0], current_bbox[1]), (x, y), (0, 255, 0), 2)
                cv2.imshow('Select Objects', temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_bbox.extend([x, y])
            bboxes.append(current_bbox.copy())
            cv2.rectangle(clone, (current_bbox[0], current_bbox[1]),
                          (current_bbox[2], current_bbox[3]), (0, 255, 0), 2)
            cv2.imshow('Select Objects', clone)

    cv2.namedWindow('Select Objects')
    cv2.setMouseCallback('Select Objects', mouse_callback)
    cv2.imshow('Select Objects', img)

    print("\n📝 Instructions:")
    print("  - Click and drag to draw bounding box")
    print("  - Press 's' to save and continue")
    print("  - Press 'q' to quit without saving")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            break
        elif key == ord('q'):
            bboxes = []
            break

    cv2.destroyAllWindows()

    return bboxes


def main():
    parser = argparse.ArgumentParser(description='Object Removal Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Path to save result')
    parser.add_argument('--bboxes', type=str, default=None,
                        help='Bounding boxes as "x1,y1,x2,y2;x1,y1,x2,y2"')
    parser.add_argument('--classes', type=str, default=None,
                        help='Class indices as "0,2,5"')
    parser.add_argument('--style', type=str, default='human',
                        choices=['human', 'art', 'graphic_design'],
                        help='Image style')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - select objects with mouse')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')

    args = parser.parse_args()

    print("=" * 60)
    print("Object Removal Inference")
    print("=" * 60)

    # Load image
    print(f"\n📷 Loading image from {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Cannot read image from {args.image}")
    print(f"✅ Image loaded: {image.shape}")

    # Get bounding boxes
    if args.interactive:
        print("\n🖱️  Interactive mode: Please select objects...")
        bboxes = interactive_bbox_selector(args.image)
        if not bboxes:
            print("❌ No objects selected. Exiting...")
            return
        # Assign default class 0 to all objects
        classes = [0] * len(bboxes)
    else:
        if args.bboxes is None:
            raise ValueError("Please provide --bboxes or use --interactive mode")

        # Parse bboxes
        bbox_strs = args.bboxes.split(';')
        bboxes = []
        for bbox_str in bbox_strs:
            coords = [int(x) for x in bbox_str.split(',')]
            bboxes.append(coords)

        # Parse classes
        if args.classes:
            classes = [int(x) for x in args.classes.split(',')]
        else:
            classes = [0] * len(bboxes)

    print(f"\n🎯 Objects to remove:")
    for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
        print(f"  {i + 1}. BBox: {bbox}, Class: {cls}")

    # Create object remover
    remover = ObjectRemover(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=args.device,
        img_size=args.img_size
    )

    # Remove objects
    print(f"\n🔄 Removing objects (style: {args.style})...")
    result, mask = remover.remove_objects(image, bboxes, classes, style=args.style)

    # Save result
    cv2.imwrite(args.output, result)
    print(f"✅ Result saved to {args.output}")

    # Save mask
    mask_path = args.output.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
    cv2.imwrite(mask_path, mask * 255)
    print(f"✅ Mask saved to {mask_path}")

    # Visualize
    if args.visualize:
        print(f"\n📊 Generating visualization...")
        vis_path = args.output.replace('.jpg', '_vis.jpg').replace('.png', '_vis.png')
        remover.visualize(image, mask, result, bboxes, save_path=vis_path)


if __name__ == '__main__':
    main()