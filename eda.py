from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from load_data.dataloader import TrainDataset, get_transform

import seaborn as sns
import numpy as np
from PIL import Image

def perform_eda(output_dir: Path = Path('eda_plots')):
    """Perform exploratory data analysis and save plots."""
    train_dir = Path('/content/drive/My Drive/cnn/data/train')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset
    dataset = TrainDataset(train_dir, transform=get_transform())
    classes = dataset.class_names
    class_aliases = dataset.class_aliases

    # Validate class names
    missing_aliases = [cls for cls in classes if cls not in class_aliases]
    if missing_aliases:
        raise KeyError(f"Missing aliases for classes: {missing_aliases}")

    # Class distribution
    class_counts = [len(list((train_dir / cls).glob('*.[jp][pn]g'))) for cls in classes]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[class_aliases[cls] for cls in classes], y=class_counts)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()

    # Sample images
    plt.figure(figsize=(15, 5))
    for i, cls in enumerate(classes[:5]):
        img_path = next((train_dir / cls).glob('*.[jp][pn]g'), None)
        if img_path:
            img = Image.open(img_path).convert('RGB')
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(class_aliases[cls])
            plt.axis('off')
    plt.savefig(output_dir / 'sample_images.png')
    plt.close()

    # Image statistics
    dimensions = []
    for cls in classes:
        for img_path in list((train_dir / cls).glob('*.[jp][pn]g'))[:50]:
            img = Image.open(img_path)
            dimensions.append(img.size)
    widths, heights = zip(*dimensions)
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Image Dimensions')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.savefig(output_dir / 'image_dimensions.png')
    plt.close()

    print(f"EDA plots saved in {output_dir}")
    return class_aliases