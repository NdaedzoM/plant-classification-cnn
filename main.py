from pathlib import Path
from load_data.dataloader import get_data_loaders
from eda import perform_eda
from model.train import train_and_evaluate

def main():

# Mount Drive

    output_dir = Path('/content/drive/My Drive/cnn/results')
    output_dir.mkdir(parents=True, exist_ok=True)


    print("Performing EDA...")
    class_aliases = perform_eda(output_dir)

    print("Loading data...")
    train_loader, val_loader, test_loader, classes, class_aliases, original_to_alias = get_data_loaders(device='cpu')
    num_classes = len(classes)

    print("Training and evaluating models...")
    results_df = train_and_evaluate(train_loader, val_loader, num_classes, output_dir, device='cpu')

    print("\nClass Aliases:", class_aliases)
    print("\nOriginal to Alias Mapping:", original_to_alias)
    print("\nFinal Results:")
    print(results_df)

if __name__ == '__main__':
    main()