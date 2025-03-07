import pandas as pd
import os

def create_sample_file(input_file, output_file, nrows=1000):
    """Create a sample file with the first nrows from the input file."""
    print(f"Creating sample for {os.path.basename(input_file)}...")
    df = pd.read_csv(input_file, nrows=nrows)
    df.to_csv(output_file, index=False)
    print(f"Created sample with {len(df)} rows at {output_file}")

def main():
    # Define paths
    raw_dir = "data/raw/RetailRocketDataset"
    sample_dir = "data/raw/samples"
    
    # Ensure sample directory exists
    os.makedirs(sample_dir, exist_ok=True)
    
    # List of files to sample
    files = [
        "category_tree.csv",
        "events.csv",
        "item_properties_part1.csv",
        "item_properties_part2.csv"
    ]
    
    # Create samples
    for file in files:
        input_path = os.path.join(raw_dir, file)
        output_path = os.path.join(sample_dir, file)
        create_sample_file(input_path, output_path)

if __name__ == "__main__":
    main()
