import numpy as np

# Load your metadata
metadata = np.load('E:/AISOC/model-development/metadata.npy', allow_pickle=True).item()

print("=== METADATA INFORMATION ===")
print(f"Available keys: {list(metadata.keys())}")

# Extract category mapping
if 'category_mapping' in metadata:
    print("\nCATEGORY MAPPING:")
    category_mapping = metadata['category_mapping']
    print(f"Content: {category_mapping}")
    
    # Generate JavaScript object
    print("\n// Copy this CATEGORIES object to your app.js:")
    print("const CATEGORIES = {")
    if isinstance(category_mapping, dict):
        for idx, name in category_mapping.items():
            print(f"    {idx}: '{name}',")
    elif isinstance(category_mapping, list):
        for idx, name in enumerate(category_mapping):
            print(f"    {idx}: '{name}',")
    print("};")

# Extract subcategory mapping
if 'subcategory_mapping' in metadata:
    print("\nSUBCATEGORY MAPPING:")
    subcategory_mapping = metadata['subcategory_mapping']
    print(f"Content: {subcategory_mapping}")
    
    # Generate JavaScript object
    print("\n// Copy this SUBCATEGORIES object to your app.js:")
    print("const SUBCATEGORIES = {")
    if isinstance(subcategory_mapping, dict):
        for idx, name in subcategory_mapping.items():
            print(f"    {idx}: '{name}',")
    elif isinstance(subcategory_mapping, list):
        for idx, name in enumerate(subcategory_mapping):
            print(f"    {idx}: '{name}',")
    print("};")

# Print input shape and other info
for key, value in metadata.items():
    if key not in ['category_mapping', 'subcategory_mapping']:
        print(f"\n{key}: {value}")
