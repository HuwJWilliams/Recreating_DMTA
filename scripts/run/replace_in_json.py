
# %%
import os
import json

def replace_in_json_files(directory, old_str, new_str):
    # Walk through the directory recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a .json extension
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    # Open and load the JSON file
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Recursively search and replace all instances of old_str with new_str in both keys and values
                    def recursive_replace(obj):
                        if isinstance(obj, dict):
                            new_dict = {}
                            for key, value in obj.items():
                                # Check and replace the key if needed
                                new_key = key.replace(old_str, new_str) if isinstance(key, str) else key
                                # Check and replace the value if needed
                                new_value = recursive_replace(value)
                                new_dict[new_key] = new_value
                            return new_dict
                        elif isinstance(obj, list):
                            return [recursive_replace(item) for item in obj]
                        elif isinstance(obj, str):
                            return obj.replace(old_str, new_str)
                        return obj

                    # Perform the replacement
                    updated_data = recursive_replace(data)

                    # Only write the file if there was a change
                    if updated_data != data:
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(updated_data, f, indent=4)
                        print(f"Updated: {file_path}")
                    else:
                        print(f"No changes for: {file_path}")

                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")



# Set your search directory and the strings you want to replace
search_dir = "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/scrambled/"  # Replace with your directory path
replace_in_json_files(search_dir, "Pearson_r", "pearson_r")

search_dir =  "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/10_mol_sel/"
replace_in_json_files(search_dir, "Pearson_r", "pearson_r")

search_dir =  "/users/yhb18174/Recreating_DMTA/results/rdkit_desc/finished_results/50_mol_sel/"
replace_in_json_files(search_dir, "Pearson_r", "pearson_r")

print("Replacement done!")
# %%
