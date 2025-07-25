import json
import numpy as np
import os
from collections import OrderedDict # Import OrderedDict

# Define file paths
modality_json_path = '/PATH/TO/YOUR/DATA/robot_sim_converted.PickNPlace/meta/modality.json'
stats_json_path = '/PATH/TO/YOUR/DATA/robot_sim_converted.PickNPlace/meta/stats.json'
output_stats_json_path = '/PATH/TO/YOUR/DATA/robot_sim_converted.PickNPlace/meta/stats_1.json'

try:
    # Load the modality JSON file
    with open(modality_json_path, 'r') as f:
        modality_config = json.load(f)

    # Get the state and action mappings, preserving their order if possible
    # The order from the JSON file is usually preserved in Python dictionaries (3.7+)
    state_mapping = modality_config.get('state', {})
    if not state_mapping:
        print("Warning: Could not find 'state' key in modality JSON. State statistics will not be transformed.")

    action_mapping = modality_config.get('action', {})
    if not action_mapping:
         print("Warning: Could not find 'action' key in modality JSON. Action statistics will not be transformed.")

    # Load the stats JSON file
    with open(stats_json_path, 'r') as f:
        stats_data = json.load(f)

    # Use OrderedDict to maintain the insertion order of keys
    transformed_stats = OrderedDict()

    # Iterate through the keys of the original stats_data to maintain order
    for original_key, original_value in stats_data.items():

        if original_key == 'observation.state' and state_mapping:
            print(f"Transforming statistics for '{original_key}'...")
            original_state_stats = original_value # This is the dictionary of stats (mean, std, etc.)

            # Iterate through the subkeys defined in the state mapping (order is maintained)
            for subkey, indices in state_mapping.items():
                start = indices.get('start')
                end = indices.get('end')

                # Handle cases where start/end might be missing or not integers or negative
                if start is None or end is None or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < 0:
                    print(f"Warning: Skipping state subkey '{subkey}' due to invalid or missing start/end indices in mapping.")
                    continue

                # Create a dictionary to hold the stats for this specific subkey
                subkey_stats = {}

                # Iterate through the statistical measures (mean, std, etc.) for the original state (order is maintained)
                for stat_name, stat_array in original_state_stats.items():
                     # Ensure the original stat_array is a list or numpy array
                     if not isinstance(stat_array, (list, np.ndarray)):
                          print(f"Warning: Stat '{stat_name}' for original key '{original_key}' is not a list or array. Skipping slicing for subkey '{subkey}'.")
                          continue

                     # Ensure the original stat_array is long enough for the indices
                     if end > len(stat_array):
                           print(f"Warning: Indices out of bounds for stat '{stat_name}' and state subkey '{subkey}'. Array length: {len(stat_array)}, start: {start}, end: {end}. Skipping this slice.")
                           continue

                     # Slice the array based on the subkey's start and end indices
                     # Convert to numpy array for slicing, then back to list for JSON
                     sliced_data = np.asarray(stat_array)[start:end].tolist()
                     subkey_stats[stat_name] = sliced_data

                # Add the collected subkey stats to the transformed_stats with the new key format
                transformed_stats[f"{original_key}.{subkey}"] = subkey_stats

            print(f"Successfully transformed statistics for '{original_key}'.")

        elif original_key == 'action' and action_mapping:
            print(f"Transforming statistics for '{original_key}'...")
            original_action_stats = original_value # This is the dictionary of stats (mean, std, etc.)

            # Iterate through the subkeys defined in the action mapping (order is maintained)
            for subkey, indices in action_mapping.items():
                start = indices.get('start')
                end = indices.get('end')

                # Handle cases where start/end might be missing or not integers or negative
                if start is None or end is None or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < 0:
                    print(f"Warning: Skipping action subkey '{subkey}' due to invalid or missing start/end indices in mapping.")
                    continue

                # Create a dictionary to hold the stats for this specific subkey
                subkey_stats = {}

                # Iterate through the statistical measures (mean, std, etc.) for the original action (order is maintained)
                for stat_name, stat_array in original_action_stats.items():
                     # Ensure the original stat_array is a list or numpy array
                     if not isinstance(stat_array, (list, np.ndarray)):
                          print(f"Warning: Stat '{stat_name}' for original key '{original_key}' is not a list or array. Skipping slicing for subkey '{subkey}'.")
                          continue

                     # Ensure the original stat_array is long enough for the indices
                     if end > len(stat_array):
                           print(f"Warning: Indices out of bounds for stat '{stat_name}' and action subkey '{subkey}'. Array length: {len(stat_array)}, start: {start}, end: {end}. Skipping this slice.")
                           continue

                     # Slice the array based on the subkey's start and end indices
                     # Convert to numpy array for slicing, then back to list for JSON
                     sliced_data = np.asarray(stat_array)[start:end].tolist()
                     subkey_stats[stat_name] = sliced_data

                # Add the collected subkey stats to the transformed_stats with the new key format
                transformed_stats[f"{original_key}.{subkey}"] = subkey_stats

            print(f"Successfully transformed statistics for '{original_key}'.")

        else:
            # For any other key, just copy it directly to the transformed_stats,
            # preserving its position from the original data.
            transformed_stats[original_key] = original_value


    # Save the transformed stats data to a new JSON file
    with open(output_stats_json_path, 'w') as f:
        json.dump(transformed_stats, f, indent=4)

    print(f"\nSuccessfully saved transformed statistics to: {output_stats_json_path}")

except FileNotFoundError as e:
    print(f"Error: A required file was not found - {e}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from one of the input files.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 