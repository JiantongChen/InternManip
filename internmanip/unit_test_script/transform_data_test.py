import pandas as pd
import json
import numpy as np
import os # Import os module

# Define base directories and modality JSON path
input_base_dir = '/PATH/TO/YOUR/DATA/robot_sim.PickNPlace/data/'
output_base_dir = '/PATH/TO/YOUR/DATA/robot_sim_converted.PickNPlace/data/'
modality_json_path = '/PATH/TO/YOUR/DATA/robot_sim.PickNPlace/meta/modality.json'

try:
    # Load the modality JSON file
    with open(modality_json_path, 'r') as f:
        modality_config = json.load(f)

    # Get the state and action mappings
    state_mapping = modality_config.get('state', {})
    if not state_mapping:
        print("Warning: Could not find 'state' key in modality JSON. State column will not be flattened.")

    action_mapping = modality_config.get('action', {})
    if not action_mapping:
         print("Warning: Could not find 'action' key in modality JSON. Action column will not be flattened.")

    print(f"Starting data transformation from {input_base_dir} to {output_base_dir} with flattening.")

    # Walk through the input directory
    for root, _, files in os.walk(input_base_dir):
        for file in files:
            if file.endswith('.parquet'):
                input_file_path = os.path.join(root, file)
                # Construct the relative path from the input base directory
                relative_path = os.path.relpath(input_file_path, input_base_dir)
                # Construct the corresponding output file path
                output_file_path = os.path.join(output_base_dir, relative_path)
                # Get the output directory
                output_dir = os.path.dirname(output_file_path)

                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory: {output_dir}")

                print(f"Processing file: {input_file_path}")

                try:
                    # Read the parquet file into a pandas DataFrame
                    df = pd.read_parquet(input_file_path)

                    # Create a new DataFrame to hold the transformed data
                    transformed_df = pd.DataFrame()

                    # Process each column in the original DataFrame
                    for column in df.columns:
                         if column == 'observation.state' and state_mapping:
                              print(f"Flattening '{column}'...")
                              # Process the 'observation.state' column
                              state_data = df[column]
                              # Prepare dictionaries to hold lists of sliced data for each subkey
                              sliced_state_data = {f'{column}.{subkey}': [] for subkey in state_mapping.keys()}

                              # Iterate through each row's state array
                              for state_array in state_data:
                                   if isinstance(state_array, (list, np.ndarray)):
                                        state_array = np.asarray(state_array)
                                        for subkey, indices in state_mapping.items():
                                             start = indices.get('start')
                                             end = indices.get('end')
                                             if start is not None and end is not None and isinstance(start, int) and isinstance(end, int):
                                                  # Ensure indices are within bounds before slicing
                                                  if 0 <= start <= end <= len(state_array):
                                                       sliced_state = state_array[start:end].tolist()
                                                       sliced_state_data[f'{column}.{subkey}'].append(sliced_state)
                                                  else:
                                                       # Handle out of bounds indices, append a placeholder (e.g., list of Nones or zeros)
                                                       # Need expected shape, which could come from metadata or first valid slice
                                                       print(f"Warning: State indices out of bounds for subkey '{subkey}' in file {file}. Appending None.")
                                                       sliced_state_data[f'{column}.{subkey}'].append(None) # Or np.zeros(expected_shape).tolist()
                                             else:
                                                  print(f"Warning: Invalid or missing start/end indices for state subkey '{subkey}' in mapping for file {file}.")
                                                  sliced_state_data[f'{column}.{subkey}'].append(None) # Append None if mapping is invalid
                                   else:
                                        # Handle cases where the cell is not a list or array
                                        for subkey in state_mapping.keys():
                                            sliced_state_data[f'{column}.{subkey}'].append(None) # Append None if original data is not array-like


                              # Add the new flattened columns to the transformed DataFrame
                              for flat_key, data_list in sliced_state_data.items():
                                   transformed_df[flat_key] = data_list
                              print(f"Successfully flattened '{column}'.")

                         elif column == 'action' and action_mapping:
                             print(f"Flattening '{column}'...")
                             # Process the 'action' column (assuming it's a top-level column)
                             action_data = df[column]
                             # Prepare dictionaries to hold lists of sliced data for each subkey
                             sliced_action_data = {f'{column}.{subkey}': [] for subkey in action_mapping.keys()}

                             # Iterate through each row's action array
                             for action_array in action_data:
                                  if isinstance(action_array, (list, np.ndarray)):
                                       action_array = np.asarray(action_array)
                                       for subkey, indices in action_mapping.items():
                                            start = indices.get('start')
                                            end = indices.get('end')
                                            if start is not None and end is not None and isinstance(start, int) and isinstance(end, int):
                                                  # Ensure indices are within bounds before slicing
                                                  if 0 <= start <= end <= len(action_array):
                                                       sliced_action = action_array[start:end].tolist()
                                                       sliced_action_data[f'{column}.{subkey}'].append(sliced_action)
                                                  else:
                                                       # Handle out of bounds indices, append a placeholder
                                                       print(f"Warning: Action indices out of bounds for subkey '{subkey}' in file {file}. Appending None.")
                                                       sliced_action_data[f'{column}.{subkey}'].append(None) # Or np.zeros(expected_shape).tolist()
                                            else:
                                                 print(f"Warning: Invalid or missing start/end indices for action subkey '{subkey}' in mapping for file {file}.")
                                                 sliced_action_data[f'{column}.{subkey}'].append(None) # Append None if mapping is invalid
                                  else:
                                       # Handle cases where the cell is not a list or array
                                       for subkey in action_mapping.keys():
                                           sliced_action_data[f'{column}.{subkey}'].append(None) # Append None if original data is not array-like

                             # Add the new flattened columns to the transformed DataFrame
                             for flat_key, data_list in sliced_action_data.items():
                                  transformed_df[flat_key] = data_list
                             print(f"Successfully flattened '{column}'.")

                         else:
                              # For any other column, copy it directly to the transformed DataFrame
                              transformed_df[column] = df[column]


                    # Save the transformed DataFrame to the new parquet file
                    # Ensure the directory exists before saving
                    if not os.path.exists(output_dir):
                         os.makedirs(output_dir)

                    transformed_df.to_parquet(output_file_path)

                    print(f"Successfully saved transformed data to: {output_file_path}")

                except Exception as e:
                    print(f"An error occurred while processing {input_file_path}: {e}")
                    # Continue processing other files even if one fails


    print("\nFinished data transformation process with flattening.")


except FileNotFoundError as e:
    print(f"Error: A required file or directory was not found - {e}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the modality file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 