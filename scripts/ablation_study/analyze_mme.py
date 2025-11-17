import os
import pandas as pd
import re

# 1. Define the top-level directory and output filename
TOP_DIR = './eval_results/mme_eval_results'
OUTPUT_CSV = 'mme_summary_averages.csv'
# Columns for which to calculate the average
COLUMNS_TO_AVERAGE = ['mme_acc', 'mme_acc_plus']

# 2. Initialize the list to store results
results = []

# 3. Iterate through subfolders
print(f"Starting directory traversal: {TOP_DIR}")
if not os.path.exists(TOP_DIR):
    print(f"Error: Directory {TOP_DIR} does not exist. Please ensure the script is run from the correct location.")
else:
    # Iterate through all items in the TOP_DIR
    for folder_name in os.listdir(TOP_DIR):
        folder_path = os.path.join(TOP_DIR, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # 4. Attempt to parse alpha parameters from the folder name
            # Example folder name format: mme_eval_results_hybrid_attn_dif_hybrid_a0.0079b1.2607c1.7920
            # Pattern matches aX.XXXXbY.YYYYcZ.ZZZZZ
            match = re.search(r'a(\d+\.?\d*)b(\d+\.?\d*)c(\d+\.?\d*)', folder_name)

            if match:
                # Extract alpha1, alpha2, alpha3 (a, b, c)
                alpha1 = float(match.group(1))
                alpha2 = float(match.group(2))
                alpha3 = float(match.group(3))

                # 5. Locate the mme_summary.csv file
                csv_path = os.path.join(folder_path, 'mme_summary.csv')

                if os.path.exists(csv_path):
                    try:
                        # 6. Read the CSV file and calculate averages
                        df = pd.read_csv(csv_path)

                        # Check if all required columns exist
                        if all(col in df.columns for col in COLUMNS_TO_AVERAGE):
                            # Calculate the mean for the specified columns
                            avg_values = df[COLUMNS_TO_AVERAGE].mean().to_dict()

                            # 7. Structure the results
                            result_row = {
                                'alpha1': alpha1,
                                'alpha2': alpha2,
                                'alpha3': alpha3,
                                'mme_acc': avg_values['mme_acc'],
                                'mme_acc_plus': avg_values['mme_acc_plus']
                            }
                            results.append(result_row)
                            print(f"Successfully processed: {folder_name}")
                        else:
                            print(f"Skipped {folder_name}: mme_summary.csv is missing required columns.")

                    except Exception as e:
                        print(f"Error processing CSV file in {folder_name}: {e}")
                else:
                    print(f"Skipped {folder_name}: mme_summary.csv file not found.")
            else:
                print(f"Skipped {folder_name}: Could not parse alpha parameters (a, b, c).")

# 8. Write the results to a new CSV file
if results:
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print("\n--- Task Complete ---")
    print(f"Statistics successfully saved to file: {OUTPUT_CSV}")
    print(f"Total folders processed: {len(results)}.")
else:
    print("\nNo valid folders or data found for processing.")