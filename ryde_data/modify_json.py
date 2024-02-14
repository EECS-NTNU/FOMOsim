import gzip
import json

# Path to the original .json.gz file
input_file_path = 'ryde_data/status_change_2023_octdec.json'

# Decompress and load the JSON data
with open(input_file_path, 'rt', encoding='utf-8') as f:
    data = json.load(f)
#     if "data" in data:
#         data = data["data"]

# # Assuming the data is a list, split it into two parts
# midpoint = len(data) // 2
# data_part1 = data[:midpoint]
# data_part2 = data[midpoint:]

# Paths for the new .json.gz files
part1_gz_path = 'ryde_data/status_changes_octdec.json.gz'
# part2_gz_path = 'ryde_data/part2.json.gz'

# Compress and save the first part
with gzip.open(part1_gz_path, 'wt', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

# # Compress and save the second part
# with gzip.open(part2_gz_path, 'wt', encoding='utf-8') as f:
#     json.dump(data_part2, f, indent=4)