import os

directory = "/Collider/scripts_2208/data/raw/"
file1_name = "complete_TTH_M9_Alpha4_13.hepmc"
file2_name = "full_ctb_TTH_M9_Alpha4_13.hepmc"
#file2_name = "full_TTH_M9_Alpha4_13.hepmc"

file1_path = os.path.join(directory, file1_name)
file2_path = os.path.join(directory, file2_name)

# Initialize line counter
line_number = 1

# Open both files
with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
    # Read lines from both files
    lines1 = file1.readlines()
    lines2 = file2.readlines()

    # Compare lines
    for line1, line2 in zip(lines1, lines2):
        if line1 != line2:
            print(f"Difference found at line {line_number}:")
            print(f"Codigo Cristian: {line1.strip()}")
            print(f"Codigo Walter  : {line2.strip()}")
            print()
        line_number += 1

# If files have different numbers of lines
if len(lines1) != len(lines2):
    print("Files have different numbers of lines.")
