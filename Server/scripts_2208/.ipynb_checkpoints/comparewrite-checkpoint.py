import os

directory = "/Collider/scripts_2208/data/raw/"
file1_name = "complete_TTH_M9_Alpha4_13.hepmc"
#file2_name = "full_ctb_TTH_M9_Alpha4_13.hepmc"
file2_name = "full_op_TTH_M9_Alpha4_13.hepmc"
output_file_name = "difference_com_op_10mil.txt"  # Name of the output file

file1_path = os.path.join(directory, file1_name)
file2_path = os.path.join(directory, file2_name)
output_file_path = os.path.join(directory, output_file_name)  # Path for the output file

# Initialize line counter
line_number = 1

# Initialize difference counter
difference_count = 0

# Open both files and output file
with open(file1_path, "r") as file1, open(file2_path, "r") as file2, open(output_file_path, "w") as output_file:
    # Read lines from both files
    lines1 = file1.readlines()
    lines2 = file2.readlines()

    # Compare lines
    for line1, line2 in zip(lines1, lines2):
        if line1 != line2:
            output_file.write(f"Difference found at line {line_number}:\n")
            output_file.write(f"Codigo Cristian: {line1.strip()}\n")
            output_file.write(f"Codigo Walter  : {line2.strip()}\n\n")
            difference_count += 1
        line_number += 1

    # If files have different numbers of lines
    if len(lines1) != len(lines2):
        output_file.write("Files have different numbers of lines.\n")

# Print total count of differences
print(f"Total differences found: {difference_count}")