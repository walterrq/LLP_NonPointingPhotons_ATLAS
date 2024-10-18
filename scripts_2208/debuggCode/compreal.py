# File path
file_path = "difference_cristian_walter.txt"

# Initialize dictionary to store vectors
vectors = {}

# Initialize list to store differences
differences = []

percent_diff_t = 0.

percent_diff_z = 0.

percent_diff_t_vector = []

percent_diff_z_vector = []
z_wc = []

def calculate_mean(value1, value2):
    return (value1 + value2) / 2
    
# Open the file and read all lines
with open(file_path, "r") as file:
    # Read all lines from the file
    lines = file.readlines()

    # Iterate over each line
    for i in range(len(lines)):
        # Check if the line contains "Difference found"
        if "Difference found" in lines[i]:
            # Extract the line number as the identifier
            identifier = int(lines[i].split()[-1].rstrip(':'))

            # Extract and split the lines for Cristian and Walter
            cristian_line_parts = lines[i+1].strip().split()[3:]
            walter_line_parts = lines[i+2].strip().split()[3:]

            # Store vectors in the dictionary
            vectors[identifier] = {'cristian': cristian_line_parts,
                                   'walter': walter_line_parts}

total_keys = len(vectors)
print("Total number of difference between Crisitian hepmc and Walter hepmc:", total_keys)

# Compare last elements of vectors and store differences

for key, value in vectors.items():
    cristian_last = float(value['cristian'][-1])
    walter_last = float(value['walter'][-1])

    z_cristian = float(value['cristian'][-2])
    z_walter = float(value['walter'][-2])

    z_wc.append((z_cristian, z_walter))

    # Calculate the percentage difference if cristian_last is not zero
    if cristian_last != 0:
        
        percent_diff_t = abs((cristian_last - walter_last) / calculate_mean(cristian_last,walter_last)) * 100
        percent_diff_t_vector.append(percent_diff_t)

        percent_diff_z = abs((z_cristian - z_walter) / calculate_mean(z_cristian,z_walter)) * 100
        percent_diff_z_vector.append(percent_diff_z)
        # Check if the difference exceeds 5%
        # abs(cristian_last) > 1e-08
        if percent_diff_t > 1e-12:
            differences.append((cristian_last, walter_last, z_cristian,z_walter))

# Print differences
#differences_z = differences_z = differences + z_wc

#print(differences_z)
#print("Differences:")

# Find the maximum value
max_value_t = max(percent_diff_t_vector)

max_value_z = max(percent_diff_z_vector)

# Print the maximum value
print(f"The maximum value in the difference tcristian vs twalter is {max_value_t} %")

print(f"The maximum value in the difference zcristian vs is {max_value_z} %")

#for cristian_last, walter_last, z_cristian, z_walter in differences:
#    print(f"Cristiant: {cristian_last}, Waltert: {walter_last}, Cristianz: {z_cristian}, Walterz: {z_walter}")

# Print total count of differences
print(f"Total differences in t that are superior than  1e-12%: {len(differences)}")









