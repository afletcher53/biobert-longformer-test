import os

def split_file(input_file, output_prefix):
    # Get the size of the file
    file_size = os.path.getsize(input_file)
    
    # Calculate the midpoint
    mid_point = file_size // 2
    
    with open(input_file, 'rb') as f:
        # Read the first half
        first_half = f.read(mid_point)
        
        # Adjust the split point to the next newline character
        while f.read(1) != b'\n':
            first_half += f.read(1)
        
        # Read the second half
        second_half = f.read()
    
    # Write the first half
    with open(f"{output_prefix}_part1.txt", 'wb') as f:
        f.write(first_half)
    
    # Write the second half
    with open(f"{output_prefix}_part2.txt", 'wb') as f:
        f.write(second_half)

# Use the function for both files
split_file('pubmed_train.txt', 'pubmed_train')
split_file('pubmed_val.txt', 'pubmed_val')

print("Files have been split successfully.")