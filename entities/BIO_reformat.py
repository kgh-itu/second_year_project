import os

def reformat_files(directory):
    new_directory = "reformatted_generated_entities"
    os.makedirs(new_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            tag = filename.split('.')[0]  # Extract the tag from the filename
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, f"{tag}.txt")  # New file path with tag name

            # Read the file and split the lines into words
            with open(file_path, 'r', encoding='utf-8') as f:
                entities = [line.strip().split() for line in f if line.strip()]

            # Write the words back to the new file, each on a new line, with a corresponding BIO tag
            with open(new_file_path, 'w', encoding='utf-8') as f:
                for entity in entities:
                    for i, word in enumerate(entity):
                        # Assign the 'B-' tag to the first word and 'I-' to the rest
                        bio_tag = 'B-' if i == 0 else 'I-'
                        f.write(f"{word}\t{bio_tag}{tag}\n")
                    f.write("\n")  # Add an empty line after each entity

if __name__ == '__main__':
    reformat_files('./generated_entities/')
