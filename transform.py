import os
import os.path

def rewrite(filename, write_filename):

    final_string = ""    
    with open(filename, "r") as read_file, open(write_filename, "w") as write_file:
        final_string += "["
        for line in read_file:
            final_string += (line + ",\n")
        
        # last comma doesn't count
        final_string = final_string[:-2]
        final_string += "]"

        write_file.write(final_string)

def transform_all_data(base_folder_raw, base_folder_new):

    for filename in os.listdir(os.getcwd() + f"{base_folder_raw}"):
        if filename.endswith('.jsonl'):
            new_filename = filename.rstrip(".jsonl") + ".json"
            rewrite(
                os.getcwd() + f"{base_folder_raw}{os.sep}{filename}",
                os.getcwd() + f"{base_folder_new}{os.sep}{new_filename}"
            )
