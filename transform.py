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

if not os.path.isdir(os.getcwd() + "/proper_data"):
    os.mkdir("proper_data")



rewrite("data/users2.jsonl", "proper_data/users2.json")
rewrite("data/sessions2.jsonl", "proper_data/sessions2.json")
rewrite("data/deliveries2.jsonl", "proper_data/deliveries2.json")
rewrite("data/products2.jsonl", "proper_data/products2.json")