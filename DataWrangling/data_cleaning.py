import re
sg_postcode_format = re.compile(r'^[0-9]{6}$')
my_postcode_format = re.compile(r'^[0-9]{5}$')

def update_data(data):
    for row in data[:]:
        if "address" in row.keys():
            if "postcode" in row["address"].keys():
                postcode = row["address"]["postcode"]
                # remove the Malaysian cities information
                if (my_postcode_format.search(postcode)):
                    data.remove(row)
                # remove postcode for cases like <different>
                elif not (sg_postcode_format.search(postcode)):
                    del row["address"]["postcode"]
    return data

def write_to_jsonfile(file_out,data):
    import json
    with open(file_out, 'w') as outfile:
        for row in data:
            json.dump(row, outfile, indent=2)

def read_from_jsonfile(inputfile):
    import json

    content = open(inputfile+".json", "r").read() 
    data = json.loads("[" + content.replace("}\n{", "},\n{") + "]")
            
    return data

def main2():
    filename = 'sample.osm'
    data2 = read_from_jsonfile(filename)
    data2 = update_data(data2)
    
    outputfile = filename + "2.json"
    write_to_jsonfile(outputfile, data2)
    print outputfile + " generated."

# comment it out to avoid accidental run
main2()

