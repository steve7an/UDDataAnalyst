#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
import collections

address_type_direction = re.compile(r'^addr:[a-z]+:[a-z]+$')
address = re.compile(r'^addr:[a-z]+$')
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
ignored_regexs = [problemchars, address_type_direction]

my_postcode_format = re.compile(r'^[0-9]{5}$')


CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
SUBMAPPING = {"addr":"address"}

def process_refs(node, element):
    refs = []
    for nodes in element.iter('nd'):
        ref = long(nodes.attrib["ref"])
        refs.append(ref)
        #print (ref)

    if refs:
        node["node_refs"] = refs
        
    return node

def process_tags(node, element):
    tagvalues = []
    for tags in element.iter('tag'):
        #print tags.attrib
        attribk = tags.attrib["k"]
        if any(regex.search(attribk) for regex in ignored_regexs):
            #do nothing
            continue
        elif address.search(attribk) or lower_colon.search(attribk):
            addressparts = attribk.split(":")

            # differentiate between addr and the rest of the other sub tags
            if addressparts[0] in SUBMAPPING.keys():
                keyword = SUBMAPPING[addressparts[0]]
            else:
                keyword = addressparts[0]

            # add an additional check here to skip nodes / ways with Malaysian's postcode 
            attr_key = addressparts[1]
            attr_value = tags.attrib["v"]
                
            tagvalues.append([keyword, attr_key, attr_value])                
            #print ([keyword, addressparts[1], tags.attrib["v"]])

        else:
            node[attribk] = tags.attrib["v"]

    # append each sub tag back to the main node
    tagdict = {}
    for maintag, k, v in tagvalues:
        tagdict.setdefault(maintag, {})[k] = v

    for k,v in tagdict.iteritems():
        node[k] = v
    
    return node

def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node["type"] = element.tag
        createddict = {}
        pos = [0, 0]
        for k,v in element.attrib.iteritems():
            if k in CREATED:
                createddict[k] = v
            elif k == "lat":
                pos[0] = float(v)
            elif k == "lon":
                pos[1] = float(v)
            else:
                node[k] = v
            
        node["created"] = createddict    
        node["pos"] = pos
        
        node = process_tags(node, element)
        
        node = process_refs(node, element)
        
        return node
    else:
        return None

def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def main():
    # NOTE: if you are running this code on your computer, with a larger dataset, 
    # call the process_map procedure with pretty=False. The pretty=True option adds 
    # additional spaces to the output, making it significantly larger.
    input_file = 'sample.osm'
    data = process_map(input_file, True)
    print "{0}.json".format(input_file) + " generated."
    #pprint.pprint(data[0])

if __name__ == "__main__":
    main()