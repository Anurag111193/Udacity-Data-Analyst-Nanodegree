
import xml.etree.cElementTree as ET
import pprint

def count_tags(filename):
    tags = {}
    for ev,elem in ET.iterparse(filename):
        tag = elem.tag
        if tag not in tags.keys():
            tags[tag] = 1
        else:
            tags[tag]+=1
    return tags


def test():

    tags = count_tags('ahmedabad_india.osm')
    pprint.pprint(tags)
    #assert tags == {'bounds': 1,
    #                 'member': 3,
    #                 'nd': 4,
     #                'node': 20,
    #                 'osm': 1,
    #                 'relation': 1,
    #                 'tag': 7,
    #                 'way': 1}

    

if __name__ == "__main__":
    test()
