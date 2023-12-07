import xml.etree.ElementTree as ET 
import numpy as np 

def contains_subtree(xml_node) -> bool:
    if xml_node.tag != "SubTree" and len(xml_node) == 0:
        return False
    if xml_node.tag == "SubTree":
        return True
    else:
        child_contains = False
        for child_xml in xml_node:
            child_contains = child_contains or contains_subtree(child_xml)
        return child_contains

def get_subtree_names(xml_node):
    if not contains_subtree(xml_node):
        return []
    elif xml_node.tag == "SubTree":
        return [xml_node.attrib["ID"]]
    else:
        name_list = []
        for child in xml_node:
            name_list.extend(get_subtree_names(child))
        return name_list

def cycle_free(xml_root):
    # get the node list
    trees = []
    for child in xml_root:
        if child.tag == "BehaviorTree":
            trees.append(child.attrib["ID"])

    g = SubtreeGraph(trees)

    # initialize the edges
    for child in xml_root:
        if child.tag != "BehaviorTree":
            continue
        subtrees = get_subtree_names(child)
        for subtree in subtrees:
            g.set_edge(child.attrib["ID"], subtree)

    # compute distances
    g.compute_connectivity()

    # for each node that has subtrees, see if can get back
    for child in xml_root:
        if child.tag != "BehaviorTree":
            continue
        subtrees = get_subtree_names(child)
        for subtree in subtrees:
            if g.get_edge(child.attrib["ID"], subtree) < np.inf:
                if g.get_edge(subtree, child.attrib["ID"]) < np.inf:
                    return False
    
    return True
        
        
class SubtreeGraph:
    '''
    A graph representing the SubTree dependencies in a collection
    of BehaviorTrees.

    A SubTree dependency exists from BehaviorTree i to BehaviorTree
    j if j appears as a SubTree in i.
    '''
    def __init__(self, nodes):
        self.nodes = {}
        for i, v in enumerate(nodes):
            self.nodes[v] = i

        self.n = len(nodes)
        self.dist = np.ones((self.n,self.n)) * np.inf

    def set_edge(self, v1, v2, val = 1):
        i = self.nodes[v1]
        j = self.nodes[v2]
        self.dist[i][j] = val

    def get_edge(self, v1, v2):
        i = self.nodes[v1]
        j = self.nodes[v2]
        return self.dist[i][j]

    def compute_connectivity(self):
        for v in self.nodes.keys():
            self.set_edge(v,v, 0)
        for k in range(0, self.n):
            for i in range(0, self.n):
                for j in range(0, self.n):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]




    