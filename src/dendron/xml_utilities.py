import xml.etree.ElementTree as ET 
import numpy as np 

from typing import List

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

def get_subtree_names(xml_node) -> List[str]:
    if not contains_subtree(xml_node):
        return []
    elif xml_node.tag == "SubTree":
        return [xml_node.attrib["ID"]]
    else:
        name_list = []
        for child in xml_node:
            name_list.extend(get_subtree_names(child))
        return name_list

def cycle_free(xml_root) -> bool:
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
    has_cycle = False
    for child in xml_root:
        if child.tag != "BehaviorTree":
            continue
        subtrees = get_subtree_names(child)
        for subtree in subtrees:
            i = g.nodes[child.attrib["ID"]]
            j = g.nodes[subtree]
            if g.dist[i,j] < np.inf and g.dist[j,i] < np.inf:
                has_cycle = True
    
    return not has_cycle # it's called cycle_free
    
def get_parse_order(xml_root) -> List:
    trees = []
    for child in xml_root:
        if child.tag == "BehaviorTree":
            trees.append(child.attrib["ID"])
    
    g = SubtreeGraph(trees)

    for child in xml_root:
        if child.tag != "BehaviorTree":
            continue
        subtrees = get_subtree_names(child)
        for subtree in subtrees:
            g.set_edge(child.attrib["ID"], subtree)

    return g.topological_sort()
        
class SubtreeGraph:
    '''
    A graph representing the SubTree dependencies in a collection
    of BehaviorTrees.

    A SubTree dependency exists from BehaviorTree i to BehaviorTree
    j if j appears as a SubTree in i.
    '''
    def __init__(self, nodes) -> None:
        self.nodes = {}
        self.reverse_nodes = {}
        for i, v in enumerate(nodes):
            self.nodes[v] = i
            self.reverse_nodes[i] = v

        self.n = len(nodes)
        self.adjacency = np.zeros((self.n, self.n))
        self.dist = np.ones((self.n,self.n)) * np.inf

        for i in range(self.n):
            self.dist[i,i] = 0

    def topological_sort(self) -> List:
        output = [] # sorted list
        S = [] # nodes with no incoming edge

        # we need to reverse the dependency relation here. 
        digraph = self.adjacency.copy().T

        for j in range(0, self.n):
            if not np.any(digraph[:,j]):
                S.append(j)

        while len(S) > 0:
            node = S.pop()
            output.append(node)

            for i in range(0, self.n):
                if digraph[node, i] == 1:
                    digraph[node,i] = 0
                    if not np.any(digraph[:,i]):
                        S.append(i)

        return [self.reverse_nodes[k] for k in output]

    def set_edge(self, v1, v2, val = 1) -> None:
        i = self.nodes[v1]
        j = self.nodes[v2]
        self.adjacency[i,j] = val
        self.dist[i,j] = val

    def get_edge(self, v1, v2) -> float:
        i = self.nodes[v1]
        j = self.nodes[v2]
        return self.adjacency[i][j]

    def compute_connectivity(self) -> None:
        for k in range(0, self.n):
            for i in range(0, self.n):
                for j in range(0, self.n):
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]
        




    