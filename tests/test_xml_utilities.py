from dendron.xml_utilities import *
import random
from typing import List

def after_in_array(arr, u, v):
    '''
    Returns true if u comes after v in arr
    '''
    return arr.index(u) >= arr.index(v)

def reverse_topologically_sorted(g : SubtreeGraph, l : List[int]) -> bool:
    n = g.n 
    for i in range(n):
        for j in range(n):
            if g.adjacency[i,j] == 1:
                i_node = g.reverse_nodes[i]
                j_node = g.reverse_nodes[j]
                if not after_in_array(l, i_node, j_node):
                    return False
    return True

def test_cycle_free1():
    g = SubtreeGraph([5,11,2,7,8,9,3,10])
    g.set_edge(5,11)
    g.set_edge(11, 2)
    g.set_edge(7, 8)
    g.set_edge(8,9)
    g.set_edge(3,10)
    g.set_edge(7,11)
    g.set_edge(3,8)
    g.set_edge(11,9)
    g.set_edge(11,10)
    
    g.compute_connectivity()

    has_cycle = False
    for node1 in g.nodes:
        for node2 in g.nodes:
            if node1 == node2:
                continue
            i = g.nodes[node1]
            j = g.nodes[node2]
            if g.dist[i,j] < np.inf:
                if g.dist[j,i] < np.inf:
                    has_cycle = True

    assert not has_cycle

def test_topological_sort1():
    g = SubtreeGraph([5,11,2,7,8,9,3,10])
    g.set_edge(5,11)
    g.set_edge(11, 2)
    g.set_edge(7, 8)
    g.set_edge(8,9)
    g.set_edge(3,10)
    g.set_edge(7,11)
    g.set_edge(3,8)
    g.set_edge(11,9)
    g.set_edge(11,10)

    out = g.topological_sort()
    assert reverse_topologically_sorted(g, out)


def test_topological_sort2():
    g = SubtreeGraph([0,1,2,3,4])
    g.set_edge(0,1)
    g.set_edge(0,2)
    g.set_edge(0,4)
    g.set_edge(1,2)
    g.set_edge(1,3)
    g.set_edge(2,4)
    
    out = g.topological_sort()
    assert reverse_topologically_sorted(g, out)

def test_cycle_free_xml0():
    xml_tree = ET.parse("tests/data/TestTree0.xml")
    xml_root = xml_tree.getroot()
    result = cycle_free(xml_root)

    assert result

def test_cycle_free_xml1():
    xml_tree = ET.parse("tests/data/TestTree2.xml")
    xml_root = xml_tree.getroot()
    assert not cycle_free(xml_root)

    