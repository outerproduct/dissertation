#!/usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
import copy


class GraphElement(object):
    """A virtual class that can take attributes and offset attributes
    which represent indices by scalar values. This property is useful in
    combining and dividing graphs across Sentences and MultiSentences.
    This class is expected to be only used in inheritance and not
    instantiated directly.
    """
    def add_attribute(self, name, value, upstream=True):
        """Add a single attribute to the element but warn when collisions
        occur.
        """
        if hasattr(self, name):
            # If the attribute and update are mutable, we merge them.
            attrib = getattr(self, name)
            if isinstance(attrib, list) and isinstance(value, list):
                attrib.extend(value)
            elif isinstance(attrib, set) and isinstance(value, set):
                attrib.update(value)
            elif isinstance(attrib, dict) and isinstance(value, dict):
                attrib.update(value)
            else:
                print "WARNING: Attribute name \'", name,
                print "\' already in use for element",
                print self.to_text()

                # Overwrite anyway
                setattr(self, name, copy.copy(value))
        else:
            # Copy to avoid bugs with mutable values
            setattr(self, name, copy.copy(value))

        # Also add the attribute upstream if a backlink is present
        if upstream and hasattr(self, '_backlink'):
            self._backlink.add_attribute(name, value, upstream=upstream)

    def add_attributes(self, upstream=True, **kwargs):
        """Add multiple attributes with overwrite warnings.
        """
        for key, value in kwargs.iteritems():
            self.add_attribute(key, value, upstream=upstream)

    def offset_idxs(self, offset, backlink=True):
        """Return a version of the element with the indices offset by some
        value. This is necessary to support the annotation descriptor class
        AnnotationDesc which is needed for compatibility with MultiSentences.
        We assume that attributes which specify indices are labeled sanely
        with 'idx' or 'idxs' suffixes.
        """
        # Removed shortcut to always copy
        # if offset == 0:
        #    return self

        # Reflect another instance of the caller.
        # Note: using __new__ instead of regular instance creation to
        # avoid initialization of members that will be instantly overwritten
        selfcls = self.__class__
        offset_element = selfcls.__new__(selfcls)

        # Offset the indices of the new element using the given value
        for key, val in self.__dict__.iteritems():
            if key.endswith('idx'):
                # Assume a single integer index or None
                setattr(offset_element, key, val + offset
                        if val is not None else None)
            elif key.endswith('idxs'):
                # Assume a list of integer indices
                setattr(offset_element, key, [idx + offset for idx in val])
            else:
                # Assume something else that should just be copied over
                setattr(offset_element, key, val)

        # Set an additional attribute linking the offset element to the
        # original element.
        # TODO: Should we use weakref for this? Doesn't seem to cause cycles.
        if backlink:
            offset_element._backlink = self

        return offset_element

    def to_text(self, *args, **kwargs):
        """Return a dummy element indicator since this is expected to be
        subclassed.
        """
        print "WARNING: Calling to_text() for virtual graph element"
        return "[?]"


class Node(GraphElement):
    """A node just consists of a collection of user-specified attributes
    and structure-specific attributes for easy traversal.
    """
    def __init__(self, idx, **kwargs):
        """Initialize the node using its token index and optional keyword
        attributes. Public attributes should be independent of the relations
        that the node participates in. Relation-aware attributes are tied
        to a particular graph structure and should only be accessed via
        methods exposed by the corresponding structure.
        """
        self.idx = idx
        self.outgoing_edges = {}
        self.incoming_edges = {}
        self.__dict__.update(kwargs)

    def add_outgoing_edge(self, edge, warnings=True):
        """Add an outgoing edge to the node.
        """
        if warnings and edge.tgt_idx in self.outgoing_edges:
            print "WARNING: Overwriting duplicate outgoing edge",
            print edge.to_text(), "from node", self.to_text()
        self.outgoing_edges[edge.tgt_idx] = edge

    def add_incoming_edge(self, edge, warnings=True):
        """Add an incoming edge to the node.
        """
        if warnings and edge.src_idx in self.incoming_edges:
            print "WARNING: Overwriting duplicate incoming edge",
            print edge.to_text(), "to node", self.to_text()
        self.incoming_edges[edge.src_idx] = edge

    def get_incoming_attribs(self, attrib_name):
        """Return the value of a particular attribute from an incoming edge.
        """
        return [getattr(edge, attrib_name)
                for edge in self.incoming_edges.itervalues()]

    def get_outgoing_attribs(self, attrib_name):
        """Return the value of a particular attribute from an outgoing edge.
        """
        return [getattr(edge, attrib_name)
                for edge in self.outgoing_edges.itervalues()]

    def to_text(self, attribute=None):
        """Return the node index and an optional attribute for printing.
        """
        # Check if the attribute value is present in the node
        ret_str = "[%d]" % (self.idx,)
        if attribute is not None and hasattr(self, attribute):
            ret_str += " %s" % (str(getattr(self, attribute)),)
        return ret_str


class Edge(GraphElement):
    """A directed edge in a structure covers a source and target node and
    has attributes specifying the relationship between them.
    """
    def __init__(self, src_idx, tgt_idx, **kwargs):
        """Initialize edge from source and target indices and optional keyword
        attributes.
        """
        self.src_idx = src_idx
        self.tgt_idx = tgt_idx
        self.__dict__.update(kwargs)

    def __hash__(self):
        """Return a hash value for edges which are defined to be immutable.
        """
        return hash((self.src_idx, self.tgt_idx))

    def __eq__(self, other):
        """Check if two edges are identical.
        """
        return self.src_idx == other.src_idx and self.tgt_idx == other.tgt_idx

    def __ne__(self, other):
        """Check if two edits are not identical.
        """
        return self.src_idx != other.src_idx or self.tgt_idx != other.tgt_idx

    def offset_idxs(self, offset):
        """Return a version of the edge with the indices offset by some value.
        This is necessary to support the annotation descriptor class
        AnnotationDesc which is needed for compatibility with MultiSentences.
        We assume that attributes which specify indices are labeled sanely with
        'idx' or 'idxs' suffixes.
        """
        if offset == 0:
            return self

        # Note: using __new__ instead of regular instance creation to
        # avoid initialization of members that will be instantly overwritten
        offset_edge = Edge.__new__(Edge)
        #offset_node.__dict__.update(self.__dict__)

        # Offset the indices of the new node using the given value
        for key, val in self.__dict__.iteritems():
            if key.endswith('idx'):
                # Assume a single integer index
                setattr(offset_edge, key, val + offset)
            elif key.endswith('idxs'):
                # Assume a list of integer indices
                setattr(offset_edge, key, [idx + offset for idx in val])
            else:
                # Assume something else that should just be copied over
                setattr(offset_edge, key, val)
        return offset_edge

    def to_text(self, ext_nodes=None, attribute=None):
        """Return the edge indices and an optional attribute for printing.
        """
        # Check if the attribute value is present in the edge
        if ext_nodes is None:
            ret_str = "[%d] -> [%d]" % (self.src_idx, self.tgt_idx)
        else:
            ret_str = "[%d] %s -> [%d] %s" % (self.src_idx,
                                              ext_nodes[self.src_idx],
                                              self.tgt_idx,
                                              ext_nodes[self.tgt_idx])

        if attribute is not None and hasattr(self, attribute):
            ret_str += "  (%s)" % (str(getattr(self, attribute)),)
        return ret_str


class DependencyGraph(object):
    """A dependency graph consists of a collection of word-linked nodes and
    relations between them. No root is necessary.

    NOTE: This is currently not supported by MultiSentences.
    """
    def __init__(self, num_tokens, **kwargs):
        """Initialize the graph with a given number of nodes, associated
        with the tokens of a sentence, along with default arguments to each.
        """
        self.__dict__.update(kwargs)
        self.nodes = []
        self.edges = []

        # By design, the first T nodes must be associated with the T tokens in
        # the corresponding Sentence object.
        for t in range(num_tokens):
            self.add_node()

        # Note the indices of token nodes and non-token auxiliary nodes
        self.token_idxs = list(range(num_tokens))
        self.aux_idxs = []

    def add_node(self, **kwargs):
        """Add a node.
        """
        new_node = Node(len(self.nodes), **kwargs)
        self.nodes.append(new_node)
        return new_node

    def add_aux_node(self, **kwargs):
        """Add an auxiliary node.
        """
        aux_node = self.add_node(**kwargs)
        self.aux_idxs.append(aux_node.idx)
        return aux_node

    def add_edge(self, src, tgt, **kwargs):
        """Add an edge directly by identifying the source and target nodes
        or indices of nodes.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)

        new_edge = Edge(src.idx, tgt.idx, **kwargs)
        src.add_outgoing_edge(new_edge)
        tgt.add_incoming_edge(new_edge)
        self.edges.append(new_edge)
        return new_edge

    def update_edge(self, src, tgt, **kwargs):
        """Update the parameters of an edge or, if it doesn't exist, add a
        new edge by identifying the source and target nodes or their indices.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)

        if tgt.idx in src.outgoing_edges:
            edge = src.outgoing_edges[tgt.idx]
            edge.add_attributes(**kwargs)
            return edge
        else:
            return self.add_edge(src, tgt, **kwargs)

    def has_edge(self, src, tgt):
        """Return whether the given edge alredy exists in the graph.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)
        return tgt.idx in src.outgoing_edges

    def get_token_nodes(self):
        """Return only the token nodes.
        """
        return [self.nodes[idx] for idx in self.token_idxs]

    def get_aux_nodes(self):
        """Return only the auxiliary nodes.
        """
        return [self.nodes[idx] for idx in self.aux_idxs]

    def get_edge(self, src, tgt):
        """Get edge directly by identifying the source and target nodes
        or indices of nodes.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)
        return src.outgoing_edges[tgt.idx]

    def reset_edge_refs(self):
        """Reset the outgoing and incoming edge references in each node
        to link to the edges present. This is useful if edges and nodes
        are desynchronized through external manipulation.
        """
        for node in self.nodes:
            node.outgoing_edges = {}
            node.incoming_edges = {}

        for edge in self.edges:
            self.nodes[edge.src_idx].add_outgoing_edge(edge, warnings=False)
            self.nodes[edge.tgt_idx].add_incoming_edge(edge, warnings=False)

    def get_nodes(self, node_idxs):
        """Return a list of nodes corresponding to the given node indices.
        """
        return [self.nodes[idx] for idx in node_idxs]

    def to_node(self, node_or_idx):
        """Map a given node index to the corresponding node. If a node is
        supplied, leave it unchanged.
        """
        if isinstance(node_or_idx, int):
            return self.nodes[node_or_idx]
        else:
            return node_or_idx

    def to_text(self, ext_nodes=None, attribute=None):
        """Return a textual representation of the graph.
        """
        if ext_nodes is not None:
            assert len(ext_nodes) == len(self.nodes)

        return '\n'.join(edge.to_text(ext_nodes=ext_nodes,
                                      attribute=attribute)
                         for edge in self.edges)


class DependencyDag(DependencyGraph):
    """A dependency DAG includes root nodes and options for traversal
    other than random walks.
    """
    def __init__(self, num_tokens, **kwargs):
        """Initialize the list of nodes and edges, along with an empty
        list of root nodes.
        """
        # Note the indices of root nodes, i.e., nodes without parents
        self.root_idxs = set(range(num_tokens))

        DependencyGraph.__init__(self, num_tokens, **kwargs)

    def add_node(self, **kwargs):
        """Add a node.
        """
        new_node = DependencyGraph.add_node(self, child_idxs=set(),
                parent_idxs=set(), **kwargs)
        self.root_idxs.add(new_node.idx)
        return new_node

    def add_edge(self, src, tgt, **kwargs):
        """Add an edge directly by identifying the source and target nodes
        or indices of nodes.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)

        # We must no longer consider the target node as a root node
        if tgt.idx in self.root_idxs:
            self.root_idxs.remove(tgt.idx)

        new_edge = DependencyGraph.add_edge(self, src, tgt, **kwargs)
        src.child_idxs.add(tgt.idx)
        tgt.parent_idxs.add(src.idx)
        return new_edge

    def is_root(self, node):
        """Return whether the given node or index is a root.
        """
        if isinstance(node, int):
            return node in self.root_idxs
        else:
            return node.idx in self.root_idxs

    def is_leaf(self, node):
        """Return whether the given node or index is a leaf node.
        """
        if isinstance(node, int):
            self.nodes[node].child_idxs == 0
        else:
            return node.child_idxs == 0

    def get_roots(self):
        """Get all the root nodes.
        """
        return [self.nodes[idx] for idx in self.root_idxs]

    def get_root_idxs(self):
        """Get indices of all the root nodes.
        """
        return self.root_idxs[:]

    def get_parents(self, node):
        """Get the parents of a node.
        """
        return [self.nodes[idx] for idx in node.parent_idxs]

    def get_parent_idxs(self, node_idx, copy=True):
        """Get the indices of a node's parents given the node's index.
        """
        return self.nodes[node_idx].parent_idxs.copy() if copy \
                else self.nodes[node_idx].parent_idxs

    def get_num_parents(self, node_idx):
        """Get the number of parents of a node given its index.
        """
        return len(self.get_parent_idxs(node_idx, copy=False))

    def get_children(self, node):
        """Get the children of a node.
        """
        return [self.nodes[idx] for idx in node.child_idxs]

    def get_child_idxs(self, node_idx, copy=True):
        """Get the indices of a node's children given the node's index.
        """
        return self.nodes[node_idx].child_idxs.copy() if copy \
                else self.nodes[node_idx].child_idxs

    def get_num_children(self, node_idx):
        """Get the number of children of a node given its index.
        """
        return len(self.get_child_idxs(node_idx, copy=False))

    def get_siblings(self, node):
        """Get the siblings of a node where a sibling is defined as the
        child of a parent. Avoids duplicates.
        """
        return [self.nodes[idx] for idx in self.get_sibling_idxs(node.idx)]

    def get_sibling_idxs(self, node_idx):
        """Get the siblings of a node where a sibling is defined as the
        child of a parent. Avoids duplicates.
        """
        sibling_idxs = set()
        for parent_idx in self.nodes[node_idx].parent_idxs:
            sibling_idxs.update(self.nodes[parent_idx].child_idxs)
        if len(sibling_idxs) > 0:
            sibling_idxs.remove(node_idx)
        return sibling_idxs

    def get_num_siblings(self, node_idx):
        """Get the number of siblings of a node given its index.
        """
        return len(self.get_sibling_idxs(node_idx))

    def get_traversal(self, node, ascending=False, deduplicate=True,
            idxs=False, limit=None):
        """Return a subgraph rooted at this node via a depth first traversal.
        Duplicates will be retained when the subgraph is not a proper tree.

        WARNING: doesn't check for cycles.
        WARNING: deduplicating currently changes the order of the traversal.
        """
        traversed = []
        successors = self.get_parents(node) if ascending else \
                     self.get_children(node)
        if limit is not None:
            limit -= 1
        if limit is None or limit >= 0:
            for successor in successors:
                traversed.append(successor)
                traversed.extend(self.get_traversal(successor,
                                                    ascending=ascending,
                                                    deduplicate=False,
                                                    idxs=False,
                                                    limit=limit))

        if deduplicate:
            # NOTE: doesn't preserve the order of the traversed nodes
            uniq_idxs = set([node.idx for node in traversed])
            return list(uniq_idxs) if idxs \
                    else [self.nodes[idx] for idx in uniq_idxs]
        else:
            return [node.idx for node in traversed] if idxs \
                    else traversed

    def get_ancestors(self, node, deduplicate=True, limit=None):
        """Return a list of the node's ancestors.
        """
        return self.get_traversal(node,
                ascending=True,
                deduplicate=deduplicate,
                idxs=False,
                limit=limit)

    def get_ancestor_idxs(self, node_idx, deduplicate=True, limit=None):
        """Return a list of the node's ancestors.
        """
        return self.get_traversal(self.nodes[node_idx],
                ascending=True,
                deduplicate=deduplicate,
                idxs=True,
                limit=limit)

    def get_descendants(self, node, deduplicate=True, limit=None):
        """Return a list of the node's descendants.
        """
        return self.get_traversal(node,
                ascending=False,
                deduplicate=deduplicate,
                idxs=False,
                limit=limit)

    def get_descendant_idxs(self, node_idx, deduplicate=True, limit=None):
        """Return a list of the node's descendants.
        """
        return self.get_traversal(self.nodes[node_idx],
                ascending=False,
                deduplicate=deduplicate,
                idxs=True,
                limit=limit)


class DependencyTree(DependencyDag):
    """A dependency tree is a like a DAG with a single parent per node.
    """
    def __init__(self, num_tokens, **kwargs):
        """Initialize the list of nodes and edges along with root nodes.
        """
        DependencyDag.__init__(self, num_tokens, **kwargs)
        self.max_depth = 0

    def add_node(self, **kwargs):
        """Add an node.
        """
        return DependencyDag.add_node(self, depth=0, parent_idx=None, **kwargs)

    def add_edge(self, src, tgt, **kwargs):
        """Add an edge directly by identifying the source and target nodes
        or indices of nodes.
        """
        if isinstance(src, int):
            src = self.nodes[src]
        if isinstance(tgt, int):
            tgt = self.nodes[tgt]

        if len(tgt.incoming_edges) > 0:
            print "ERROR: tree must have at most one parent per node"
            raise Exception

        self.set_depth_recursively(tgt, src.depth + 1, set((src.idx, tgt.idx)))
        edge = DependencyDag.add_edge(self, src, tgt, **kwargs)
        tgt.parent_idx = src.idx
        return edge

    def set_depth_recursively(self, node, depth, seen):
        """Recursively update the depth of each node.
        """
        node.depth = depth
        if len(node.child_idxs) == 0:
            if depth > self.max_depth:
                self.max_depth = depth
        else:
            if node.child_idxs.intersection(seen):
                print "ERROR: found a cycle when assigning depths at",
                print node.idx, "with children", node.child_idxs,
                print "having seen nodes", seen
                raise Exception

            seen.update(node.child_idxs)
            for child in self.get_children(node):
                self.set_depth_recursively(child, depth + 1, seen)

    def get_root(self):
        """Get the root node.
        """
        return self.nodes[self.root_idxs[0]]

    def get_root_idxs(self):
        """Get the index of the root node.
        """
        return self.root_idxs[0]

    def get_parent(self, node):
        """Get the parent of a node.
        """
        if node.parent_idx is None:
            return None
        else:
            return self.nodes[node.parent_idx]

    def get_parent_idx(self, node_idx):
        """Get the index of a node's parents given the node's index.
        """
        return self.nodes[node_idx].parent_idx

    def get_ancestors(self, node, limit=None):
        """Return a list of the node's ancestors.
        """
        return self.get_traversal(node,
                ascending=True,
                deduplicate=False,
                idxs=False,
                limit=limit)

    def get_ancestor_idxs(self, node_idx, limit=None):
        """Return a list of indices of the node's ancestors.
        """
        return self.get_traversal(self.nodes[node_idx],
                ascending=True,
                deduplicate=False,
                idxs=True,
                limit=limit)

    def get_descendants(self, node, limit=None):
        """Return a list of the node's descendants.
        """
        return self.get_traversal(node,
                ascending=False,
                deduplicate=False,
                idxs=False,
                limit=limit)

    def get_descendant_idxs(self, node_idx, limit=None):
        """Return a list of indices of the node's descendants.
        """
        return self.get_traversal(self.nodes[node_idx],
                ascending=False,
                deduplicate=False,
                idxs=True,
                limit=limit)

    def get_elder_sibling(self, node):
        """Return the nearest elder sibling of the node in a projective tree.
        This is the closest sibling between the node and its parent or the
        parent itself if there is no elder sibling.
        """
        return self.nodes[self.get_elder_sibling_idx(node.idx)]

    def get_elder_sibling_idx(self, node_idx):
        """Return the index of the nearest elder sibling in a projective tree.
        """
        if self.is_root(node_idx):
            # Assume that multiple roots don't share a sibling relationship.
            return None

        parent_idx = self.get_parent_idx(node_idx)
        sibling_idxs = list(self.get_sibling_idxs(node_idx)) + [parent_idx]
        if node_idx > parent_idx:
            return max(sibling_idx for sibling_idx in sibling_idxs
                    if parent_idx <= sibling_idx < node_idx)
        elif node_idx < parent_idx:
            return min(sibling_idx for sibling_idx in sibling_idxs
                    if node_idx < sibling_idx <= parent_idx)
        else:
            print "ERROR: child and parent share node idx:", node_idx

    def get_path_of_descent(self, src, tgt):
        """Return a list of edges between two nodes who share an ancestral
        relationship. This is empty if the source is not an ancestor of the
        target.
        """
        src = self.to_node(src)
        tgt = self.to_node(tgt)

        if src.depth >= tgt.depth:
            return []

        current = tgt
        path = []
        while current is not None:
            if current.idx == src.idx:
                # Reverse the path so that edges are ordered by depth
                return path[::-1]
            elif len(current.incoming_edges) == 0:
                # Hit the top of the tree without finding the source
                return []
            else:
                # Add incoming edge to path
                parent_edge = current.incoming_edges.values()[0]
                path.append(parent_edge)

                # Look at the next parent
                current = self.get_parent(current)

        # Sanity check -- should never reach here
        assert False

    def is_well_formed(self):
        """Check whether this is a well-formed tree by testing whether every
        node reaches the root when traversing through parents.
        """
        node_idxs = set(range(len(self.nodes)))

        while len(node_idxs) > 0:
            random_idx = node_idxs.pop()
            if random_idx in self.root_idxs:
                continue

            ancestor_idxs = self.get_ancestor_idxs(random_idx)
            if ancestor_idxs[-1] not in self.root_idxs:
                return False
            node_idxs.difference_update(ancestor_idxs)

        return True

    def is_projective(self):
        """Check whether this tree is projective.
        """
        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1.src_idx == edge2.src_idx:
                    # Ignore identical arcs and arcs which share a head
                    continue
                sm1, lg1 = (edge1.src_idx, edge1.tgt_idx) \
                        if edge1.src_idx < edge1.tgt_idx \
                        else (edge1.tgt_idx, edge1.src_idx)
                sm2, lg2 = (edge2.src_idx, edge2.tgt_idx) \
                        if edge2.src_idx < edge2.tgt_idx \
                        else (edge2.tgt_idx, edge2.src_idx)
                if sm1 < sm2 < lg1 < lg2 or sm2 < sm1 < lg2 < lg1:
                    return False
        return True

    def get_crossing_edges(self):
        """Return the crossing edges in this tree.
        """
        crossing_edges = set()

        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1.src_idx == edge2.src_idx:
                    # Ignore identical arcs and arcs which share a head
                    continue
                # Avoid double-counting the same crossing edge pair
                if (edge2, edge1) in crossing_edges:
                    continue

                sm1, lg1 = (edge1.src_idx, edge1.tgt_idx) \
                        if edge1.src_idx < edge1.tgt_idx \
                        else (edge1.tgt_idx, edge1.src_idx)
                sm2, lg2 = (edge2.src_idx, edge2.tgt_idx) \
                        if edge2.src_idx < edge2.tgt_idx \
                        else (edge2.tgt_idx, edge2.src_idx)
                if sm1 < sm2 < lg1 < lg2 or sm2 < sm1 < lg2 < lg1:
                    crossing_edges.add((edge1, edge2))

        return crossing_edges
