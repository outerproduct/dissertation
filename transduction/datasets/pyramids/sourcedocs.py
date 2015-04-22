#!/usr/bin/env python
# Author: Kapil Thadani (kapil@cs.columbia.edu)

from __future__ import division, with_statement
from collections import defaultdict
import instance
from xml.etree import ElementTree


class SourceDocs(object):
    """A class to describe a set of source documents with optional SCU
    annotations.
    """
    def __init__(self, filepath, year):
        """Initialize the contents.
        """
        self.filepath = filepath
        self.year = year
        self.name = filepath[filepath.rfind('/') + 1:-4]
        self.docid = self.get_docid(self.name)

        self.docs = defaultdict(list)
        self.scus = {}
        self.parse_from_xml()

    def get_docid(self, name):
        """Get docid by dropping all characters after the initial D0?\d\d\d
        until a hyphen is reached or the string terminates.
        """
        name = name.upper()

        i = 1
        while i < len(name) and name[i].isdigit():
            i += 1

        if i == len(name) or name[i] == '-':
            return name  # eg: D0701, D0820-A

        j = i
        while j < len(name) and name[j] != '-':
            j += 1

        if j == len(name):
            return name[:i]  # eg: D0624f, d400b2
        else:
            return name[:i] + name[j:]  # eg: D0711C-A, D0740I-C

    def parse_from_xml(self):
        """Read in the document contents from an XML file.
        """
        etree = ElementTree.parse(self.filepath)
        collection = etree.getroot()
        self.name = collection.get('name')

        for doc in collection:
            doc_name = doc.get('name')
            for line in doc:
                self.docs[doc_name].append(line.text)

                for annotation in line:
                    for scu in annotation:
                        self.add_scu_mention(doc_name, scu)

    def add_scu_mention(self, doc_name, scu_etree):
        """Add an SCU mention for the most recently added line in the
        given document.
        """
        uid = scu_etree.get('uid')
        if uid in self.scus:
            scu = self.scus[uid]
            assert scu.is_consistent(**scu_etree.attrib)
        else:
            scu = SourceSCU(**scu_etree.attrib)

        scu.record_mention(doc_name,
                           len(self.docs[doc_name]) - 1,
                           self.docs[doc_name][-1]);
        self.scus[uid] = scu

    def get_fusions(self):
        """Return a list of SCUs that come from multiple lines.
        """
        fusions = []
        for scu in self.scus.itervalues():
            if scu.is_fusion_candidate():
                fusions.append(instance.FusionInstance(
                    input_sents=scu.get_mentions(),
                    output_sent=scu.label,
                    filepaths=(self.filepath,)))
        return fusions


class SourceSCU(object):
    """A class to describe an SCU within a source document collection.
    """
    def __init__(self, uid, label, weight):
        """Initialize the SCU.
        """
        self.uid = uid
        self.label = label
        self.weight = weight

        self.doc_mentions = defaultdict(list)

    def record_mention(self, name, line_num, line_text):
        """Record an annotated mention in a particular document.
        """
        self.doc_mentions[name].append((line_num, line_text))

    def get_mentions(self):
        """Return a list of the text mentions for this SCU.
        """
        return [mention[1] for mention_list in self.doc_mentions.itervalues()
                for mention in mention_list]

    def is_consistent(self, uid, label, weight):
        """Confirm that the given attributes match the stored attributes.
        """
        return self.uid == uid and \
                self.label == label and \
                self.weight == weight

    def is_fusion_candidate(self):
        """Return whether this SCU comes from multiple lines.
        """
        return len(self.doc_mentions) > 1
