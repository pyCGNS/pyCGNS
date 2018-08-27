#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation System -
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#
from __future__ import unicode_literals
import numpy as np
import h5py

def load(fname,nodata = False, maxdata = 64, subtree = '', follow_links = False):
    
    def _load(grp,links,paths,load_children = True):
        
        if grp.attrs['type'] == 'LK':
            if not follow_links:
                links.append(['',
                              grp[' file'].value[:-1].tostring().decode('ascii'),
                              grp[' path'].value[:-1].tostring().decode('ascii'),
                              grp.name,
                              64])
                return None
            else:
                grp = grp[' link']
            
        node = [unicode(grp.attrs['name']),
                None,
                [],
                unicode(grp.attrs['label'])
                ]
        
        if ' data' in grp:
            dset  = grp[' data']
            n = dset.size 
            if not nodata or n < maxdata:
                load_data = True
            else:
                load_data = False
                paths.append([grp.name,
                              1,
                              grp.attrs['type'],
                              dset.shape
                              ])
            
            if load_data:
                node[1] = np.copy(dset.value.transpose(),order = 'F')
                if grp.attrs['type'] == 'C1':
                    node[1] = np.vectorize(chr)(node[1]).astype(np.dtype('c'))
                
        if load_children:
            cnames = [x for x in grp.keys() if x[0] != ' ']
            for cname in cnames:
                child = _load(grp[cname],links,paths)
                if child is not None:
                    node[2].append(child)
            
        return node
        
    links = []
    paths = []
    with h5py.File(fname,'r') as h5f:
        if subtree == '':
            tree = _load(h5f,links,paths)
        else:
            grp = h5f[subtree]
            node = _load(grp,links,paths)
            while grp.name != '/':
                pnode = _load(grp.parent,links,paths,load_children = False)
                pnode[2].append(node)
                node = pnode
                grp = grp.parent

            tree = node
        
    tree[0] = u'CGNSTree'
    tree[3] = u'CGNSTree_t'

    return tree,links,paths

def save(fname,tree, links = []):
    
    def _cst_size_str(s,n):
            
        return np.string_(s.ljust(n,'\x00'))
        
    def _save(grp,node):
        
        _cst_size_str(node[0],33)
        
        if node[3] == 'CGNSTree_t':
            grp.attrs['name']  = _cst_size_str('HDF5 MotherNode', 33)
            grp.attrs['label'] = _cst_size_str('Root Node of HDF5 File', 33)
        else:
            grp.attrs['name']  = _cst_size_str(node[0], 33)
            grp.attrs['label'] = _cst_size_str(node[3], 33)
            grp.attrs['flags'] = np.array([0],dtype = np.int32)
            
        if node[1] is None:
            grp.attrs['type'] = _cst_size_str("MT", 3)
        else:
            if node[1].dtype == np.float32:
                grp.attrs['type'] = _cst_size_str("R4", 3)
            elif node[1].dtype == np.float64:
                grp.attrs['type'] = _cst_size_str("R8", 3)
            elif node[1].dtype == np.int32:
                grp.attrs['type'] = _cst_size_str("I4", 3)
            elif node[1].dtype == np.int64:
                grp.attrs['type'] = _cst_size_str("I8", 3)
            elif node[1].dtype == np.dtype('|S1'):
                grp.attrs['type'] = _cst_size_str("C1", 3)
            else:
                raise NotImplementedError("unknown dtype %s"%node[1].dtype)
        
            data = node[1]
            if node[1].dtype == np.dtype('|S1'):
                idx1 = np.nonzero(data != '')
                idx2 = np.nonzero(data == '')
                tmp = data.copy()
                data = np.zeros(data.shape,dtype = np.int8)
                data[idx1] = np.vectorize(ord)(tmp[idx1])
                data[idx2] = 0
                data = data.astype(np.int8)
            
            data = data.transpose()
            grp.create_dataset(' data',data = data)
            
        
        for child in node[2]:
            sgrp = grp.create_group(child[0])
            _save(sgrp,child)
            
    def _add_links(h5f,links):
        
        for dname,fname,rpth,lpth,n in links:
            tmp = lpth.split('/')
            name = tmp[-1]
            ppth = '/'.join(tmp[:-1])
            
            grp = h5f[ppth]
            sgrp = grp.create_group(name)
            
            sgrp.attrs['name']  = _cst_size_str(name, 33)
            sgrp.attrs['label'] = _cst_size_str('', 33)
            sgrp.attrs['type'] = _cst_size_str("LK", 3)
            
            data = np.array([ord(x) for x in rpth+'\x00'],dtype = np.int8)
            sgrp.create_dataset(' path',data = data)
            data = np.array([ord(x) for x in fname+'\x00'],dtype = np.int8)
            sgrp.create_dataset(' file',data = data)
            sgrp[' link'] = h5py.ExternalLink(fname, rpth)
            
            assert dname == ''
            
            
            
    with h5py.File(fname,'w') as h5f:
        h5f.create_dataset(' format', data = np.array([78,65,84,73,86,69,0],dtype = np.int8))
        h5f.create_dataset(' hdf5version', data = np.array([ 72,68,70,53,32,86,101,114,115,105,111,110,32,49,46,56,46,49
,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype = np.int8))
        _save(h5f,tree)
        _add_links(h5f,links)
        
#
# --- last line
