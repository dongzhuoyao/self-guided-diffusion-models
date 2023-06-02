import h5py 



def get_patchfeat_by_index(dl, index):

    filename = dl.id2name(index)
    id_in_h5 = int(dl.filename2id[filename])
    
    feat_list = h5py.File(dl.feat_file, 'r')[dl.split_name] 
    return feat_list[id_in_h5]