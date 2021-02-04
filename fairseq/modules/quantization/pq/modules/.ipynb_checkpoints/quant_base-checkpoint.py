import torch
import torch.nn as nn

from collections import OrderedDict

class PQBase:
    def __init__(self, buc_x, buc_y, n_centroids, block_size, x_dim):
        self.buc_x = buc_x
        self.buc_y = buc_y
        self.bucket_num = buc_x * buc_y
        self.n_centroids = n_centroids
        self.block_size = block_size
        self.x_dim = x_dim
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        assignment_buckets = state_dict[prefix + "assignment_buckets"]
        state_dict[prefix + "assignments"] = decode_buc(assignment_buckets, self.x_dim, self.n_centroids)
        del state_dict[prefix + "assignment_buckets"]

        centroids = []
        for ix in range(self.bucket_num):
            lyr = "%s%s.%d" %  (prefix, "centroids", ix)
            loc_cent = state_dict[lyr]
            centroids.append(loc_cent)
            del state_dict[lyr]
        state_dict[prefix + "centroids"] = torch.cat(centroids, dim=0)
        return state_dict

    def _save_to_state_dict(self, destination, prefix, keep_vars, implementation):
        for name, param in implementation._parameters.items():
            if param is not None:
                if "centroids" in name:
                    new_param = param.reshape(-1, self.n_centroids, self.block_size)
                    for ix in range(self.bucket_num):
                        loc_param = new_param[ix]
                        new_name = name.replace("centroids", "centroids.%d" % ix)
                        destination[prefix + new_name] = loc_param if keep_vars else loc_param.data
                else:
                    destination[prefix + name] = param if keep_vars else param.data
        for name, buf in implementation._buffers.items():
            if buf is not None:
                if "assignments" in name:
                    buf = encode_buc(buf, self.buc_x, self.buc_y, self.n_centroids, self.x_dim)
                    name = name.replace("assignments", "assignment_buckets")
                destination[prefix + name] = buf if keep_vars else buf.data
        
class PQEmbeddingBase(nn.Embedding):

    def __init__(self, buc_x, buc_y, n_centroids, block_size, *args, **kwargs):
        self.pq = PQBase(buc_x, buc_y, n_centroids, block_size, args[0]) # take num embeddings
        super().__init__(*args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        new_state_dict = self.pq._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.pq._save_to_state_dict(destination, prefix, keep_vars, self)


class PQLinearBase(nn.Module):

    def __init__(self, buc_x, buc_y, n_centroids, block_size, out_features, *args, **kwargs):
        self.pq = PQBase(buc_x, buc_y, n_centroids, block_size, out_features)
        super().__init__(*args, **kwargs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        new_state_dict = self.pq._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.pq._save_to_state_dict(destination, prefix, keep_vars, self)


def decode_buc(assignment_buckets, x_dim, n_centroids):
    buc_x = assignment_buckets.shape[0]
    buc_y = assignment_buckets.shape[1]
    weight = []
    bucket_num = 0
    assignments = []
    for ix in range(buc_x):
        for jx in range(buc_y):
            idx = ix * buc_y + jx
            loc_assign = assignment_buckets[ix, jx].type(torch.LongTensor)
            new_assign = loc_assign + (idx * n_centroids)
            assignments.append(new_assign.reshape(1,-1))
    assignments = torch.cat(assignments, dim=0)
    y_div = (assignments.shape[1] * buc_x // x_dim)
    assignments = assignments.reshape(buc_x, buc_y, -1).flatten(1, 2).reshape(buc_x, buc_y, y_div, -1).flatten(1, 2).permute(1, 0, 2).flatten()
    return assignments

def encode_buc(assignments, buc_x, buc_y, n_centroids, x_dim):
    y_div = ((assignments.shape[0]//(buc_x*buc_y))*buc_x)//x_dim
    assignment_buckets = torch.fmod(assignments
                                    .reshape(buc_y*y_div, buc_x, -1)
                                    .permute(1,0,2).reshape(buc_x,buc_y,y_div,-1)
                                    .reshape(buc_x,-1)
                                    .reshape(buc_x,buc_y,-1),n_centroids)
    return assignment_buckets.byte().data.clone()