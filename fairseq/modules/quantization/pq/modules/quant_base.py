import torch
import torch.nn as nn

class PQBase:
    __slots__ = "centroids", "assignments", "buc_x", "buc_y", "n_centroids", "block_size", "x_dim"

    def __init__(self, centroid_buckets, assignment_buckets, bias=None):
        centroid_buckets = centroid_buckets.reshape(-1, self.block_size)
        self.centroids = nn.Parameter(centroid_buckets, requires_grad=True)
        assignments = decode_buc(assignment_buckets, self.x_dim, self.n_centroids)
        self.register_buffer("assignments", assignments)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    def weight(self):
        return (
            self.centroids[self.assignments]
                .reshape(-1, self.x_dim, self.block_size)
                .permute(1, 0, 2)
                .flatten(1, 2)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        assignment_buckets = state_dict[prefix + "assignment_buckets"]
        state_dict[prefix + "assignments"] = decode_buc(assignment_buckets, self.x_dim, self.n_centroids)
        del state_dict[prefix + "assignment_buckets"]

        centroids = []
        for ix in range(self.buc_x * self.buc_y):
            lyr = "%s%s.%d" %  (prefix, "centroids", ix)
            loc_cent = state_dict[lyr]
            centroids.append(loc_cent)
            del state_dict[lyr]
        state_dict[prefix + "centroids"] = torch.cat(centroids, dim=0)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for name, param in self._parameters.items():
            if param is None:
                continue
            if "centroids" in name:
                new_param = param.reshape(-1, self.n_centroids, self.block_size)
                for ix in range(self.bucket_num):
                    loc_param = new_param[ix]
                    new_name = name.replace("centroids", "centroids.%d" % ix)
                    destination[prefix + new_name] = loc_param if keep_vars else loc_param.data
            else:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is None:
                continue
            if "assignments" in name:
                buf = encode_buc(buf, self.buc_x, self.buc_y, self.n_centroids, self.x_dim)
                name = name.replace("assignments", "assignment_buckets")
            destination[prefix + name] = buf if keep_vars else buf.data

 # centroid assignment ne kadar layer typları ne kadar yer kaplıyor
def decode_buc(assignment_buckets, x_dim, n_centroids):
    buc_x = assignment_buckets.shape[0]
    buc_y = assignment_buckets.shape[1]
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
    y_div = ((assignments.shape[0] // (buc_x * buc_y)) * buc_x) // x_dim # TODO: why not => ((assignments.shape[0] // buc_y)) // x_dim
    assignment_buckets = torch.fmod(assignments
                                    .reshape(buc_y * y_div, buc_x, -1)
                                    .permute(1,0,2).reshape(buc_x, buc_y, y_div, -1)
                                    .reshape(buc_x, -1)
                                    .reshape(buc_x, buc_y, -1), n_centroids)
    return assignment_buckets.byte().data.clone()