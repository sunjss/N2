import torch
import torch.nn as nn


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, expand_x=True):
        super().__init__()
        dim_pe = 16  # Size of Laplace PE embedding
        n_layers = 2  # Num. layers in PE encoder model
        self.raw_norm = None  # max_freqs = 10 for d_model
        self.pass_as_var = False  # Pass PE also as a separate variable

        self.expand_x = expand_x

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)

        activation = nn.ReLU()  # register.act_dict[cfg.gnn.act]
        # DeepSet model for LapPE
        layers = []
        if n_layers == 1:
            layers.append(activation)
        else:
            self.linear_A = nn.Linear(2, 2 * dim_pe)
            layers.append(activation)
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                layers.append(activation)
            layers.append(nn.Linear(2 * dim_pe, dim_pe))
            layers.append(activation)
        self.pe_encoder = nn.Sequential(*layers)


    def forward(self, x, EigVals, EigVecs):
        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        return torch.cat((x, pos_enc), 1)


class RRWPLinearNodeEncoder(torch.nn.Module):
    """
        Adopted from GRIT
        FC_1(RRWP) + FC_2 (Node-attr)
        note: FC_2 is given by the Typedict encoder of node-attr in some cases
        Parameters:
        num_classes - the number of classes for the embedding mapping to learn
    """
    def __init__(self, emb_dim, out_dim, use_bias=False):
        super().__init__()
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, rrwp):
        # Encode just the first dimension if more exist
        # rrwp = data[f"{self.name}"]
        rrwp = self.fc(rrwp)

        x = x + rrwp
        return x