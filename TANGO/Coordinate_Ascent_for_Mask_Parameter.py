# Update each parameter using coordinate ascent
for name, param in zip(['edge_mask', 'node_feat_mask'],
                       [(edge_mask_loc, edge_mask_scale_positive),
                        (node_feat_mask_loc, node_feat_mask_scale_positive)]):

    # Sample other masks while keeping current mask fixed
    if name == 'edge_mask':
        node_feat_mask = Normal(node_feat_mask_loc, node_feat_mask_scale_positive).rsample()
        x_masked = x * node_feat_mask
    elif name == 'node_feat_mask':
        edge_mask = Normal(edge_mask_loc, edge_mask_scale_positive).rsample()
        edge_index_masked = edge_index[:, edge_mask > 0.5]
        x_masked = x * node_feat_mask_loc

    # Model execution to compute likelihood
    out, node_output = self.model(x_masked, edge_index_masked, return_node_output=True)
    nll = -node_output[node].sum()

    # Coordinate update step
    loc, scale = param
    kl = Normal(loc, scale).log_prob(loc) - prior.log_prob(loc).sum()
    elbo = nll + kl
    elbo.backward()
    optimizer.step()
