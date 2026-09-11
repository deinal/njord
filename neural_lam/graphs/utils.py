# Third-party
import networkx
import numpy as np
import scipy
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx


def node_lon_lat_to_cart(node_xy):
    """Convert node positions from longitude-latitude to
    Cartesian on unit sphere.

    Convention: x = longitude (column 0), y = latitude (column 1), matching
    datastore get_xy and plotting (x-axis = lon, y-axis = lat).

    Parameters
    ----------
    node_xy : np.ndarray
        (N_nodes, 2) array, columns [longitude, latitude] in degrees.

    Returns
    -------
    np.ndarray
        (N_nodes, 3) array, Cartesian coordinates on unit sphere.
    """
    lon_rad = np.deg2rad(node_xy[:, 0])
    lat_rad = np.deg2rad(node_xy[:, 1])
    # theta = 90 - lat (so z = cos(theta) = sin(lat)), phi = lon
    theta_grid = np.deg2rad(90.0) - lat_rad
    phi_grid = lon_rad

    cart = np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )
    return cart


def node_cart_to_lon_lat(node_cart: np.ndarray) -> np.ndarray:
    """Convert (N, 3) Cartesian on unit sphere to (N, 2) [longitude, latitude]
    in degrees. Inverse of node_lon_lat_to_cart (x=lon, y=lat convention)."""
    r = np.linalg.norm(node_cart, axis=1, keepdims=True)
    node_cart = node_cart / (r + 1e-12)
    lon_rad = np.arctan2(node_cart[:, 1], node_cart[:, 0])
    lat_rad = np.arcsin(np.clip(node_cart[:, 2], -1.0, 1.0))
    return np.stack([np.rad2deg(lon_rad), np.rad2deg(lat_rad)], axis=1).astype(
        np.float32
    )


def filter_global_edges_land(
    mesh_cart: np.ndarray,
    mesh_edge_index: np.ndarray,
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
    max_chord_len: float = 0.1,
    edges_only: bool = False,
) -> tuple:
    """Keep only mesh nodes over sea and/or edges whose midpoint is over sea.

    Uses 3D Cartesian. When edges_only=False (default): node kept if dist to
    nearest sea <= dist to nearest land; edge kept if both endpoints kept AND
    edge midpoint (normalized to unit sphere) is over sea (and chord length <=
    max_chord_len if set). When edges_only=True: no node filter, no reindexing;
    keep only edges whose midpoint is over sea (e.g. for m2g). Chord length
    is in [0, 2]; ~0.1 corresponds to ~5.7° great-circle arc.

    Parameters
    ----------
    mesh_cart : (N, 3) pos on unit sphere (mesh only or mesh|grid if edges_only)
    mesh_edge_index : (2, E) edge index [src, dst]
    sea_xy, land_xy : (n_sea, 2), (n_land, 2) [longitude, latitude] in degrees
    max_chord_len : if set, drop edges with chord length > this (unit sphere)
    edges_only : if True, only filter by edge midpoint, no node filter

    Returns
    -------
    mesh_cart_filtered : (N', 3) or unchanged pos if edges_only
    edge_index_filtered : (2, E') reindexed into N' if not edges_only
    """
    sea_cart = node_lon_lat_to_cart(sea_xy)
    land_cart = node_lon_lat_to_cart(land_xy)
    kdt_sea = scipy.spatial.KDTree(sea_cart)
    kdt_land = scipy.spatial.KDTree(land_cart)
    src, dst = mesh_edge_index[0], mesh_edge_index[1]
    chord = np.linalg.norm(mesh_cart[dst] - mesh_cart[src], axis=1)
    length_ok = (
        chord <= max_chord_len
        if max_chord_len is not None
        else np.ones(chord.shape[0], dtype=bool)
    )
    mid = (mesh_cart[src] + mesh_cart[dst]) / 2.0
    norm = np.linalg.norm(mid, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    mid_unit = mid / norm
    d_sea_mid, _ = kdt_sea.query(mid_unit, k=1)
    d_land_mid, _ = kdt_land.query(mid_unit, k=1)
    mid_over_sea = (d_sea_mid <= d_land_mid).ravel()
    keep_edge_base = length_ok & mid_over_sea

    if edges_only:
        return mesh_cart, mesh_edge_index[:, keep_edge_base]

    num_mesh = mesh_cart.shape[0]
    d_sea, _ = kdt_sea.query(mesh_cart, k=1)
    d_land, _ = kdt_land.query(mesh_cart, k=1)
    keep_node = (d_sea <= d_land).ravel()
    old_to_new = np.full(num_mesh, -1, dtype=np.int64)
    new_idx = 0
    for old_idx in range(num_mesh):
        if keep_node[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    mesh_cart_filtered = mesh_cart[keep_node]
    both_kept = keep_node[src] & keep_node[dst]
    keep_edge = both_kept & keep_edge_base
    new_src = old_to_new[src[keep_edge]]
    new_dst = old_to_new[dst[keep_edge]]
    edge_index_filtered = np.stack([new_src, new_dst], axis=0)
    return mesh_cart_filtered, edge_index_filtered


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def add_edge_features_pyg(graph):
    """
    Adds `len` and `vdiff` edge features to given pyg graph
    with a `pos` attribute.
    Modifies graph in-place.
    """
    graph["vdiff"] = (
        graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
    )
    graph["len"] = torch.norm(graph["vdiff"], dim=-1)


def add_edge_features_pyg_sphere(graph):
    """
    Adds `len` (chord length) and `vdiff` (3D Cartesian receiver - sender)
    from graph with `pos` (N, 2) [longitude, latitude] in degrees.
    Use for global/spherical meshes to match GraphCast-style edge features.
    Modifies graph in-place; leaves graph.pos unchanged (lon, lat).
    """
    pos_xy = (
        graph.pos.cpu().numpy()
        if graph.pos.is_cuda
        else graph.pos.detach().numpy()
    )
    pos_3d = node_lon_lat_to_cart(pos_xy)
    pos_3d = torch.from_numpy(pos_3d.astype(np.float32)).to(
        device=graph.pos.device, dtype=graph.pos.dtype
    )
    src, dst = graph.edge_index[0], graph.edge_index[1]
    vdiff = pos_3d[dst] - pos_3d[src]
    graph["vdiff"] = vdiff
    graph["len"] = torch.norm(vdiff, dim=-1)


def filter_edges_land(
    graph: pyg.data.Data,
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
    max_edge_len: float = 20000,  # in m
    edges_only: bool = False,
):
    """
    Filter edge set to only keep edges not crossing land.

    Uses projected xy: node kept if dist to nearest sea <= dist
    to nearest land; edge kept if both endpoints kept and edge midpoint is over
    sea (and edge length < max_edge_len). When edges_only=True (e.g. m2g): no
    node filter, no reindexing; keep only edges whose midpoint is over sea.

    `graph` is pyg Data with `edge_index` and `pos` (N, 2) in same coords as
    sea_xy, land_xy.
    """
    pos_np = (
        graph.pos.cpu().numpy()
        if graph.pos.is_cuda
        else graph.pos.detach().numpy()
    )
    send_pos = graph.pos[graph.edge_index[0]]
    rec_pos = graph.pos[graph.edge_index[1]]
    midpoint_pos = (send_pos + rec_pos) / 2
    edge_len = torch.norm(rec_pos - send_pos, dim=1)
    edge_len_filter = edge_len < max_edge_len

    kdt_sea = scipy.spatial.KDTree(sea_xy)
    kdt_land = scipy.spatial.KDTree(land_xy)
    midpoint_pos_np = midpoint_pos.numpy()
    d_sea_mid, _ = kdt_sea.query(midpoint_pos_np, k=1)
    d_land_mid, _ = kdt_land.query(midpoint_pos_np, k=1)
    midpoint_over_sea = (d_sea_mid <= d_land_mid).ravel()
    midpoint_filter = torch.tensor(midpoint_over_sea, dtype=bool)

    if edges_only:
        edge_filter = edge_len_filter & midpoint_filter
        graph.edge_index = graph.edge_index[:, edge_filter]
        return

    # Node filter: keep node iff nearest sea <= nearest land
    d_sea_node, _ = kdt_sea.query(pos_np, k=1)
    d_land_node, _ = kdt_land.query(pos_np, k=1)
    keep_node = (d_sea_node <= d_land_node).ravel()
    num_nodes = pos_np.shape[0]
    old_to_new = np.full(num_nodes, -1, dtype=np.int64)
    new_idx = 0
    for old_idx in range(num_nodes):
        if keep_node[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    pos_filtered = pos_np[keep_node]
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    both_kept = keep_node[src] & keep_node[dst]
    keep_edge = both_kept & midpoint_filter.numpy() & edge_len_filter.numpy()
    new_src = old_to_new[src[keep_edge]]
    new_dst = old_to_new[dst[keep_edge]]
    graph.pos = torch.from_numpy(pos_filtered.astype(np.float32))
    graph.edge_index = torch.from_numpy(
        np.stack([new_src, new_dst], axis=0).astype(np.int64)
    )


def _check_g2m_disconnected(
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    g2m_radius,
    g2m_radius_boundary,
    g2m_radius_atm,
    check_interior=True,
    check_boundary=True,
    check_atm=True,
):
    """
    Check for disconnected nodes in g2m without building full graph.
    Returns (num_disconnected_grid, num_disconnected_mesh).
    """
    num_grid_interior = len(xy) if check_interior else 0
    num_grid_boundary = len(xy_boundary) if check_boundary else 0
    num_grid_atm = len(xy_atmosphere) if check_atm else 0
    num_mesh = len(vm_xy)

    # Track which grid nodes have outgoing edges
    grid_has_outgoing = np.zeros(
        num_grid_interior + num_grid_boundary + num_grid_atm, dtype=bool
    )
    # Track which mesh nodes have incoming edges
    mesh_has_incoming = np.zeros(num_mesh, dtype=bool)

    # Check interior grid nodes
    if check_interior:
        for i, grid_pos in enumerate(xy):
            neigh_idxs = kdt_m.query_ball_point(grid_pos, dm * g2m_radius)
            if len(neigh_idxs) > 0:
                grid_has_outgoing[i] = True
                mesh_has_incoming[neigh_idxs] = True

    # Check boundary grid nodes
    if check_boundary:
        for i, grid_pos in enumerate(xy_boundary):
            idx = num_grid_interior + i
            neigh_idxs = kdt_m.query_ball_point(
                grid_pos, dm * g2m_radius_boundary
            )
            if len(neigh_idxs) > 0:
                grid_has_outgoing[idx] = True
                mesh_has_incoming[neigh_idxs] = True

    # Check atmospheric grid nodes
    if check_atm:
        for i, grid_pos in enumerate(xy_atmosphere):
            idx = num_grid_interior + num_grid_boundary + i
            neigh_idxs = kdt_m.query_ball_point(grid_pos, dm * g2m_radius_atm)
            if len(neigh_idxs) > 0:
                grid_has_outgoing[idx] = True
                mesh_has_incoming[neigh_idxs] = True

    # Only count disconnected grid nodes of the types we're checking
    if check_interior and check_boundary and check_atm:
        # Checking all types - count all disconnected grid nodes
        num_disc_grid = np.sum(~grid_has_outgoing)
    elif check_interior:
        # Only checking interior - only count interior disconnected nodes
        num_disc_grid = np.sum(~grid_has_outgoing[:num_grid_interior])
    elif check_boundary:
        # Only checking boundary - only count boundary disconnected nodes
        start_idx = num_grid_interior
        end_idx = num_grid_interior + num_grid_boundary
        num_disc_grid = np.sum(~grid_has_outgoing[start_idx:end_idx])
    else:  # check_atm
        # Only checking atmosphere - only count atmosphere disconnected nodes
        start_idx = num_grid_interior + num_grid_boundary
        num_disc_grid = np.sum(~grid_has_outgoing[start_idx:])

    # Don't count mesh nodes when searching for a specific radius
    # (they'll be connected by the combination of all radii)
    num_disc_mesh = 0

    return num_disc_grid, num_disc_mesh


def _g2m_mean_out_degree(xy_subset, kdt_m, dm, radius):
    """
    mean number of mesh nodes within dm*radius of each
    grid point in xy_subset.
    """
    if len(xy_subset) == 0:
        return 0.0
    counts = np.array(
        [
            len(kdt_m.query_ball_point(grid_pos, dm * radius))
            for grid_pos in xy_subset
        ],
        dtype=np.float64,
    )
    return float(np.mean(counts))


def _search_single_g2m_radius_by_mean_degree(
    radius_idx,
    radius_name,
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    mean_degree,
    precision=0.01,
    base_radii=None,
    low=0.01,
    high=5.0,
):
    """
    Binary search for smallest radius such that mean G2M out-degree
    (for this grid type) >= mean_degree.
    """
    if radius_idx == 0:
        xy_subset = xy
    elif radius_idx == 1:
        xy_subset = xy_boundary
    else:
        xy_subset = xy_atmosphere

    if len(xy_subset) == 0:
        return low

    current_radii = (
        [None, None, None] if base_radii is None else base_radii.copy()
    )
    best_radius = None

    # Ensure high gives at least mean_degree
    current_radii[radius_idx] = high
    med_deg = _g2m_mean_out_degree(
        xy_subset, kdt_m, dm, current_radii[radius_idx]
    )
    while med_deg < mean_degree and high < 20.0:
        high *= 2
        current_radii[radius_idx] = high
        med_deg = _g2m_mean_out_degree(
            xy_subset, kdt_m, dm, current_radii[radius_idx]
        )

    if med_deg < mean_degree:
        return high

    # Binary search for smallest radius with mean_degree >= target
    while high - low > precision:
        mid = (low + high) / 2
        current_radii[radius_idx] = mid
        med_deg = _g2m_mean_out_degree(xy_subset, kdt_m, dm, mid)
        print(
            f"{radius_name}: radius={mid:.2f} -> mean outdegree={med_deg:.2f}"
        )
        if med_deg >= mean_degree:
            best_radius = mid
            high = mid
        else:
            low = mid

    if best_radius is None:
        best_radius = high
    else:
        # Refine to 2-decimal precision
        test_start = round(best_radius - 0.01, 2)
        test_end = round(best_radius + 0.02, 2)
        for test_val in np.arange(test_start, test_end + 0.01, 0.01):
            test_val = round(test_val, 2)
            med_deg = _g2m_mean_out_degree(xy_subset, kdt_m, dm, test_val)
            if med_deg >= mean_degree:
                best_radius = test_val
                break
        else:
            best_radius = round(best_radius + 0.01, 2)

    return best_radius


def search_g2m_radii_by_mean_degree(
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    mean_degree,
    precision=0.01,
    check_atm=True,
):
    """
    Search for G2M radii (interior, boundary, atmosphere) that achieve
    the given mean out-degree per grid node for each type.
    Returns (g2m_radius, g2m_radius_boundary, g2m_radius_atm).
    When check_atm is False, g2m_radius_atm is returned as g2m_radius_boundary.
    """
    print(f"Searching for G2M radii with mean connectivity {mean_degree}...")

    g2m_radius = _search_single_g2m_radius_by_mean_degree(
        0,
        "g2m_radius",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        mean_degree,
        precision,
    )

    base_radii = [g2m_radius, None, None]
    g2m_radius_boundary = _search_single_g2m_radius_by_mean_degree(
        1,
        "g2m_radius_boundary",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        mean_degree,
        precision,
        base_radii,
    )

    if check_atm and len(xy_atmosphere) > 0:
        base_radii = [g2m_radius, g2m_radius_boundary, None]
        g2m_radius_atm = _search_single_g2m_radius_by_mean_degree(
            2,
            "g2m_radius_atm",
            xy,
            xy_boundary,
            xy_atmosphere,
            vm_xy,
            kdt_m,
            dm,
            mean_degree,
            precision,
            base_radii,
        )
        print(
            f"Found radii: interior={g2m_radius:.2f}, "
            f"boundary={g2m_radius_boundary:.2f}, "
            f"atmosphere={g2m_radius_atm:.2f}"
        )
    else:
        g2m_radius_atm = g2m_radius_boundary
        print(
            f"Found radii: interior={g2m_radius:.2f}, "
            f"boundary={g2m_radius_boundary:.2f}"
        )

    return g2m_radius, g2m_radius_boundary, g2m_radius_atm


def connect_disconnected_g2m(
    pyg_g2m,
    is_mesh,
    is_grid_interior,
    is_grid_boundary,
    is_grid_atm,
    vm_list,
    vm_xy,
    kdt_m,
    dm,
    g2m_radius,
    g2m_radius_boundary,
    g2m_radius_atm,
):
    """
    Connect disconnected nodes in g2m graph using nearest neighbor.
    Modifies graph in-place.
    """
    pos = pyg_g2m.pos.cpu().numpy()
    src = pyg_g2m.edge_index[0]
    dst = pyg_g2m.edge_index[1]
    num_nodes = pyg_g2m.num_nodes

    outdeg = pyg.utils.degree(src, num_nodes=num_nodes)
    indeg = pyg.utils.degree(dst, num_nodes=num_nodes)

    # Find disconnected nodes
    is_any_grid = is_grid_interior | is_grid_boundary | is_grid_atm
    grid_mask_t = torch.as_tensor(is_any_grid, device=outdeg.device)
    mesh_mask_t = torch.as_tensor(is_mesh, device=outdeg.device)

    disc_grid = torch.where((outdeg == 0) & grid_mask_t)[0].cpu().numpy()
    disc_mesh = torch.where((indeg == 0) & mesh_mask_t)[0].cpu().numpy()

    new_edges = []

    # Connect disconnected grid nodes to nearest mesh node
    for grid_idx in disc_grid:
        grid_pos = pos[grid_idx]
        # Find nearest mesh node (index in vm_xy/vm_list)
        # Mesh nodes come first in graph, so index matches vm_list index
        _, mesh_idx_vm = kdt_m.query(grid_pos, k=1)
        mesh_graph_idx = int(np.asarray(mesh_idx_vm).flat[0])
        new_edges.append([grid_idx, mesh_graph_idx])

    # Connect disconnected mesh nodes to nearest grid node
    if len(disc_mesh) > 0:
        grid_kdt = scipy.spatial.KDTree(pos[is_any_grid])
        grid_indices = np.where(is_any_grid)[0]
    for mesh_graph_idx in disc_mesh:
        mesh_pos = pos[mesh_graph_idx]
        _, grid_idx_in_subset = grid_kdt.query(mesh_pos, k=1)
        grid_idx = int(np.asarray(grid_idx_in_subset).flat[0])
        grid_idx = grid_indices[grid_idx]
        new_edges.append([grid_idx, mesh_graph_idx])

    if len(new_edges) > 0:
        new_edge_index = torch.tensor(
            new_edges, dtype=torch.long, device=pyg_g2m.edge_index.device
        ).T
        pyg_g2m.edge_index = torch.cat(
            [pyg_g2m.edge_index, new_edge_index], dim=1
        )
        # Update edge features
        add_edge_features_pyg(pyg_g2m)
        print(f"Connected {len(new_edges)} disconnected nodes in g2m")


def connect_disconnected_m2g(
    pyg_m2g,
    is_mesh,
    is_grid,
    xy,
    vm_list,
    vm_xy,
    kdt_m,
    xy_land,
):
    """
    Connect disconnected grid nodes in m2g graph using nearest neighbor.
    Modifies graph in-place.
    """
    pos = pyg_m2g.pos.cpu().numpy()
    dst = pyg_m2g.edge_index[1]
    num_nodes = pyg_m2g.num_nodes

    indeg = pyg.utils.degree(dst, num_nodes=num_nodes)
    grid_mask_t = torch.as_tensor(is_grid, device=indeg.device)

    disc_grid = torch.where((indeg == 0) & grid_mask_t)[0].cpu().numpy()

    new_edges = []

    # Connect disconnected grid nodes to nearest mesh node
    for grid_idx in disc_grid:
        grid_pos = pos[grid_idx]
        # Find nearest mesh node (index in vm_xy/vm_list)
        # Mesh nodes come first in graph, so index matches vm_list index
        dist, mesh_idx_vm = kdt_m.query(grid_pos, k=1)
        # Mesh nodes are at indices 0 to len(vm_list)-1 in the graph
        mesh_graph_idx = mesh_idx_vm
        new_edges.append([mesh_graph_idx, grid_idx])

    if len(new_edges) > 0:
        new_edge_index = torch.tensor(
            new_edges, dtype=torch.long, device=pyg_m2g.edge_index.device
        ).T
        pyg_m2g.edge_index = torch.cat(
            [pyg_m2g.edge_index, new_edge_index], dim=1
        )
        # Update edge features
        add_edge_features_pyg(pyg_m2g)
        print(f"Connected {len(new_edges)} disconnected nodes in m2g")


def compute_voronoi_areas_2d(xy: np.ndarray) -> np.ndarray:
    """Compute Voronoi cell areas for 2D planar mesh nodes.

    Parameters
    ----------
    xy : np.ndarray
        (N, 2) array of 2D coordinates (x, y).

    Returns
    -------
    np.ndarray
        (N,) array of Voronoi cell areas.
    """
    voronoi = scipy.spatial.Voronoi(xy)
    areas = np.zeros(xy.shape[0], dtype=np.float32)

    for point_idx, region_idx in enumerate(voronoi.point_region):
        region = voronoi.regions[region_idx]
        if -1 in region or len(region) == 0:
            areas[point_idx] = 0.0
        else:
            vertices = voronoi.vertices[region]
            if len(vertices) >= 3:
                # Use shoelace formula for polygon area
                area = 0.0
                for i in range(len(vertices)):
                    j = (i + 1) % len(vertices)
                    area += vertices[i][0] * vertices[j][1]
                    area -= vertices[j][0] * vertices[i][1]
                areas[point_idx] = abs(area) / 2.0
            else:
                areas[point_idx] = 0.0

    # Zero out extreme outliers (=coastal points)
    positive = areas > 0
    med = np.median(areas[positive])
    mad = np.median(np.abs(areas[positive] - med))
    if mad > 0:
        threshold = med + 10.0 * mad
        areas[areas > threshold] = 0.0

    return areas.astype(np.float32)


def compute_voronoi_areas_spherical(cart: np.ndarray) -> np.ndarray:
    """Compute Voronoi cell areas for 3D spherical mesh nodes on unit sphere.

    Parameters
    ----------
    cart : np.ndarray
        (N, 3) array of Cartesian coordinates on unit sphere.

    Returns
    -------
    np.ndarray
        (N,) array of Voronoi cell areas on the unit sphere.
    """
    # Normalize to unit sphere
    r = np.linalg.norm(cart, axis=1, keepdims=True)
    cart = cart / r

    sv = scipy.spatial.SphericalVoronoi(cart, radius=1.0)
    areas = sv.calculate_areas()

    # Zero out extreme outliers (=coastal points)
    positive = areas > 0
    med = np.median(areas[positive])
    mad = np.median(np.abs(areas[positive] - med))
    if mad > 0:
        threshold = med + 10.0 * mad
        areas[areas > threshold] = 0.0

    return areas.astype(np.float32)


def print_graph_stats(save_graphs, pyg_g2m, pyg_m2g):
    """Print node and edge counts for all graph components."""
    print("\n" + "=" * 50)
    print("Graph statistics")
    print("=" * 50)
    m2m_graphs = save_graphs["m2m"]
    for lev, g in enumerate(m2m_graphs):
        n_nodes = g.num_nodes
        n_edges = g.edge_index.shape[1]
        print(f"  m2m (level {lev}): {n_nodes} nodes, {n_edges} edges")
    total_mesh_nodes = sum(g.num_nodes for g in m2m_graphs)
    total_m2m_edges = sum(g.edge_index.shape[1] for g in m2m_graphs)
    print(f"  m2m (total): {total_mesh_nodes} nodes, {total_m2m_edges} edges")
    if "mesh_up" in save_graphs:
        for lev, g in enumerate(save_graphs["mesh_up"]):
            n_edges = g.edge_index.shape[1]
            print(f"  mesh_up (level {lev}->{lev + 1}): {n_edges} edges")
        for lev, g in enumerate(save_graphs["mesh_down"]):
            n_edges = g.edge_index.shape[1]
            print(f"  mesh_down (level {lev + 1}->{lev}): {n_edges} edges")
    n_g2m_nodes = pyg_g2m.num_nodes
    n_g2m_edges = pyg_g2m.edge_index.shape[1]
    print(f"  g2m: {n_g2m_nodes} nodes, {n_g2m_edges} edges")
    n_m2g_nodes = pyg_m2g.num_nodes
    n_m2g_edges = pyg_m2g.edge_index.shape[1]
    print(f"  m2g: {n_m2g_nodes} nodes, {n_m2g_edges} edges")
    print("=" * 50)
