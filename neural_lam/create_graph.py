# Standard library
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# Third-party
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx

# Local
from .config import load_config_and_datastores
from .datastore.base import BaseRegularGridDatastore
from .graphs import cluster_mesh, regular_mesh, saving
from .graphs import utils as gutils
from .graphs import vis


def create_graph(
    graph_dir_path: str,
    xy: np.ndarray,
    xy_boundary: np.ndarray,
    xy_atmosphere: np.ndarray,
    xy_land: np.ndarray,
    g2m_radius: float,
    g2m_radius_boundary: float,
    g2m_radius_atm: float,
    mesh_node_distance: float,
    mesh_refinement_factor: float,
    grid_to_first_mesh_refinement: float,
    n_max_levels: int | None = None,
    graph_type: str = "hierarchical",
    create_plot: bool = False,
    allow_disconnected: bool = False,
    m2g_k: int = 4,
    g2m_mean_degree: int = 0,
    connect_disconnected: bool = False,
    use_atmosphere_g2m: bool = True,
    max_edge_len: float = 20000,
):
    """
    Create graph components from `xy` grid coordinates and store in
    `graph_dir_path`.

    Creates the following files for all graphs:
    - g2m_edge_index.pt  [2, N_g2m_edges]
    - g2m_features.pt    [N_g2m_edges, d_features]
    - m2g_edge_index.pt  [2, N_m2m_edges]
    - m2g_features.pt    [N_m2m_edges, d_features]
    - m2m_edge_index.pt  list of [2, N_m2m_edges_level], length==n_levels
    - m2m_features.pt    list of [N_m2m_edges_level, d_features],
                         length==n_levels
    - mesh_features.pt   list of [N_mesh_nodes_level, d_mesh_static],
                         length==n_levels

    where
      d_features:
            number of features per edge (currently d_features==3, for
            edge-length, x and y)
      N_g2m_edges:
            number of edges in the graph from grid-to-mesh
      N_m2g_edges:
            number of edges in the graph from mesh-to-grid
      N_m2m_edges_level:
            number of edges in the graph from mesh-to-mesh at a given level
            (list index corresponds to the level)
      d_mesh_static:
            number of static features per mesh node (currently
            d_mesh_static==2, for x and y)
      N_mesh_nodes_level:
            number of nodes in the mesh at a given level

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_up_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1
    - mesh_down_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_down_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1

    where N_mesh_updown_edges_level is the number of edges in the graph from
    mesh-to-mesh between two consecutive levels (list index corresponds index
    of lower level)


    Parameters
    ----------
    graph_dir_path : str
        Path to store the graph components.
    xy : np.ndarray
        Grid (interior) coordinates, expected to be of shape (num_grid, 2).
    xy_boundary : np.ndarray
        Auxiliary grid coordinates, for boundary grid nodes to only use
        in encoding (g2m), expected to be of shape (num_boundary, 2).
    xy_atmosphere : np.ndarray
        Auxiliary grid coordinates, for atmospheric grid nodes to only use
        in encoding (g2m), expected to be of shape (num_atm, 2).
    g2m_radius : float
        Radius to connect interior and boundary nodes within for g2m.
    g2m_radius_atm : float
        Radius to connect atmospheric nodes within for g2m.
    mesh_node_distance: float,
        (For hierarchical/multiscale graphs) Distance between mesh nodes
        in meters at the bottom mesh level.
    mesh_refinement_factor: float,
        Factor between number of mesh nodes at each level in cluster graphs.
    grid_to_first_mesh_refinement: float,
        Factor between number of grid nodes and mesh nodes at bottom level.
    n_max_levels : int
        Limit multi-scale mesh to given number of levels, from bottom up
        (default: None (no limit)).
    graph_type : str
        What type of graph to generate multiscale/hierarchical/cluster
    create_plot : bool
        If graphs should be plotted during generation (default: False).
    allow_disconnected : bool
        Allow disconnected nodes in g2m. If False, will exit when disconnected
        grid or mesh nodes are found in g2m.
    xy_land: np.ndarray
        Grid coordinates of land points, for edge filtering for cluster graph.
        Expected to be of shape (num_grid, 2).
    m2g_k : int
        Number of nearest mesh neighbors to connect to each grid node in m2g.
    g2m_mean_degree : int
        Search for G2M radii (interior, boundary, atmosphere)
        that achieve mean connectivity N per grid node.
    connect_disconnected : bool
        If True, connect remaining disconnected nodes using nearest neighbor.
    use_atmosphere_g2m : bool
        If True, add atmospheric grid nodes and their g2m edges. If False,
        atmosphere is omitted from the g2m graph (e.g. when using atmosphere
        concat to interior instead).

    Returns
    -------
    None

    """
    os.makedirs(graph_dir_path, exist_ok=True)

    print(f"Writing graph components to {graph_dir_path}")

    #
    # Mesh
    #
    if create_plot:

        def mesh_plot_func(graph, title):
            vis.plot_graph(
                graph,
                title,
                graph_dir_path,
            )
            plt.show()

    else:
        mesh_plot_func = None

    # Build mesh graph over both interior and boundary nodes
    xy_for_mesh = np.concatenate((xy, xy_boundary), axis=0)

    if graph_type == "cluster":
        mesh_pos, G_bottom_mesh, save_graphs = (
            cluster_mesh.build_cluster_mesh_graph(
                xy_for_mesh,
                xy_land=xy_land,
                limit_mesh_levels=n_max_levels,
                grid_to_first_mesh_refinement=grid_to_first_mesh_refinement,
                mesh_refinement_factor=mesh_refinement_factor,
                mesh_plot_function=mesh_plot_func,
                max_edge_len=max_edge_len,
            )
        )
    else:
        mesh_pos, G_bottom_mesh, save_graphs = (
            regular_mesh.build_regular_mesh_graph(
                xy_for_mesh,
                mesh_node_distance=mesh_node_distance,
                mesh_refinement_factor=mesh_refinement_factor,
                limit_mesh_levels=n_max_levels,
                hierarchical=(graph_type == "hierarchical"),
                mesh_plot_function=mesh_plot_func,
            )
        )

    # Save all graphs
    for graph_name, graph in save_graphs.items():
        saving.save_edges_list(graph, graph_name, graph_dir_path)

    # Compute Voronoi cell areas for each mesh level and add to features
    mesh_features_with_area = []
    for level_pos in mesh_pos:
        pos_np = (
            level_pos.cpu().numpy()
            if isinstance(level_pos, torch.Tensor)
            else level_pos
        )
        # Compute Voronoi areas (2D planar)
        voronoi_areas = gutils.compute_voronoi_areas_2d(pos_np)
        # Concatenate position (x, y) with Voronoi area
        features = np.concatenate(
            [pos_np, voronoi_areas.reshape(-1, 1)], axis=1
        )
        mesh_features_with_area.append(
            torch.from_numpy(features.astype(np.float32))
        )

    # Save mesh features (positions + Voronoi areas)
    torch.save(
        mesh_features_with_area,
        os.path.join(graph_dir_path, "mesh_features.pt"),
    )  # mesh features: (x, y, voronoi_area) in float32

    # Plot Voronoi areas
    if create_plot:
        for level_i, g in enumerate(save_graphs["m2m"]):
            pos_t = mesh_pos[level_i]
            areas = mesh_features_with_area[level_i][:, 2].cpu().numpy()
            plot_g = Data(pos=pos_t, edge_index=g.edge_index)
            vis.plot_node_values(
                plot_g,
                f"Mesh Voronoi areas, level {level_i}",
                graph_dir_path,
                node_values=areas,
                value_name="Voronoi area (m²)",
            )
            plt.show()

    #
    # Grid2Mesh
    #

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_list = list(vm)
    vm_xy = np.stack([xy for _, xy in vm.data("pos")], axis=0)

    # build kd tree for mesh point pos
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # compute dm as mean edge length of bottom mesh graph
    edge_lengths = [
        np.linalg.norm(
            G_bottom_mesh.nodes[u]["pos"] - G_bottom_mesh.nodes[v]["pos"]
        )
        for u, v in G_bottom_mesh.edges
    ]
    dm = np.mean(edge_lengths)
    print(f"dm = {dm}")

    # Search for G2M radii achieving target mean connectivity
    if g2m_mean_degree is not None:
        g2m_radius, g2m_radius_boundary, g2m_radius_atm = (
            gutils.search_g2m_radii_by_mean_degree(
                xy,
                xy_boundary,
                xy_atmosphere if use_atmosphere_g2m else np.empty((0, 2)),
                vm_xy,
                kdt_m,
                dm,
                mean_degree=g2m_mean_degree,
                precision=0.01,
                check_atm=use_atmosphere_g2m,
            )
        )

    # grid nodes
    interior_nodes = [
        ((1, node_i), {"pos": pos}) for node_i, pos in enumerate(xy)
    ]
    G_interior = networkx.Graph()
    G_interior.add_nodes_from(interior_nodes)

    # build kd tree for grid point pos
    vg_list = list(G_interior.nodes)

    # now add (all bottom) mesh nodes, include features (pos)
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(G_bottom_mesh.nodes(data=True))

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m.add_nodes_from(sorted(G_interior.nodes(data=True)))

    # Add auxiliary encoding nodes
    boundary_nodes = [
        ((2, aux_node_i), {"pos": pos})
        for aux_node_i, pos in enumerate(xy_boundary)
    ]
    atmospheric_nodes = (
        [
            ((3, aux_node_i), {"pos": pos})
            for aux_node_i, pos in enumerate(xy_atmosphere)
        ]
        if use_atmosphere_g2m and xy_atmosphere.size > 0
        else []
    )
    G_g2m.add_nodes_from(boundary_nodes)
    if atmospheric_nodes:
        G_g2m.add_nodes_from(atmospheric_nodes)

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges from each grid node set to mesh
    edge_sources = [
        (G_interior.nodes(data=True), g2m_radius),
        (boundary_nodes, g2m_radius_boundary),
    ]
    if atmospheric_nodes:
        edge_sources.append((atmospheric_nodes, g2m_radius_atm))
    for node_list, connect_radius in edge_sources:
        # Note: Below could likely be vectorized, if networkx can play along
        for grid_node, node_attrs in node_list:
            # find neighbours (index in mesh graph)
            grid_node_pos = node_attrs["pos"]
            neigh_idxs = kdt_m.query_ball_point(
                grid_node_pos, dm * connect_radius
            )

            # Add edges for each mesh node within radius
            for mesh_node_i in neigh_idxs:
                mesh_node = vm_list[mesh_node_i]
                mesh_node_pos = vm_xy[mesh_node_i]
                # add edge from grid to mesh
                G_g2m.add_edge(grid_node, mesh_node)
                vdiff = mesh_node_pos - grid_node_pos
                d = np.linalg.norm(vdiff)
                G_g2m.edges[grid_node, mesh_node]["len"] = d
                G_g2m.edges[grid_node, mesh_node]["vdiff"] = vdiff

    node_list = list(G_g2m.nodes)

    # Build boolean masks per node category
    is_mesh = np.array(
        [not isinstance(n, tuple) for n in node_list]
    )  # mesh nodes are not tuples
    is_grid_interior = np.array(
        [isinstance(n, tuple) and n[0] == 1 for n in node_list]
    )
    is_grid_boundary = np.array(
        [isinstance(n, tuple) and n[0] == 2 for n in node_list]
    )
    is_grid_atm = np.array(
        [isinstance(n, tuple) and n[0] == 3 for n in node_list]
    )
    is_any_grid = is_grid_interior | is_grid_boundary | is_grid_atm

    pyg_g2m = from_networkx(G_g2m)

    # Check for disconnected nodes in g2m
    src = pyg_g2m.edge_index[0]
    dst = pyg_g2m.edge_index[1]
    num_nodes = pyg_g2m.num_nodes

    outdeg = degree(src, num_nodes=num_nodes)
    indeg = degree(dst, num_nodes=num_nodes)

    # Convert masks to torch
    grid_mask_t = torch.as_tensor(is_any_grid, device=outdeg.device)
    mesh_mask_t = torch.as_tensor(is_mesh, device=outdeg.device)

    # Find grid nodes with no outgoing edges and mesh nodes
    # with no incoming edges
    disc_grid = torch.where((outdeg == 0) & grid_mask_t)[0]
    disc_mesh = torch.where((indeg == 0) & mesh_mask_t)[0]

    if len(disc_grid) > 0 or len(disc_mesh) > 0:
        msg = "Disconnected nodes found in G2M\n"
        msg += f"{len(disc_grid)} disconnected grid nodes (outdeg==0)\n"
        msg += f"{len(disc_mesh)} disconnected mesh nodes (indeg==0)"
        print("Warning:", msg)

    pos = pyg_g2m.pos.cpu().numpy()
    disc_grid_np = disc_grid.cpu().numpy()
    disc_mesh_np = disc_mesh.cpu().numpy()

    # Print counts by subset among disconnected grid nodes
    if len(disc_grid_np) > 0:
        print("Disconnected interior:", np.sum(is_grid_interior[disc_grid_np]))
        print("Disconnected boundary:", np.sum(is_grid_boundary[disc_grid_np]))
        print("Disconnected atmosphere:", np.sum(is_grid_atm[disc_grid_np]))

    # Connect disconnected nodes if requested
    if connect_disconnected and (len(disc_grid) > 0 or len(disc_mesh) > 0):
        gutils.connect_disconnected_g2m(
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
        )
        # Re-check disconnected nodes after connecting
        src = pyg_g2m.edge_index[0]
        dst = pyg_g2m.edge_index[1]
        outdeg = degree(src, num_nodes=pyg_g2m.num_nodes)
        indeg = degree(dst, num_nodes=pyg_g2m.num_nodes)
        disc_grid = torch.where((outdeg == 0) & grid_mask_t)[0]
        disc_mesh = torch.where((indeg == 0) & mesh_mask_t)[0]
        disc_grid_np = disc_grid.cpu().numpy()
        disc_mesh_np = disc_mesh.cpu().numpy()

    # Plot disconnected nodes
    if create_plot:
        vis.plot_disconnected_nodes(
            pos,
            is_mesh,
            is_any_grid,
            disc_grid_np,
            disc_mesh_np,
            graph_dir_path,
        )

    # Check for disconnected nodes
    if len(disc_grid) > 0 or len(disc_mesh) > 0:
        msg = "Disconnected nodes found in G2M\n"
        msg += f"{len(disc_grid)} disconnected grid nodes (outdeg==0)\n"
        msg += f"{len(disc_mesh)} disconnected mesh nodes (indeg==0)"
        if allow_disconnected:
            print("Warning:", msg)
        else:
            raise ValueError(msg)

    if create_plot:
        vis.plot_graph(
            pyg_g2m, "Grid-to-mesh", graph_dir_path, order_by_degree=True
        )
        plt.show()
        pyg_g2m_r = pyg_g2m.clone()
        pyg_g2m_r.edge_index = pyg_g2m.edge_index[[1, 0]]
        vis.plot_graph(pyg_g2m_r, "Grid-to-mesh-r", graph_dir_path)
        plt.show()

    #
    # Mesh2Grid
    #

    # similar to Grid2Mesh, but only with grid nodes
    G_m2g = networkx.DiGraph()
    G_m2g.add_nodes_from(G_bottom_mesh.nodes(data=True))
    G_m2g.add_nodes_from(sorted(G_interior.nodes(data=True)))

    # add edges from mesh to grid
    # order in vm should be same as in vm_xy
    for v in vg_list:
        # find k nearest neighbours (index to vm_xy)
        if m2g_k == 1:
            _, neigh_idx = kdt_m.query(G_m2g.nodes[v]["pos"], k=1)
            neigh_idxs = (
                [neigh_idx]
                if not isinstance(neigh_idx, np.ndarray)
                else neigh_idx.flatten()
            )
        else:
            _, neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], k=m2g_k)
            if neigh_idxs.ndim > 1:
                neigh_idxs = neigh_idxs.flatten()
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            vdiff = G_m2g.nodes[v]["pos"] - G_m2g.nodes[u]["pos"]
            G_m2g.edges[u, v]["len"] = np.linalg.norm(vdiff)
            G_m2g.edges[u, v]["vdiff"] = vdiff

    pyg_m2g = from_networkx(G_m2g)

    # Remove m2g edges over land (edges_only: no node filter, no reindex)
    gutils.filter_edges_land(
        pyg_m2g, xy, xy_land, max_edge_len=max_edge_len, edges_only=True
    )

    # Check for disconnected nodes in m2g
    m2g_node_list = list(G_m2g.nodes)
    m2g_is_mesh = np.array([not isinstance(n, tuple) for n in m2g_node_list])
    m2g_is_grid = np.array(
        [isinstance(n, tuple) and n[0] == 1 for n in m2g_node_list]
    )

    m2g_dst = pyg_m2g.edge_index[1]
    m2g_num_nodes = pyg_m2g.num_nodes

    m2g_indeg = degree(m2g_dst, num_nodes=m2g_num_nodes)
    m2g_grid_mask_t = torch.as_tensor(m2g_is_grid, device=m2g_indeg.device)

    # Find grid nodes with no incoming edges from mesh
    m2g_disc_grid = torch.where((m2g_indeg == 0) & m2g_grid_mask_t)[0]

    if len(m2g_disc_grid) > 0:
        msg = "Disconnected nodes found in M2G\n"
        msg += f"{len(m2g_disc_grid)} disconnected grid nodes (indeg==0)"
        print("Warning:", msg)

    # Connect disconnected nodes if requested
    if connect_disconnected and len(m2g_disc_grid) > 0:
        gutils.connect_disconnected_m2g(
            pyg_m2g,
            m2g_is_mesh,
            m2g_is_grid,
            xy,
            vm_list,
            vm_xy,
            kdt_m,
            xy_land,
        )
        # Re-check disconnected nodes after connecting
        m2g_dst = pyg_m2g.edge_index[1]
        m2g_indeg = degree(m2g_dst, num_nodes=pyg_m2g.num_nodes)
        m2g_disc_grid = torch.where((m2g_indeg == 0) & m2g_grid_mask_t)[0]

    # Plot disconnected nodes
    if create_plot:
        m2g_pos = pyg_m2g.pos.cpu().numpy()
        m2g_disc_grid_np = m2g_disc_grid.cpu().numpy()

        vis.plot_disconnected_nodes(
            m2g_pos,
            m2g_is_mesh,
            m2g_is_grid,
            m2g_disc_grid_np,
            np.array([], dtype=int),  # No disconnected mesh nodes in m2g
            graph_dir_path,
            title="m2g_disconnected",
        )

    # Check for disconnected nodes
    if len(m2g_disc_grid) > 0:
        msg = "Disconnected nodes found in M2G\n"
        msg += f"{len(m2g_disc_grid)} disconnected grid nodes (indeg==0)"
        if allow_disconnected:
            print("Warning:", msg)
        else:
            raise ValueError(msg)

    if create_plot:
        vis.plot_graph(
            pyg_m2g, "Mesh-to-grid", graph_dir_path, reindex_edges=False
        )
        plt.show()
        pyg_m2g_r = pyg_m2g.clone()
        pyg_m2g_r.edge_index = pyg_m2g.edge_index[[1, 0]]
        vis.plot_graph(
            pyg_m2g_r,
            "Mesh-to-grid-r",
            graph_dir_path,
            reindex_edges=False,
            order_by_degree=True,
        )
        plt.show()

    # Save g2m and m2g everything
    # g2m
    saving.save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    saving.save_edges(pyg_m2g, "m2g", graph_dir_path)

    gutils.print_graph_stats(save_graphs, pyg_g2m, pyg_m2g)


def create_graph_from_datastore(
    datastore: BaseRegularGridDatastore,
    datastore_boundary: BaseRegularGridDatastore | None,
    datastore_atmosphere: BaseRegularGridDatastore | None,
    use_atmosphere_g2m: bool,
    **kwargs,
):
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy_interior = datastore.get_projected_xy("state", stacked=True)

    if datastore_boundary is not None:
        boundary_mask = datastore_boundary.get_mask(
            surface=True, stacked=True, invert=False
        )
        xy_boundary = datastore_boundary.get_projected_xy(
            "forcing", stacked=True
        )[boundary_mask]
    else:
        xy_boundary = np.empty((0, 2), dtype=xy_interior.dtype)

    if use_atmosphere_g2m:
        atmosphere_mask = datastore_atmosphere.get_atmosphere_mask(
            stacked=True, invert=False
        )
        xy_atmosphere = datastore_atmosphere.get_projected_xy(
            "forcing", stacked=True
        )[atmosphere_mask]
    else:
        xy_atmosphere = np.empty((0, 2), dtype=xy_interior.dtype)

    # Node ordering is: 1) interior, 2) boundary, 3) atmosphere (if specified)
    create_graph(
        xy=xy_interior[interior_mask],
        xy_land=xy_interior[~interior_mask],  # Coordinates of land points
        xy_boundary=xy_boundary,
        xy_atmosphere=xy_atmosphere,
        use_atmosphere_g2m=use_atmosphere_g2m,
        **kwargs,
    )


def cli(input_args=None):
    parser = ArgumentParser(
        description="Graph generation arguments",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to neural-lam configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multiscale",
        help="Name to save graph as",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up",
    )
    parser.add_argument(
        "--g2m_radius",
        type=float,
        default=0.67,
        help="Radius within which to connect grid nodes (interior and boundary)"
        "to mesh, a multiple of mean edge length in mesh",
    )
    parser.add_argument(
        "--g2m_radius_boundary",
        type=float,
        default=0.67,
        help="Radius within which to connect grid nodes (boundary)"
        "to mesh, a multiple of mean edge length in mesh",
    )
    parser.add_argument(
        "--g2m_radius_atm",
        type=float,
        default=0.67,
        help="Radius within which to connect grid nodes (atmospheric)"
        "to mesh, a multiple of mean edge length in mesh",
    )
    parser.add_argument(
        "--mesh_node_distance",
        type=float,
        default=5000.0,
        help="(For hierarchical/multiscale graphs) "
        "Distance between mesh nodes in meters at the bottom mesh level.",
    )
    parser.add_argument(
        "--mesh_refinement_factor",
        type=float,
        default=9,
        help="Factor between number of mesh nodes at each cluster mesh level.",
    )
    parser.add_argument(
        "--grid_to_first_mesh_refinement",
        type=float,
        default=9,
        help="Factor between number of grid nodes and mesh nodes at bottom "
        "level.",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Which type of graph structure to generate",
        choices=["multiscale", "hierarchical", "cluster"],
    )
    parser.add_argument(
        "--allow_disconnected",
        action="store_true",
        help="Allow disconnected nodes in g2m. This is generally a bad idea and"
        "should only be used for testing purposes.",
    )
    parser.add_argument(
        "--m2g_k",
        type=int,
        default=3,
        help="Number of nearest mesh nodes connected to each grid node in m2g.",
    )
    parser.add_argument(
        "--g2m_mean_degree",
        type=int,
        default=None,
        help="Search for G2M radii (interior, boundary, atmosphere) "
        "that achieve mean connectivity N per grid node.",
    )
    parser.add_argument(
        "--connect_disconnected",
        action="store_true",
        help="Connect remaining disconnected nodes using nearest neighbor.",
    )
    parser.add_argument(
        "--max_edge_len",
        type=float,
        default=20000,
        help="Max edge length in meters for land filtering.",
    )
    parser.add_argument(
        "--use_atmosphere_g2m",
        action="store_true",
        help="Atmosphere as separate grid nodes in g2m encoding (experimental)",
    )
    args = parser.parse_args(input_args)

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    # Load neural-lam configuration and datastore to use
    _, datastore, datastore_boundary, datastore_atmosphere, *_ = (
        load_config_and_datastores(config_path=args.config_path)
    )

    create_graph_from_datastore(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        graph_dir_path=os.path.join(datastore.root_path, "graphs", args.name),
        g2m_radius=args.g2m_radius,
        g2m_radius_boundary=args.g2m_radius_boundary,
        g2m_radius_atm=args.g2m_radius_atm,
        mesh_node_distance=args.mesh_node_distance,
        mesh_refinement_factor=args.mesh_refinement_factor,
        grid_to_first_mesh_refinement=args.grid_to_first_mesh_refinement,
        n_max_levels=args.levels,
        graph_type=args.type,
        create_plot=args.plot,
        allow_disconnected=args.allow_disconnected,
        m2g_k=args.m2g_k,
        g2m_mean_degree=args.g2m_mean_degree,
        connect_disconnected=args.connect_disconnected,
        max_edge_len=args.max_edge_len,
        use_atmosphere_g2m=args.use_atmosphere_g2m,
    )


if __name__ == "__main__":
    cli()
