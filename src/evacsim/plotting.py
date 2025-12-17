import matplotlib.pyplot as plt
import networkx as nx
from collections import abc

# ======================================================================================================================
# method to plot policy comparison histograms
# ======================================================================================================================

def plot_travel_times_by_policy(grouped_travel_times, percent_cars_finished_by_policy, num_timesteps, generation_rates, dpi=100):
    # 1. Determine unique generator IDs and policy names
    policy_names = list(grouped_travel_times.keys())
    # Collect all unique generator IDs across all policies
    all_generator_ids = set()
    for policy_data in grouped_travel_times.values():
        all_generator_ids.update(policy_data.keys())
    generator_ids = sorted(list(all_generator_ids)) # Sort for consistent plotting order

    num_generators = len(generator_ids)
    num_policies = len(policy_names)

    fig, axes = plt.subplots(num_generators, num_policies, figsize=(num_policies * 5, num_generators * 4), dpi=dpi, squeeze=False)

    # Determine global min and max travel times for consistent axes
    global_min_time = float('inf')
    global_max_time = float('-inf')
    for policy_name, policy_data in grouped_travel_times.items():
        for generator_id, travel_times in policy_data.items():
            if travel_times:
                global_min_time = min(global_min_time, min(travel_times))
                global_max_time = max(global_max_time, max(travel_times))

    # Adjust bins to cover the entire global range
    global_bins = range(global_min_time, global_max_time + 3)

    for i, generator_id in enumerate(generator_ids):
        for j, policy_name in enumerate(policy_names):
            ax = axes[i, j]

            # Access travel times for the current policy and generator
            travel_times = grouped_travel_times.get(policy_name, {}).get(generator_id, [])

            if travel_times:
                ax.hist(travel_times, bins=global_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
                # plot a red vertical line at the mean
                mean_time = sum(travel_times) / len(travel_times)
                ax.axvline(mean_time, color='red', linewidth=0.5, label=f'Mean: {mean_time:.2f}')
                # plot the max time
                max_time = max(travel_times)
                ax.axvline(max_time, color='red', linestyle='dashed', linewidth=0.5, label=f'Max: {max_time:.2f}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No cars completed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')

            # Set titles and labels
            if i == 0:
                ax.set_title(f"{policy_name.replace('_', ' ').title()}: {percent_cars_finished_by_policy[policy_name]*100:.2f}% Completed", fontsize=12)
            if j == 0:
                ax.set_ylabel(f'Generator $g_{{{generator_id}}}$\nDensity', fontsize=12)
            if i == num_generators - 1:
                ax.set_xlabel('Total Travel Time', fontsize=12)

            ax.grid(axis='y', alpha=0.75)

    # Ensure all subplots have the same x-axis limits
    if global_min_time != float('inf') and global_max_time != float('-inf'):
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim(global_min_time - 0.5, global_max_time + 1.5) # Add a small buffer
                
    default_generation_rate = generation_rates
    if isinstance(default_generation_rate, abc.Mapping):
        default_generation_rate = max(default_generation_rate.values())
    fig.suptitle(f'Car Travel Time Density by Generator and Policy.\nTimesteps Per Simulation: {num_timesteps}\nGenerator Rate: {default_generation_rate:.3f}', y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(f'gen_hist_{default_generation_rate:.2f}.pdf', bbox_inches='tight')
    # plt.show()

# ======================================================================================================================
# method to plot network graph with overlaid cost values
# ======================================================================================================================

def plot_network_with_generator_stats(G, policy_list, grouped_travel_times, intersection_costs_history, generation_rates, percent_cars_finished_by_policy, num_timesteps, dpi=100, seed=42):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=dpi)
    axes = axes.flatten() # Flatten the 2x2 array of axes for easier iteration

    # Pre-calculate position for consistent layout across all subplots
    plot_pos = nx.spring_layout(G, seed=seed)

    for i, policy_func in enumerate(policy_list):
        policy_name = policy_func.__name__
        ax = axes[i]

        # Retrieve data for the current policy
        current_grouped_travel_times = grouped_travel_times[policy_name]
        current_intersection_costs_history = intersection_costs_history[policy_name]
        current_percent_cars_finished = percent_cars_finished_by_policy[policy_name]

        # Calculate generator and intersection costs based on max travel/wait times
        generator_costs = {}
        for gen_id, times in current_grouped_travel_times.items():
            generator_costs[gen_id] = max(times) if times else 0 # Use 0 if no cars completed

        intersection_costs = {}
        for int_id, costs_list in current_intersection_costs_history.items():
            intersection_costs[int_id] = max(costs_list) if costs_list else 0 # Use 0 if no costs recorded

        node_colors = []
        labels = {}
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if node_type == 'generator':
                node_colors.append('skyblue')
                labels[node] = f"\n\n$g_{{{node}}}$\n$\lambda_{{{node}}} = {generation_rates.get(node, 0.0):.2f}$\n $\\tau_{{{node}}} = {generator_costs.get(node, 0):.2f}$"
            elif node_type == 'n intersection':
                node_colors.append('salmon')
                labels[node] = f"$n_{{{node}}}$\n$J_{{{node}}} = {intersection_costs.get(node, 0):.2f}$"
            else: # Safe Region
                node_colors.append('lightgreen')
                labels[node] = f"$v_{{{node}}}$"

        nx.draw_networkx_nodes(G, plot_pos, node_color=node_colors, node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, plot_pos, edgelist=G.edges(), arrows=True, arrowsize=20, width=2, edge_color='gray', ax=ax)
        nx.draw_networkx_labels(G, plot_pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

        social_welfare = sum(generator_cost for generator_cost in generator_costs.values()) / len(generator_costs) if generator_costs else 0
        ax.set_title(f"Policy: {policy_name.replace('_', ' ').title()}\nSocial Welfare W={social_welfare:.2f}\nCompleted {current_percent_cars_finished*100:.2f} % of Cars", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f"Traffic Network Statistics by Policy\nTimesteps: {num_timesteps}", y=1.02, fontsize=16)
    plt.savefig(f'net_states_{max([_ for _ in generation_rates.values()]):.2f}.pdf', bbox_inches='tight')
    # plt.show()