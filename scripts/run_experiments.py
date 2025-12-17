# scripts/run_experiments.py

# minimal required cli command
# > python scripts/run_experiments.py --num-timesteps 2000 --generation-rate 0.167


import argparse

from evacsim.policies import (
    all_open_policy,
    cycle_policy,
    longest_current_wait_policy,
    longest_cumulative_wait_policy,
)
from evacsim.sim import (
    build_road_network,
    run_multiple_simulations
)
from evacsim.plotting import (
    plot_travel_times_by_policy,
    plot_network_with_generator_stats,
)

def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run evacuation traffic simulations under multiple signal policies."
    )

    # REQUIRED arguments
    parser.add_argument(
        "--num-timesteps",
        type=int,
        required=True,
        help="Number of simulation timesteps to run"
    )

    parser.add_argument(
        "--generation-rate",
        type=float,
        required=True,
        help="Per-timestep vehicle generation probability at generator nodes"
    )

    # OPTIONAL arguments
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=15,
        help="Number of nodes in the road network (default: 15)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for network generation (default: 42)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (0 = silent, higher = more output)"
    )

    return parser.parse_args()


def main(args):
    """
    Run the full experiment pipeline using parsed CLI arguments.
    """

    # Unpack arguments
    num_timesteps = args.num_timesteps
    default_generation_rate = args.generation_rate
    num_nodes = args.num_nodes
    seed = args.seed
    dpi = args.dpi
    verbose = args.verbose

    # Define policies
    policy_list = [
        all_open_policy,
        cycle_policy,
        longest_current_wait_policy,
        longest_cumulative_wait_policy,
    ]

    # Build road network
    G = build_road_network(
        num_nodes=num_nodes,
        seed=seed,
        verbose=verbose,
    )

    # Assign generation rates to generator nodes
    generation_rates = {
        node: default_generation_rate
        for node in G.nodes()
        if G.in_degree(node) == 0
    }

    # Run simulations
    grouped_travel_times, intersection_costs_history, percent_cars_finished = (
        run_multiple_simulations(
            G,
            num_timesteps=num_timesteps,
            all_policies_list=policy_list,
            generation_rates=generation_rates,
            verbose=verbose,
        )
    )

    # Plot results
    plot_travel_times_by_policy(
        grouped_travel_times,
        percent_cars_finished,
        num_timesteps,
        generation_rates,
        dpi=dpi,
    )

    plot_network_with_generator_stats(
        G,
        policy_list,
        grouped_travel_times,
        intersection_costs_history,
        generation_rates,
        percent_cars_finished,
        num_timesteps,
        dpi=dpi,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
