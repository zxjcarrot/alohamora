""" Implements the command for querying a trained policy """
import json

import random

import grpc
import collections
import json
import random
import urllib
import subprocess
import traceback
from typing import Callable, List, Optional, Set, Tuple

from blaze.action import Policy
from blaze.config.client import (
    get_client_environment_from_parameters,
    get_default_client_environment,
    ClientEnvironment,
)
from blaze.config.config import get_config, Config
from blaze.config.environment import EnvironmentConfig, ResourceType
from blaze.config.client import get_client_environment_from_parameters
from blaze.config.config import get_config
from blaze.config.environment import EnvironmentConfig
from blaze.evaluator.analyzer import get_num_rewards
from blaze.evaluator.simulator import Simulator
from blaze.logger import logger as log
from blaze.preprocess.record import get_page_load_time_in_replay_server
from blaze.serve.client import Client
from typing import Callable, List, Optional, Set, Tuple
from . import command
from blaze.action import Policy

@command.argument("--manifest", "-m", help="The location of the page manifest to query the model for", required=True)
@command.argument("--bandwidth", "-b", help="The bandwidth to query the model for (kbps)", type=int, required=True)
@command.argument("--latency", "-l", help="The latency to query the model for (ms)", type=int, required=True)
@command.argument(
    "--cpu_slowdown",
    "-s",
    help="The cpu slowdown of the device to query the model for",
    type=int,
    choices=[1, 2, 4],
    default=1,
)
@command.argument("--host", help="The host of the gRPC policy server to connect to", default="127.0.0.1")
@command.argument("--port", help="The port of the gRPC policy server to connect to", default=24450, type=int)
@command.command
def query(args):
    """
    Queries a trained model that is served on a gRPC server.
    """
    log.info("querying server...", host=args.host, port=args.port)

    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    client = Client(channel)

    manifest = EnvironmentConfig.load_file(args.manifest)
    client_env = get_client_environment_from_parameters(args.bandwidth, args.latency, args.cpu_slowdown)
    policy = client.get_policy(url=manifest.request_url, client_env=client_env, manifest=manifest)

    print(json.dumps(policy.as_dict, indent=4))


@command.argument("--location", help="The path to the saved model", required=True)
@command.argument(
    "--model",
    help="The RL technique used during training for the saved model",
    required=True,
    choices=["A3C", "APEX", "PPO"],
)
@command.argument("--manifest", "-m", help="The location of the page manifest to query the model for", required=True)
@command.argument("--bandwidth", "-b", help="The bandwidth to query the model for (kbps)", type=int, required=True)
@command.argument("--latency", "-l", help="The latency to query the model for (ms)", type=int, required=True)
@command.argument(
    "--cpu_slowdown",
    "-s",
    help="The cpu slowdown of the device to query the model for",
    type=int,
    choices=[1, 2, 4],
    default=1,
)
@command.argument(
    "--reward_func", help="Reward function to use", default=1, choices=list(range(get_num_rewards())), type=int
)
@command.argument("--use_aft", help="Use Speed Index metric", action="store_true")
@command.argument(
    "--cache_time", help="Simulate cached object expired after this time (in seconds)", type=int, default=None
)
@command.argument("--verbose", "-v", help="Output more information in the JSON output", action="store_true")
@command.argument(
    "--run_simulator", help="Run the outputted policy through the simulator (implies -v)", action="store_true"
)
@command.argument(
    "--run_replay_server", help="Run the outputted policy through the replay server (implies -v)", action="store_true"
)
@command.argument(
    "--extract_critical_requests", help="extract_critical_requests", action="store_true"
)
@command.command
def evaluate(args):
    """
    Instantiate the given model and checkpoint and query it for the policy corresponding to the given
    client and network conditions. Also allows running the generated policy through the simulator and
    replay server to get the PLTs and compare them under different conditions.
    """
    log.info("evaluating model...", model=args.model, location=args.location, manifest=args.manifest)
    client_env = get_client_environment_from_parameters(args.bandwidth, args.latency, args.cpu_slowdown)
    manifest = EnvironmentConfig.load_file(args.manifest)

    cached_urls = set(
        res.url
        for group in manifest.push_groups
        for res in group.resources
        if args.cache_time is not None and res.cache_time > args.cache_time
    )
    log.info("using cached resources", cached_urls=cached_urls)
    config = get_config(manifest, client_env, args.reward_func).with_mutations(
        cached_urls=cached_urls, use_aft=args.use_aft
    )

    if args.model == "A3C":
        from blaze.model import a3c as model
    if args.model == "APEX":
        from blaze.model import apex as model
    if args.model == "PPO":
        from blaze.model import ppo as model

    import ray

    ray.init(num_cpus=2, log_to_driver=False)

    saved_model = model.get_model(args.location)
    instance = saved_model.instantiate(config)
    import time
    tic = time.perf_counter()

    import time
    tic = time.perf_counter()
    policy = None
    for x in range(10):
        instance.clear()
        policy = instance.policy
    toc = time.perf_counter()
    print(f"All Policies generated in {toc - tic:0.4f} seconds")
    average_model_inference_time = (toc - tic) / 10
    print(f"Single policy generation overhead: {average_model_inference_time:0.4f} seconds")
    
    data = policy.as_dict

    print("Model Policy Loaded")
    if args.verbose or args.run_simulator or args.run_replay_server:
        data = {
            "manifest": args.manifest,
            "location": args.location,
            "client_env": client_env._asdict(),
            "policy": policy.as_dict,
        }

    average_model_inference_time *= 1000

    baseline_policy_generator = _push_preload_all_policy_generator()
    baseline_policy = baseline_policy_generator(manifest)
    push_all_critical_policy = _push_preload_all_critical_policy_generator()(manifest)
    print("Baseline Policy ", baseline_policy.as_dict)
    print("Push Critical Policy: ", push_all_critical_policy.as_dict)
    print("Model Policy: ", policy.as_dict)

    if args.run_simulator:
        sim = Simulator(manifest)
        sim_plt = sim.simulate_load_time(client_env, use_aft=args.use_aft)
        mode_push_plt = sim.simulate_load_time(client_env, policy, use_aft=args.use_aft)
        baseline_push_plt = sim.simulate_load_time(client_env, baseline_policy, use_aft=args.use_aft)
        data["simulator"] = {"without_policy": sim_plt, "with_model_policy": mode_push_plt, "with_baseline_policy": baseline_push_plt}

    if args.run_replay_server:
        *_, plts, crs = get_page_load_time_in_replay_server(config.env_config.request_url, client_env, config, extract_critical_requests=args.extract_critical_requests)
        print("without_policy, plts=", plts, ", crs=", crs)
        *_, model_push_plts, model_push_crs = get_page_load_time_in_replay_server(
            config.env_config.request_url, client_env, config, policy=policy, extract_critical_requests=args.extract_critical_requests
        )
        
        print("with_model_policy, plts=", model_push_plts, ", crs=", model_push_crs)

        *_, push_critical_plts, push_critical_crs = get_page_load_time_in_replay_server(
            config.env_config.request_url, client_env, config, policy=push_all_critical_policy, extract_critical_requests=args.extract_critical_requests
        )
        print("push_all_critical_policy, plts=", push_critical_plts, ", crs=", push_critical_crs)
        
        *_, baseline_push_plts, baseline_push_crs = get_page_load_time_in_replay_server(
            config.env_config.request_url, client_env, config, policy=baseline_policy, extract_critical_requests=args.extract_critical_requests
        )

        model_push_plts_with_inference_overhead = [x + average_model_inference_time for x in model_push_plts]
        print("with_model_policy_and_inference, plts=", model_push_plts_with_inference_overhead, ", crs=", model_push_crs)
        print("with_baseline_policy, plts=", baseline_push_plts, ", crs=", baseline_push_crs)


        data["replay_server"] = {
            "without_policy": [plts,crs],
            "with_model_policy_and_inference": [model_push_plts_with_inference_overhead, model_push_crs], 
            "with_model_policy": [model_push_plts,model_push_crs], 
            "with_push_critical_policy": [push_critical_plts,push_critical_crs], 
            "with_baseline_policy": [baseline_push_plts, baseline_push_crs]}
    data["average_model_inference_time"] = average_model_inference_time
    print(json.dumps(data, indent=4))

def _push_preload_all_critical_policy_generator() -> Callable[[EnvironmentConfig], Policy]:
    """
    Returns a generator than always choose to push/preload all assets
    Push all in same domain. Preload all in other domains.
    """

    def _generator(env_config: EnvironmentConfig) -> Policy:
        push_groups = env_config.push_groups
        # Collect all resources and group them by type
        all_resources = sorted([res for group in push_groups for res in group.resources], key=lambda res: res.order)
        # choose the weight factor between push and preload
        main_domain = urllib.parse.urlparse(env_config.request_url)
        policy = Policy()
        for r in all_resources:
            request_domain = urllib.parse.urlparse(r.url)
            print(r, main_domain.netloc, request_domain.netloc)
            push = request_domain.netloc == main_domain.netloc
            if r.critical == False and (r.source_id == 0 or r.order == 0):
                continue

            if r.critical == False:
                continue

            policy.steps_taken += 1
            if push:
                source = random.randint(0, r.source_id - 1) if r.source_id > 0 else r.source_id
                policy.add_default_push_action(push_groups[r.group_id].resources[source], r)
            else:
                source = random.randint(0, r.order - 1)
                policy.add_default_preload_action(all_resources[source], r)
        return policy

    return _generator


def _push_preload_all_policy_generator() -> Callable[[EnvironmentConfig], Policy]:
    """
    Returns a generator than always choose to push/preload all assets
    Push all in same domain. Preload all in other domains.
    """

    def _generator(env_config: EnvironmentConfig) -> Policy:
        push_groups = env_config.push_groups
        # Collect all resources and group them by type
        all_resources = sorted([res for group in push_groups for res in group.resources], key=lambda res: res.order)
        # choose the weight factor between push and preload
        main_domain = urllib.parse.urlparse(env_config.request_url)
        policy = Policy()
        for r in all_resources:
            request_domain = urllib.parse.urlparse(r.url)
            push = request_domain.netloc == main_domain.netloc
            if r.source_id == 0 or r.order == 0:
                continue
            
            policy.steps_taken += 1
            if push:
                source = random.randint(0, r.source_id - 1)
                policy.add_default_push_action(push_groups[r.group_id].resources[source], r)
            else:
                source = random.randint(0, r.order - 1)
                policy.add_default_preload_action(all_resources[source], r)
        return policy

    return _generator
