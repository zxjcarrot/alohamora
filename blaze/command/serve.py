""" Implements the command for serving a trained policy """
import time
import os

from blaze.config.config import get_config
from blaze.config.serve import ServeConfig
from blaze.evaluator.analyzer import get_num_rewards
from blaze.logger import logger as log

from . import command


@command.argument("location", help="The path to the saved model")
@command.argument(
    "--model",
    help="The RL technique used during training for the saved model",
    required=True,
    choices=["A3C", "APEX", "PPO"],
)
@command.argument("--host", help="The host to bind the gRPC server to", default="0.0.0.0")
@command.argument("--port", help="The port to bind the gRPC server to", default=24450, type=int)
@command.argument("--max_workers", help="The maximum number of RPC workers", default=4, type=int)
@command.argument("--reward_func", help="Reward function to use", default=3, choices=list(range(get_num_rewards())))
@command.command
def serve(args):
    """
    Serves a trained model via a gRPC server. By default, the server binds to 0.0.0.0:24450.
    Check the protobuf specification to see the request and response formats.
    """
    log.info(
        "starting server...",
        model=args.model,
        location=args.location,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        reward_func=args.reward_func,
    )

    # check that the passed model location exists
    if not os.path.exists(args.location) or not os.path.isfile(args.location):
        raise IOError("The model location must be a valid file")

    # lazy load import statements
    from blaze.serve.server import Server
    from blaze.serve.policy_service import PolicyService

    if args.model == "A3C":
        from blaze.model import a3c as model
    if args.model == "APEX":
        from blaze.model import apex as model
    if args.model == "PPO":
        from blaze.model import ppo as model

    import ray

    ray.init()

    serve_config = ServeConfig(host=args.host, port=args.port, max_workers=args.max_workers)
    saved_model = model.get_model(args.location)
    server = Server(serve_config)
    server.set_policy_service(PolicyService(saved_model, config=get_config(reward_func=args.reward_func)))
    server.start()
    log.info("started server successfully")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        log.info("stopping server")
        server.stop()
        ray.shutdown()
