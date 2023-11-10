import gym


class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        # make myosuite envs
        if env_name == "CustomMyoRelocateP1":
            return gym.make("CustomMyoChallengeRelocateP1-v0", **kwargs)
        elif env_name == "CustomMyoRelocateP2":
            return gym.make("CustomMyoChallengeRelocateP2-v0", **kwargs)
        else:
            raise ValueError("Environment name not recognized:", env_name)
