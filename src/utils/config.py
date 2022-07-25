from collections import OrderedDict

class Config(object):
    def __init__(self, config: dict) -> None:
        '''
        Initialize the configuration.
        Parameters:
            config: The configuration dictionary.
        Returns:
            None
        '''
        self.config = OrderedDict(config)
     
    def print_config(self) -> None:
        """
        Print the configuration.
        Parameters:
            None
        Returns:
            None
        """
        print(f"\nGiven below is the configuration present in the config file:")
        i = 1
        for config_param, config_value in self.config.items():
            print(f"\t{i}. Parameter : {config_param} | Value : {config_value}")
            i += 1
        print("\n")