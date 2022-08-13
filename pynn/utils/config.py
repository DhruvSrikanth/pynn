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
    
    def __getitem__(self, key: str):
        '''
        Get the value of the configuration parameter.
        Parameters:
            key: The configuration parameter.
        Returns:
            The value of the configuration parameter.
        '''
        return self.config[key]
    
    def __setitem__(self, key: str, value: str) -> None:
        '''
        Set the value of the configuration parameter.
        Parameters:
            key: The configuration parameter.
            value: The value of the configuration parameter.
        Returns:
            None
        '''
        self.config[key] = value
     
    def __repr__(self) -> None:
        """
        Print the configuration.
        Parameters:
            None
        Returns:
            None
        """
        cfg = f"\nGiven below is the configuration present in the config file:"
        i = 1
        for config_param, config_value in self.config.items():
            cfg += f"\t{i}. {config_param} = {config_value}"
            i += 1
        cfg += "\n"
        return cfg
    
    def __str__(self) -> None:
        """
        Print the configuration.
        Parameters:
            None
        Returns:
            None
        """
        cfg = f"\nGiven below is the configuration present in the config file:"
        i = 1
        for config_param, config_value in self.config.items():
            cfg += f"\t{i}. {config_param} = {config_value}"
            i += 1
        cfg += "\n"
        return cfg
        