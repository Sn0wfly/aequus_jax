# En: poker_bot/core/config.py

import yaml
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainerConfig:
    """Enhanced configuration for CFR training with hybrid CFR+ support"""
    # Core training parameters
    batch_size: int = 128
    num_actions: int = 9
    max_info_sets: int = 1_000_000
    learning_rate: float = 0.1
    
    # CFR parameters
    regret_floor: float = -100.0
    regret_ceiling: float = 100.0
    strategy_threshold: float = 1e-15
    
    # Hybrid CFR+ parameters
    discount_factor: float = 0.9995
    use_cfr_plus: bool = False
    use_regret_discounting: bool = True
    
    # MC-CFR parameters
    mc_sampling_rate: float = 0.50
    mc_exploration_epsilon: float = 0.6
    mc_min_samples_per_info_set: int = 100
    mc_max_samples_per_info_set: int = 10000
    
    # Training control
    save_interval: int = 1000
    log_interval: int = 100
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainerConfig':
        """Load configuration from YAML file with hybrid CFR+ support"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        default_config = cls()
        
        return cls(
            batch_size=config_dict.get('batch_size', default_config.batch_size),
            num_actions=config_dict.get('num_actions', default_config.num_actions),
            max_info_sets=config_dict.get('max_info_sets', default_config.max_info_sets),
            learning_rate=config_dict.get('learning_rate', default_config.learning_rate),
            regret_floor=config_dict.get('regret_floor', default_config.regret_floor),
            regret_ceiling=config_dict.get('regret_ceiling', default_config.regret_ceiling),
            strategy_threshold=config_dict.get('strategy_threshold', default_config.strategy_threshold),
            save_interval=config_dict.get('save_interval', default_config.save_interval),
            log_interval=config_dict.get('log_interval', default_config.log_interval),
            discount_factor=config_dict.get('discount_factor', default_config.discount_factor),
            use_cfr_plus=config_dict.get('use_cfr_plus', default_config.use_cfr_plus),
            use_regret_discounting=config_dict.get('use_regret_discounting', default_config.use_regret_discounting),
            mc_sampling_rate=config_dict.get('mc_sampling_rate', default_config.mc_sampling_rate),
            mc_exploration_epsilon=config_dict.get('mc_exploration_epsilon', default_config.mc_exploration_epsilon),
            mc_min_samples_per_info_set=config_dict.get('mc_min_samples_per_info_set', default_config.mc_min_samples_per_info_set),
            mc_max_samples_per_info_set=config_dict.get('mc_max_samples_per_info_set', default_config.mc_max_samples_per_info_set),
        ) 