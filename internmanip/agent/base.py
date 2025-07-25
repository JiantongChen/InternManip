from internmanip.configs import AgentCfg
from enum import Enum


class AgentRegistry(Enum):
    """
    Registry of agent subclasses.
    The key is the agent type.
    The value is the agent subclass.
    """
    GR00T_N1 = "Gr00t_N1_Agent"
    GR00T_N15 = "Gr00t_N1_Agent"
    GR00T_N15_GENMANIP = "Gr00tAgent_Genmanip"
    PI0 = "PI0Agent"
    DP = "DPAgent"
    
    @property
    def value(self):
        if self.name == "GR00T_N1":
            from internmanip.agent.gr00t.gr00t_agent import Gr00t_N1_Agent
            return Gr00t_N1_Agent
        elif self.name == "GR00T_N15":
            from internmanip.agent.gr00t.gr00t_agent import Gr00t_N1_Agent
            return Gr00t_N1_Agent
        elif self.name == "GR00T_N15_GENMANIP":
            from internmanip.agent.gr00t.gr00t_agent_genmanip import Gr00tAgent_Genmanip
            return Gr00tAgent_Genmanip
        elif self.name == "DP":
            from internmanip.agent.dp_agent_genmanip import DPAgent
            return DPAgent
        elif self.name == "PI0":
            # from internmanip.agent.pi0_agent import Pi0Agent
            # return Pi0Agent
            from internmanip.agent.gr00t.gr00t_agent import Gr00t_N1_Agent
            return Gr00t_N1_Agent
        else:
            raise ValueError(f"Invalid agent type: {self.name}. Only {[agent_type.name for agent_type in AgentRegistry]} are registered.")


class BaseAgent:
    
    def __init__(self, config: AgentCfg):
        self.config = config
        
        if self.config.server_cfg is not None:
            from internmanip.agent.utils import PolicyClient
            self.policy_model = PolicyClient(self.config)
        else:
            from internmanip.model.basemodel.base import BasePolicyModel
            self.policy_model = BasePolicyModel.init(
                model_type=self.config.agent_type,
                model_name_or_path=self.config.model_name_or_path, 
                **self.config.model_kwargs
            )

    def step(self):
        raise NotImplementedError("Not implemented in base agent class")

    def reset(self):
        raise NotImplementedError("Not implemented in base agent class")
    
    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        print(f"Initializing agent {config.agent_type}")
        return AgentRegistry[config.agent_type].value(config)
