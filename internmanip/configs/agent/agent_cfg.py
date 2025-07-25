from pydantic import BaseModel, field_validator, field_serializer
from typing import Optional, Dict, Any
from transformers import PretrainedConfig, AutoConfig
from internmanip.configs.agent.server_cfg import ServerCfg


class AgentCfg(BaseModel):
    agent_type: str
    model_name_or_path: Optional[str] = None
    model_cfg: Optional[PretrainedConfig] = None
    model_kwargs: Optional[Dict[str, Any]] = {}
    server_cfg: Optional[ServerCfg] = None
    agent_settings: Optional[Dict[str, Any]] = {}

    class Config:
        arbitrary_types_allowed = True

    @field_validator("agent_type", mode="before")
    def validate_agent_type(cls, v):
        return v.upper() if isinstance(v, str) else v
    
    @field_validator("model_cfg", mode="before")
    def validate_model_cfg(cls, v):
        """处理model_cfg字段的反序列化，利用transformers的AutoConfig机制"""
        if v is None:
            return None
        
        if isinstance(v, PretrainedConfig):
            return v
        
        if isinstance(v, dict):
            model_type = v.get('model_type', '')
            
            # 尝试使用 transformers 的 CONFIG_MAPPING
            try:
                from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                if model_type in CONFIG_MAPPING:
                    config_class = CONFIG_MAPPING[model_type]
                    return config_class(**v)
            except Exception:
                pass
            
            # 回退到手动处理已知的配置类型
            if model_type == 'DP':
                from internmanip.configs.model.dp_cfg import DiffusionConfig
                return DiffusionConfig(**v)
            elif model_type == 'pi0':
                from internmanip.configs.model.pi0_cfg import PI0Config  
                return PI0Config(**v)
            elif model_type == 'radio':
                from internmanip.model.backbone.eagle2_hg_model_15.radio_model import RADIOConfig
                return RADIOConfig(**v)
            elif model_type == 'eagle2_chat':
                from internmanip.model.backbone.eagle2_hg_model.configuration_eagle_chat import Eagle2ChatConfig
                return Eagle2ChatConfig(**v)
            else:
                # 如果都没有匹配，尝试使用通用的PretrainedConfig
                return PretrainedConfig(**v)
        
        return v
    
    @field_serializer('model_cfg')
    def serialize_model_cfg(self, v: PretrainedConfig):
        return v.to_dict() if v is not None else None
