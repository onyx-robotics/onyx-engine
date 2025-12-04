# onyxengine/modeling/__init__.py
from typing import Annotated, Union
from pydantic import BaseModel, Field

from .validate_hyperparam import validate_param, validate_opt_param
from .model_features import (
    BaseFeature,
    Output,
    Input,
    Feature,
    FeatureScaler,
    # FeatureScalerJax,
)
from .model_base_config import (
    validate_inputs_and_outputs,
    OnyxModelBaseConfig,
    OnyxModelOptBaseConfig,
)
from .model_simulator import ModelSimulator
from .models import (
    MLP,
    MLPConfig,
    MLPOptConfig,
    # MLPJax,
    RNN,
    RNNConfig,
    RNNOptConfig,
    Transformer,
    TransformerConfig,
    TransformerOptConfig,
)

# Generic model config types and classes
ModelUnion = Union[MLPConfig, RNNConfig, TransformerConfig]
ModelOptUnion = Union[MLPOptConfig, RNNOptConfig, TransformerOptConfig]
OnyxModelConfig = Annotated[ModelUnion, Field(discriminator='type')]
OnyxModelOptConfig = Annotated[ModelOptUnion, Field(discriminator='type')]
class OnyxModelConfigClass(BaseModel):
    config: ModelUnion = Field(..., discriminator='type')
class OnyxModelOptConfigClass(BaseModel):
    config: ModelOptUnion = Field(..., discriminator='type')

from .model_from_config import model_from_config
from .optimizers import (
    AdamWConfig,
    AdamWOptConfig,
    SGDConfig,
    SGDOptConfig,
)
from .lr_schedulers import (
    CosineDecayWithWarmupConfig,
    CosineDecayWithWarmupOptConfig,
    CosineAnnealingWarmRestartsConfig,
    CosineAnnealingWarmRestartsOptConfig,
)
from .model_training import (
    TrainingConfig,
    OptimizationConfig,
    TrainingJob,
    OptimizationJob,
)
    
# Generic optimizer config types and classes
OptimizerUnion = Union[AdamWConfig, SGDConfig]
OptimizerOptUnion = Union[AdamWOptConfig, SGDOptConfig]
OptimizerConfig = Annotated[OptimizerUnion, Field(discriminator='type')]
OptimizerOptConfig = Annotated[OptimizerOptUnion, Field(discriminator='type')]
class OptimizerConfigClass(BaseModel):
    config: OptimizerUnion = Field(..., discriminator='type')
class OptimizerOptConfigClass(BaseModel):
    config: OptimizerOptUnion = Field(..., discriminator='type')
    
# Generic scheduler config types and classes
SchedulerUnion = Union[CosineDecayWithWarmupConfig, CosineAnnealingWarmRestartsConfig]
SchedulerOptUnion = Union[CosineDecayWithWarmupOptConfig, CosineAnnealingWarmRestartsOptConfig]
SchedulerConfig = Annotated[SchedulerUnion, Field(discriminator='type')]
SchedulerOptConfig = Annotated[SchedulerOptUnion, Field(discriminator='type')]
class SchedulerConfigClass(BaseModel):
    config: SchedulerUnion = Field(..., discriminator='type')
class SchedulerOptConfigClass(BaseModel):
    config: SchedulerOptUnion = Field(..., discriminator='type')
