"""OpenPRISM: provenance-rich multisensor fusion for machines and operators."""

from .contracts import (
    Detection,
    FusionOutput,
    PrismFrame,
    SensorObservation,
    SynchronizationStatus,
    Timestamp,
)
from .fusion import EvidenceFusionEngine, FusionConfig
from .autonomy import (
    AdaptiveFusionController,
    FusionControlRecommendation,
    FusionPolicyModel,
)
from .atlas import (
    AtlasMissionConfig,
    AtlasMissionMapper,
    GeoObjectObservation,
    ImageObjectObservation,
    RegisteredThermalFrameContract,
    SemanticFrameContract,
)
from .mapping import CameraIntrinsics, GeodeticCoordinate, LocalENUFrame
from .pixhawk import CameraPoseRecord, PixhawkBridge, PixhawkBridgeConfig
from .survey import (
    RoughOverlapContract,
    SurveyDatumContract,
    SurveyPreparationConfig,
    SurveyPreparationResult,
    prepare_odm_survey_project,
)

__all__ = [
    "Detection",
    "AdaptiveFusionController",
    "AtlasMissionConfig",
    "AtlasMissionMapper",
    "CameraIntrinsics",
    "CameraPoseRecord",
    "EvidenceFusionEngine",
    "FusionConfig",
    "FusionControlRecommendation",
    "FusionOutput",
    "FusionPolicyModel",
    "GeoObjectObservation",
    "GeodeticCoordinate",
    "ImageObjectObservation",
    "LocalENUFrame",
    "PixhawkBridge",
    "PixhawkBridgeConfig",
    "PrismFrame",
    "RegisteredThermalFrameContract",
    "RoughOverlapContract",
    "SemanticFrameContract",
    "SensorObservation",
    "SurveyDatumContract",
    "SurveyPreparationConfig",
    "SurveyPreparationResult",
    "SynchronizationStatus",
    "Timestamp",
    "prepare_odm_survey_project",
]

__version__ = "0.4.0.dev0"
