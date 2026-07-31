from .agglomerate import deduplicate_spikes
from .clustering import (
    TMMRefinement,
    clustering_strategies,
    get_clusterer,
    refinement_strategies,
)
from .clustering_features import SimpleMatrixFeatures, StableWaveformFeatures
