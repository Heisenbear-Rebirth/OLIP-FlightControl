"""
AeroCat v18.0 Networks Module
"""

from .actor_critic import (
    # Network classes
    ActorCritic,
    StochasticActorCritic,
    Encoder,
    Decoder,
    LSTMState,
    Transition,
    
    # Constants
    OBS_DIM,
    ACTION_DIM,
    LSTM_DIM,
    MAX_RP_RATE,
    MAX_YAW_RATE,
    PSC_NUM_BASIS,
    
    # Functions
    create_actor_critic,
    create_stochastic_actor_critic,
    map_action_to_physical,
    compute_v_physics,
    inference_step,
    compute_log_prob,
)

__all__ = [
    'ActorCritic',
    'StochasticActorCritic',
    'Encoder',
    'Decoder',
    'LSTMState',
    'Transition',
    'OBS_DIM',
    'ACTION_DIM',
    'LSTM_DIM',
    'MAX_RP_RATE',
    'MAX_YAW_RATE',
    'PSC_NUM_BASIS',
    'create_actor_critic',
    'create_stochastic_actor_critic',
    'map_action_to_physical',
    'compute_v_physics',
    'inference_step',
    'compute_log_prob',
]
