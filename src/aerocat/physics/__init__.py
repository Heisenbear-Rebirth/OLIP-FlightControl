"""
AeroCat v18.0 Physics Engine

物理引擎模块导出。
"""

from .dynamics import (
    physics_dynamics_step,
    physics_substep,
    compute_thrust_and_torque,
    compute_quadratic_drag,
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    rotate_vector,
    quaternion_integrate,
)

from .battery import (
    battery_ecm_step,
    compute_ocv,
    compute_motor_power_current,
)

from .sensors import (
    imu_noise_model,
    imu_simple_update,
    apply_sensor_delay,
)

from .turbulence import (
    dryden_turbulence_step,
    gust_model,
    wind_shear_model,
    wind_shear_model,
    compute_wind_field,
    create_default_gust_params,
    randomize_gust_params,
)

from .motors import (
    motor_first_order_lag,
    motor_with_deadband,
    motor_throttle_to_thrust,
    motor_thrust_to_torque,
    motor_dynamics_complete,
    estimate_motor_bandwidth,
)

__all__ = [
    # Dynamics
    'physics_dynamics_step',
    'physics_substep',
    'compute_thrust_and_torque',
    'compute_quadratic_drag',
    'quaternion_multiply',
    'quaternion_conjugate',
    'quaternion_normalize',
    'rotate_vector',
    'quaternion_integrate',
    
    # Battery
    'battery_ecm_step',
    'compute_ocv',
    'compute_motor_power_current',
    
    # Sensors
    'imu_noise_model',
    'imu_simple_update',
    'apply_sensor_delay',
    
    # Turbulence
    'dryden_turbulence_step',
    'gust_model',
    'wind_shear_model',
    'wind_shear_model',
    'compute_wind_field',
    'create_default_gust_params',
    'randomize_gust_params',
    
    # Motors
    'motor_first_order_lag',
    'motor_with_deadband',
    'motor_throttle_to_thrust',
    'motor_thrust_to_torque',
    'motor_dynamics_complete',
    'estimate_motor_bandwidth',
]
