import logging
from spikingjelly.activation_based import base, layer


def set_step_mode(net, step_mode, keep_instance):
    keep_step_mode_instance = (
        layer.StepModeContainer, layer.ElementWiseRecurrentContainer, layer.LinearRecurrentContainer
    )
    keep_step_mode_instance += keep_instance
    # step_mode of sub-modules in keep_step_mode_instance will not be changed
    
    keep_step_mode_containers = []
    for m in net.modules():
        if isinstance(m, keep_step_mode_instance):
            keep_step_mode_containers.append(m)
    
    for m in net.modules():
        if hasattr(m, "step_mode"):
            is_contained = False
            for container in keep_step_mode_containers:
                if not isinstance(m, keep_step_mode_instance) and m in container.modules():
                    is_contained = True
                    break
            if is_contained:
                # this function should not change step_mode of submodules in keep_step_mode_containers
                pass
            else:
                if not isinstance(m, (base.StepModule)):
                    logging.warning(f"Trying to set the step mode for {m}, which is not spikingjelly.activation_based"
                                    f".base.StepModule")
                m.step_mode = step_mode


def set_backend(net, backend, instance, keep_instance):
    keep_backend_instance = keep_instance
    
    keep_backend_containers = []
    for m in net.modules():
        if isinstance(m, keep_backend_instance):
            keep_backend_containers.append(m)

    for m in net.modules():
        if isinstance(m, instance):
            if hasattr(m, 'backend'):
                is_contained = False
                for container in keep_backend_containers:
                    if m in container.modules():
                        is_contained = True
                        break
                if is_contained:
                    pass
                else:
                    if not isinstance(m, base.MemoryModule):
                        logging.warning(
                            f'Trying to set the backend for {m}, which is not spikingjelly.activation_based.base.MemoryModule')
                    if backend in m.supported_backends:
                        m.backend = backend
                    else:
                        logging.warning(f'{m} does not supports for backend={backend}. It will still use backend={m.backend}.')
    