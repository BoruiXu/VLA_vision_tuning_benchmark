"""DELTA-style post-projector matching with frozen stock features.

Taylor channel sensitivity is an approximation, not the original exhaustive
channel-ablation attention. The teacher is never registered on the student.
"""
import copy
import torch


def install(ft, model, coefficient, temperature=1.0):
    teacher = torch.nn.Sequential(copy.deepcopy(model.vision_backbone), copy.deepcopy(model.projector))
    teacher.requires_grad_(False)
    captured = {}
    original = ft.run_forward_pass
    calls = 0

    def capture(_module, _inputs, output):
        if 'student' in captured:
            raise RuntimeError('Multiple projector calls in one OFT training forward')
        captured['student'] = output

    # Register before PEFT; the projector module remains the same object.
    model.projector.register_forward_hook(capture)

    def forward(*args, **kwargs):
        nonlocal calls
        assert not args and kwargs['use_l1_regression'] and not kwargs['use_diffusion'] and not kwargs['use_film']
        captured.clear()
        device = torch.device('cuda', kwargs['device_id']) if isinstance(kwargs['device_id'], int) else torch.device(kwargs['device_id'])
        before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        loss, metrics = original(**kwargs)
        after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        student = captured.pop('student')
        if not torch.is_grad_enabled():
            return loss, metrics
        teacher.to(device=device)
        teacher.train(kwargs['vla'].training)
        pixels = kwargs['batch']['pixel_values'].to(device=device, dtype=torch.bfloat16)
        try:
            torch.set_rng_state(before_cpu); torch.cuda.set_rng_state(before_cuda, device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                target = teacher(pixels).detach()
        finally:
            torch.set_rng_state(after_cpu); torch.cuda.set_rng_state(after_cuda, device)
        assert student.shape == target.shape and student.requires_grad and not target.requires_grad
        gradient = torch.autograd.grad(loss, student, retain_graph=True)[0].detach()
        importance = (student.detach().float() * gradient.float()).abs().mean(dim=(0, 1))
        scaled = importance / importance.mean().clamp_min(1e-12)
        attention = torch.softmax(scaled / temperature, dim=0) * scaled.numel()
        penalty = (attention.detach() * (student.float() - target.float()).square().mean(dim=(0, 1))).mean()
        total = loss + coefficient * penalty
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite DELTA-style loss')
        calls += 1
        if calls <= 2 or calls % 50 == 0:
            print(f'[delta-oft] call={calls} task={loss.item():.7g} feature_raw={penalty.item():.7g} weighted={coefficient*penalty.item():.7g}', flush=True)
        metrics['loss_value'] = total.item()
        metrics['delta_feature'] = penalty.item()
        return total, metrics

    ft.run_forward_pass = forward
