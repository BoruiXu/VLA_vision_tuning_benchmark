"""Batch-dispatch compatibility without changing OFT's batch-one math.

Explicitly sequential inside predict_batch: this is NOT a vectorized speedup.
Used identically for all checkpoint-curve points, including stock.
"""
from oft_dynamic import DynamicStatsOFTServer


class CurveOFTServer(DynamicStatsOFTServer):
    def predict_batch(self, obs_batch, ctx_batch):
        if len(obs_batch) != len(ctx_batch):
            raise ValueError('Batch/context length mismatch')
        return [self.predict(obs, ctx) for obs, ctx in zip(obs_batch, ctx_batch)]


if __name__ == '__main__':
    from vla_eval.model_servers.serve import run_server
    run_server(CurveOFTServer)
