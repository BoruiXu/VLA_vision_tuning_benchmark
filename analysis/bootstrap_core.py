import numpy as np
DIMENSIONS=("camera","robot","language","light","background","noise","layout")

def estimate(base, tuned, counts, weights, target):
    b=100*base.sum(axis=0)/counts.sum(axis=0)
    t=100*tuned.sum(axis=0)/counts.sum(axis=0)
    bn=weights@base;tn=weights@tuned;dn=weights@counts
    bb=100*bn/dn;tb=100*tn/dn
    off=[i for i,d in enumerate(DIMENSIONS) if d!=target]
    result={}
    def add(name, point_b, point_t, boot_b, boot_t):
        delta=boot_t-boot_b
        result[name]=dict(stock=float(point_b),tuned=float(point_t),delta_pp=float(point_t-point_b),
            stock_ci95=np.quantile(boot_b,[.025,.975]).tolist(),
            tuned_ci95=np.quantile(boot_t,[.025,.975]).tolist(),
            delta_ci95_pp=np.quantile(delta,[.025,.975]).tolist())
    def add_delta(name, point, boot):
        result[name]=dict(delta_pp=float(point),delta_ci95_pp=np.quantile(boot,[.025,.975]).tolist())
    for i,d in enumerate(DIMENSIONS): add(d,b[i],t[i],bb[:,i],tb[:,i])
    add('overall_micro',100*base.sum()/counts.sum(),100*tuned.sum()/counts.sum(),
        100*bn.sum(axis=1)/dn.sum(axis=1),100*tn.sum(axis=1)/dn.sum(axis=1))
    add('off_target_micro',100*base[:,off].sum()/counts[:,off].sum(),100*tuned[:,off].sum()/counts[:,off].sum(),
        100*bn[:,off].sum(axis=1)/dn[:,off].sum(axis=1),100*tn[:,off].sum(axis=1)/dn[:,off].sum(axis=1))
    add('off_target_macro',b[off].mean(),t[off].mean(),bb[:,off].mean(axis=1),tb[:,off].mean(axis=1))
    add('off_target_sum',b[off].sum(),t[off].sum(),bb[:,off].sum(axis=1),tb[:,off].sum(axis=1))
    add_delta('worst_off_target',(t-b)[off].min(),(tb-bb)[:,off].min(axis=1))
    # Catch accidental CI-bound summation or denominator drift.
    assert np.allclose(np.array(result['off_target_sum']['delta_ci95_pp'])/6,
                       result['off_target_macro']['delta_ci95_pp'])
    return result

