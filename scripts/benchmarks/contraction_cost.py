"""Measure prebuilt contraction compute versus wrapper cost by bond dimension.

Example: python contraction_cost.py --current . --out contraction --bonds 1,2,4,8,16,32,64,128
results.json reports environment-update and H-v medians, their signed difference,
and a backend-lookup share estimate. The difference is an estimate, not an
isolated measurement of dispatch; negative differences expose timing noise.
"""
from _common import dispatch, measure, provenance, worker_parser, write_json


def main():
    ap = worker_parser(__doc__)
    ap.set_defaults(number=1000)
    ap.add_argument('--bonds', default='1,2,4,8,16,32,64,128')
    opt = ap.parse_args()
    if opt.baseline:
        ap.error('this probe requires current multibackend APIs; omit --baseline')
    if not dispatch(opt, __file__):
        return
    metadata = provenance(opt.source)
    import numpy as np
    import opt_einsum as oe
    from renormalizer.backend.context import current_backend
    from renormalizer.mps.oe_contract_wrap import oe_contract, oe_contract_expression
    rng = np.random.default_rng(0)
    module = current_backend().opt_einsum_module
    lookup = measure(current_backend, opt.number, opt.repeat, opt.warmup)
    rows = []
    for bond in map(int, opt.bonds.split(',')):
        if bond < 1:
            raise ValueError('bond dimensions must be positive')
        left = rng.standard_normal((bond, 5, bond)) + 0j
        right = rng.standard_normal((bond, 5, bond)) + 0j
        state = rng.standard_normal((bond, 2, bond)) + 1j * rng.standard_normal((bond, 2, bond))
        operator = rng.standard_normal((5, 2, 2, 5)) + 0j
        sub = 'abc,ade,bdfg,cfh->egh'
        operands = (left, state.conj(), operator, state)
        path = oe.contract_path(sub, *operands, optimize='optimal')[0]
        pre = oe.contract_expression(sub, *[x.shape for x in operands], optimize=path)
        hsub = 'abc,bdef,gfh,ceh->adg'
        hop = oe_contract_expression(hsub, left, operator, right, state.shape, constants=[0, 1, 2])
        hpath = oe.contract_path(hsub, left, operator, right, state, optimize='optimal')[0]
        hpre = oe.contract_expression(hsub, left, operator, right, state.shape, constants=[0, 1, 2], optimize=hpath)
        for label, compute, wrapped, lookups in [
            ('environment', lambda: pre(*operands, backend=module), lambda: oe_contract(sub, *operands), 4),
            ('hv', lambda: hpre(state, backend=module), lambda: hop(state), 1)]:
            a = measure(compute, opt.number, opt.repeat, opt.warmup)
            b = measure(wrapped, opt.number, opt.repeat, opt.warmup)
            rows.append(dict(bond=bond, operation=label, compute=a, wrapped=b,
                             difference_seconds=b['median_seconds'] - a['median_seconds'],
                             estimated_lookup_share=lookups * lookup['median_seconds'] / b['median_seconds']))
    write_json(opt.out / 'results.json', dict(metadata=metadata, lookup=lookup, measurements=rows))


if __name__ == '__main__':
    main()
