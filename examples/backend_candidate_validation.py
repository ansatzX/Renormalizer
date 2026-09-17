"""Run independently scored fixed CPU controls, optionally the reviewed GPU control.

Use --sandbox-config for the pinned, locally validated CPU isolation provider.
The fixed reviewed CPU examples are ordinary subprocesses, not a sandbox.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import tempfile
import numpy as np
from tools.backend_validation.policy import ExecutionPolicy, review_source
from tools.backend_validation.protocol import freeze_manifest
from tools.backend_validation.runner import run_candidate
from tools.backend_validation.report import write_report


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='backend-validation-report.json');p.add_argument('--gpu',action='store_true');p.add_argument('--sandbox-config');args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='reno-validation-example-') as root:
        root=Path(root);inputs=root/'inputs';refs=root/'reference';inputs.mkdir();refs.mkdir()
        a=(np.arange(7*5,dtype='float64').reshape(7,5)%7-3)/7
        b=(np.arange(5*3,dtype='float64').reshape(5,3)%5-2)/5
        reference=np.array([[math.fsum(float(a[i,k])*float(b[k,j]) for k in range(5)) for j in range(3)] for i in range(7)])
        np.save(inputs/'a.npy',a,allow_pickle=False);np.save(inputs/'b.npy',b,allow_pickle=False);np.save(refs/'reference.npy',reference,allow_pickle=False)
        env=digest(json.dumps({'python':sys.version,'numpy':np.__version__},sort_keys=True).encode())
        source=digest(Path(__file__).read_bytes());results={}
        for name,code in [('correct','def candidate(a,b): return a@b\n'),('wrong','import numpy as np\ndef candidate(a,b): return np.zeros((a.shape[0],b.shape[1]),dtype=a.dtype)\n')]:
            candidate=root/(name+'.py');candidate.write_text(code)
            # These two literal implementations above are the reviewed sources.
            review=review_source(candidate)
            manifest=freeze_manifest(inputs,refs,candidate_hash=review.source_hash,source_hash=source,environment_hash=env)
            results[name]=run_candidate(manifest,candidate,ExecutionPolicy(review=review))
        if results['correct']['status']!='pass' or results['wrong']['status']!='fail' or results['wrong'].get('stage')!='scoring' or results['wrong'].get('reason')!='numerical_error':raise RuntimeError('control verification failed')
        candidate=root/'correct.py'
        isolated=freeze_manifest(inputs,refs,candidate_hash=digest(candidate.read_bytes()),source_hash=source,environment_hash=env,mode='untrusted')
        config=json.loads(Path(args.sandbox_config).read_text()) if args.sandbox_config else None
        results['untrusted']=run_candidate(isolated,candidate,ExecutionPolicy(provider_config=config))
        if config and results['untrusted']['status']!='pass':raise RuntimeError('configured isolated control failed: '+str(results['untrusted']))
        if args.gpu:
            from tools.backend_validation.gpu import run_reviewed_gpu
            results['gpu']=run_reviewed_gpu(a,b,reference,tolerances=manifest['tolerances'])
            if results['gpu']['status']!='pass':raise RuntimeError('GPU control failed')
        expected={'correct':'pass','wrong':'fail','untrusted':'pass'}
        if args.gpu: expected['gpu']='pass'
        report=dict(expected_statuses=expected,schema_version=1,status='pass' if config else 'blocked',claim='fixed-control correctness',device='cpu',
                    source_hash=source,environment_hash=env,fixture_hash=digest((inputs/'a.npy').read_bytes()+(inputs/'b.npy').read_bytes()),
                    reference_hash=digest((refs/'reference.npy').read_bytes()),scorer_hash=manifest['scorer_hash'],
                    tolerance_hash=manifest['tolerance_hash'],candidate_hash=digest(candidate.read_bytes()),
                    domain='7x5 times 5x3 float64 fixed controls',mode='reviewed CPU; optional isolated CPU and reviewed GPU',
                    resource_coverage={'reviewed_cpu':'not a security sandbox','memory':'no total machine guarantee'},
                    unsupported=[] if config else ['untrusted CPU needs validated provider configuration'],results=results)
        write_report(args.output,report)
        print(json.dumps({'correct':results['correct']['status'],'wrong':results['wrong']['status'],'untrusted':results['untrusted']['status'],'gpu':results.get('gpu',{}).get('status'),'report':args.output}))
if __name__=='__main__':main()
