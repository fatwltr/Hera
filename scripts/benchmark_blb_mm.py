#!/usr/bin/env python3
"""Build and measure BLB's real bounded ring conversion with its native SCI/IKNP.

Does not rebuild or alter native SCI archives. Correct general and bounded
kernels are measured against the same integer oracle, not a simulated carry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import shutil

from benchmark_monitor import monitored

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',type=Path,default=ROOT.parent/'blb/SCI')
    parser.add_argument('--output',type=Path,default=ROOT/'build/full/verification/blb_mm_2026-10-18')
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--reuse-build',action='store_true')
    parser.add_argument('--bridge',action='store_true',help='Measure original BLB bridge function overhead separately')
    parser.add_argument('--bridge-workers',type=int,choices=[1,6],default=6)
    args=parser.parse_args()
    native=args.native.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    build=native/'build-hera';source=native/'tests/bert_large_bolt'
    adapted=out/'source';adapted.mkdir(exist_ok=True)
    flags=['-std=c++17','-O3','-DNDEBUG','-fopenmp','-pthread','-maes','-msse4.1','-mavx2','-mrdseed',
           '-DUSE_VOLE=0','-DBLB_CONVERSION_WORKERS=6','-Wno-deprecated-declarations']
    for p in [adapted,source,native,native/'src',native/'deps/emp-tool',native/'deps/emp-ot',
              native/'extern/eigen',native/'extern/SEAL/native/src',build/'deps/SEAL/native/src',
              native/'extern/SEAL/thirdparty/msgsl-src/include']:
        flags+=['-I'+str(p)]
    commands=[]
    libraries=[build/'lib'/('libSCI-'+n+'.a') for n in ['LinearHE','FloatingPoint','Math','BuildingBlocks','LinearOT']]
    libraries+=[build/'deps/SEAL/lib/libseal-4.1.a']
    binary=out/'blb_mm_conversion_benchmark'
    if not args.reuse_build:
        for name,src in [('nonlinear',source/'nonlinear.cpp'),('driver',ROOT/'tests/blb_mm_conversion_benchmark.cpp')]:
            cmd=['c++',*flags,'-c',str(src),'-o',str(out/(name+'.o'))];commands.append(cmd)
            with (out/(name+'.build.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        cmd=['c++',*flags,str(out/'driver.o'),str(out/'nonlinear.o'),'-Wl,--start-group',*map(str,libraries),
             '-Wl,--end-group','-lcrypto','-lssl','-lgmp','-o',str(binary)]
        commands.append(cmd)
        with (out/'link.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        # Keep original BLB FFT, NTT, scale and active-prime behavior. Extract
        # only its two actual bridge methods, avoiding model/weight setup.
        original=(source/'ckks_bert_basic.cpp').read_text()
        start=original.index('vector<vector<HE_CKKS::scalar_t>> Bert_ckks::ckks_to_mpc(')
        end=original.index('// // evaluate input^2',start)
        body=original[start:end].replace('Bert_ckks::','NativeBridge::')
        (adapted/'bridge_adapter.h').write_text('''#pragma once
#include "he.h"
#include "nonlinear.h"
#include "mm_ckks_bridge.h"
struct NativeBridge {
  NonLinear &nonlinear;
  std::vector<std::vector<uint64_t>> ckks_to_mpc(std::vector<seal::Plaintext>,int,uint64_t,int,int,HE_CKKS*,bool=false);
  std::vector<seal::Plaintext> mpc_to_ckks(std::vector<std::vector<uint64_t>>,int,uint64_t,int,int,int,HE_CKKS*,bool=false);
};
''')
        (adapted/'bridge_adapter.cpp').write_text('#include "bridge_adapter.h"\n#undef MAX_THREADS\n#define MAX_THREADS 1\n'+body)
        shutil.copy2(source/'he.cpp',adapted/'he.cpp')
        he_header=(source/'he.h').read_text()
        # Galois/relinearization keys are offline and unused in these functions.
        he_header=he_header.replace('keygen.create_galois_keys(gal_keys_);','/* Unused offline Galois key omitted in both timing variants. */')
        he_header=he_header.replace('keygen.create_relin_keys(relin_keys_);','/* Unused offline relinearization key omitted in both timing variants. */')
        (adapted/'he.h').write_text(he_header)
        for name,src in [('he',adapted/'he.cpp'),('bridge_adapter',adapted/'bridge_adapter.cpp'),
                         ('mm_ckks_bridge',source/'mm_ckks_bridge.cpp'),
                         ('mm_wide_ring',source/'mm_wide_ring.cpp'),
                         ('mm_fixed_ckks_transform',source/'mm_fixed_ckks_transform.cpp'),
                         ('bridge_driver',ROOT/'tests/blb_mm_bridge_benchmark.cpp')]:
            cmd=['c++',*flags,'-c',str(src),'-o',str(out/(name+'.o'))];commands.append(cmd)
            with (out/(name+'.build.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        cmd=['c++',*flags,*[str(out/(name+'.o')) for name in ['bridge_driver','bridge_adapter','he','nonlinear',
                                                         'mm_ckks_bridge','mm_wide_ring','mm_fixed_ckks_transform']],
             '-Wl,--start-group',*map(str,libraries),'-Wl,--end-group','-lcrypto','-lssl','-lgmp',
             '-o',str(out/'blb_mm_bridge_benchmark')]
        commands.append(cmd)
        with (out/'bridge_link.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        (out/'build_commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        paths=[binary,source/'mm_conversion.h',source/'nonlinear.cpp',source/'nonlinear.h',
               source/'ckks_bert_basic.cpp',ROOT/'tests/blb_mm_conversion_benchmark.cpp',
               ROOT/'tests/blb_mm_bridge_benchmark.cpp',out/'blb_mm_bridge_benchmark',Path(__file__),
               *source.glob('mm_*'),*adapted.iterdir(),*libraries]
        (out/'build_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    if args.build_only:return
    env=dict(os.environ,OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OMP_MAX_ACTIVE_LEVELS='1',
             BLB_PUBLIC_QUARTER_RANGE='1',BLB_CORRECT_CONVERSION='0',BLB_CONVERSION_THREADS='1')
    if args.bridge:
        for name,bounded,corrected in [('bridge_legacy',False,False),('bridge_general',False,True),('bridge_bounded',True,True)]:
            env['BLB_PUBLIC_QUARTER_RANGE']='1' if bounded else '0'
            env['BLB_CORRECT_CONVERSION']='1' if corrected else '0'
            env['BLB_CONVERSION_THREADS']=str(args.bridge_workers if corrected else 1)
            commands=[['taskset','-c','0,2,4,6,8,10',str(out/'blb_mm_bridge_benchmark'),str(party),'28300'] for party in [1,2]]
            monitored(commands,env,out/name,as_gib=4,timeout=600)
            (out/name/'conversion_environment.json').write_text(json.dumps({k:env[k] for k in
                ['BLB_PUBLIC_QUARTER_RANGE','BLB_CORRECT_CONVERSION','BLB_CONVERSION_THREADS']},indent=2)+'\n')
            print(name+' measured (see separate numerical statuses).',flush=True)
    else:
        mismatch=[['env','BLB_PUBLIC_QUARTER_RANGE='+str(party-1),'taskset','-c','0,2,4,6,8,10',
                   str(binary),str(party),'28300','--expect-contract-rejection'] for party in [1,2]]
        monitored(mismatch,env,out/'mode_mismatch',as_gib=4,timeout=30)
        commands=[['taskset','-c','0,2,4,6,8,10',str(binary),str(party),'28300'] for party in [1,2]]
        monitored(commands,env,out/'kernels',as_gib=4,timeout=600)
        print('BLB MM native kernel checks and benchmark passed.',flush=True)

if __name__=='__main__':main()
