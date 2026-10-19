#!/usr/bin/env python3
"""Audit BLB's source dataflow and actual native SEAL region parameters.

Counts below describe explicitly named assumptions, not a completed BERT model.
No performance extrapolation or production cryptographic operation is made.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
#include <seal/seal.h>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
struct Region {
  const char *name; size_t degree; int scale;
  std::vector<int> bits; std::unique_ptr<seal::SEALContext> context;
};
int main() {
  std::vector<Region> r;
  r.push_back({"R1_he4",16384,38,{60,34,60,60,60,60},nullptr});
  r.push_back({"R2_he1",32768,44,{60,34,60,60,60,60,60},nullptr});
  r.push_back({"R3_softmax",32768,40,{60,40,40,40,40,40,40,40,60},nullptr});
  r.push_back({"R4_he2",32768,43,{60,60,60,60,60,60,60},nullptr});
  r.push_back({"R5_he3",32768,46,{60,32,60,60,60,60,60,60},nullptr});
  for (auto &p:r) {
    seal::EncryptionParameters params(seal::scheme_type::ckks);
    params.set_poly_modulus_degree(p.degree);
    params.set_coeff_modulus(seal::CoeffModulus::Create(p.degree,p.bits));
    p.context=std::make_unique<seal::SEALContext>(params,true,seal::sec_level_type::tc128);
    if (!p.context->parameters_set()) return 1;
    std::cout<<"REGION,name="<<p.name<<",degree="<<p.degree<<",scale="<<p.scale
      <<",data_primes="<<p.context->first_context_data()->parms().coeff_modulus().size()
      <<",terminal_q="<<p.context->last_context_data()->parms().coeff_modulus()[0].value()<<"\n";
  }
  for (auto edge:std::vector<std::pair<int,int>>{{0,1},{1,2},{2,3},{3,4},{4,0},{1,3},{0,0}}) {
    auto &a=r[edge.first];auto &b=r[edge.second];
    const auto af=a.context->first_parms_id(),bf=b.context->first_parms_id();
    bool shared_level=false;
    for(auto level=a.context->first_context_data();level;level=level->next_context_data())
      shared_level=shared_level || bool(b.context->get_context_data(level->parms_id()));
    std::cout<<"EDGE,source="<<a.name<<",target="<<b.name
      <<",same_degree="<<(a.degree==b.degree)
      <<",same_first_parameters="<<(af==bf)
      <<",target_first_in_source_chain="<<bool(a.context->get_context_data(bf))
      <<",source_first_in_target_chain="<<bool(b.context->get_context_data(af))
      <<",shared_data_level="<<shared_level
      <<",same_context_instance="<<(edge.first==edge.second)<<"\n";
  }
  // Public metadata propagation from the exact multiply/rescale operations
  // in native ckks_bert.cpp and linear_ckks.cpp. This is not a model run.
  struct State {int level; double scale;};
  auto rescale=[&](Region &p,State s) {
    auto level=p.context->first_context_data();
    for(int i=0;i<s.level;++i)level=level->next_context_data();
    return State{s.level+1,s.scale-std::log2(double(level->parms().coeff_modulus().back().value()))};
  };
  auto print=[](const char *name,const char *region,State s) {
    std::cout<<"STATE,name="<<name<<",region="<<region<<",level="<<s.level<<",scale_bits="<<s.scale<<"\n";
  };
  auto &a=r[1];
  auto attention_skip=rescale(a,{0,3.*a.scale});
  auto v=rescale(a,{attention_skip.level,attention_skip.scale+a.scale});
  auto masked=rescale(a,{v.level,v.scale+a.scale});
  auto qk=rescale(a,{masked.level,2.*masked.scale});
  qk=rescale(a,{qk.level,qk.scale+a.scale});
  print("attention_normalized_residual",a.name,attention_skip);
  print("v",a.name,v);print("qk",a.name,qk);
  auto &f=r[4];
  auto ffn_skip=rescale(f,{0,3.*f.scale});
  auto x=rescale(f,{ffn_skip.level,ffn_skip.scale+f.scale});
  auto x2=rescale(f,{x.level,2.*x.scale});
  auto x4=rescale(f,{x2.level,2.*x2.scale});
  auto poly=rescale(f,{x4.level,x4.scale+f.scale});
  poly=rescale(f,poly);
  print("ffn_normalized_residual",f.name,ffn_skip);
  print("gelu_x",f.name,x);print("gelu_polynomial",f.name,poly);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=ROOT.parent/'blb/SCI')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native, out = args.native.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = native/'tests/bert_large_bolt'

    def anchor(filename, fragment):
        path = source/filename
        lines = path.read_text().splitlines()
        found = [i+1 for i,line in enumerate(lines) if fragment in line]
        if len(found) != 1:
            raise RuntimeError(f'nonunique source anchor {filename}: {fragment}: {found}')
        return {'path':str(path), 'line':found[0], 'text':lines[found[0]-1].strip()}

    evidence = {
        'region_parameters': anchor('linear_ckks.cpp','void Linear_ckks::set_HE()'),
        'independent_key_generation': anchor('he.h','KeyGenerator keygen(*context);'),
        'v_in_designed_source_vector': anchor('linear_ckks.cpp','HE_result[data.image_size * data.image_size * PACKING_NUM / data.slot_count + i] ='),
        'source_half_is_prefix_copy': anchor('ckks_bert.cpp','output_cts_half[i] = q_k_v[i];'),
        'v_is_fake_at_target': anchor('ckks_bert.cpp','vector<double> fake_vec(slot, 1.0);'),
        'attention_residual_is_fake': anchor('ckks_bert.cpp','// 模拟residual, h3 scale'),
        'ffn_residual_is_fake': anchor('ckks_bert.cpp','//simulate residual'),
        'gelu_x_is_fake': anchor('ckks_bert.cpp','x_array = fpmath->fix->input(party, gelu_input_size,1, true, bitwidth, s);'),
        'block_entry': anchor('ckks_bert_large_main.cpp','void transformer_block_test('),
    }
    # Native wrapper validates a common scale for every member of a frame.
    evidence['c2m_batch_scale_check'] = anchor('mm_ckks_bridge.cpp','input[ct].scale()!=input[0].scale()')
    base = [
        {'name':'ln1_center','direction':'M2C','source':'MPC','target':'R2'},
        {'name':'ln1_inverse','direction':'M2C','source':'MPC','target':'R2'},
        {'name':'qk_and_v_combined','direction':'C2M','source':'R2','target':'MPC'},
        {'name':'softmax_shift','direction':'M2C','source':'MPC','target':'R3'},
        {'name':'softmax_exp','direction':'C2M','source':'R3','target':'MPC'},
        {'name':'softmax_probability','direction':'M2C','source':'MPC','target':'R4'},
        {'name':'ln2_center','direction':'C2M','source':'R4','target':'MPC'},
        {'name':'ln2_variance','direction':'C2M','source':'R4','target':'MPC'},
        {'name':'ln2_center_to_ffn','direction':'M2C','source':'MPC','target':'R5'},
        {'name':'ln2_inverse_to_ffn','direction':'M2C','source':'MPC','target':'R5'},
        {'name':'gelu_polynomial','direction':'C2M','source':'R5','target':'MPC'},
        {'name':'gelu_to_wd','direction':'M2C','source':'MPC','target':'R1'},
        {'name':'next_ln_center','direction':'C2M','source':'R1','target':'MPC'},
        {'name':'next_ln_variance','direction':'C2M','source':'R1','target':'MPC'},
    ]
    missing = [
        {'name':'v_to_av','direction':'M2C','source':'MPC','target':'R4','active_values':98304,
         'reason':'Assume a repaired common-scale qk_and_v source batch; target currently uses fake_vec'},
        {'name':'attention_residual_source','direction':'C2M','source':'R2','target':'MPC','active_values':98304,
         'reason':'Assume normalized-residual semantics and residual retained only as HE LN1 output'},
        {'name':'attention_residual_target','direction':'M2C','source':'MPC','target':'R4','active_values':98304,
         'reason':'Distinct HE context/key/chain; self-addition is a placeholder'},
        {'name':'ffn_residual_source','direction':'C2M','source':'R5','target':'MPC','active_values':98304,
         'reason':'Assume residual retained only as HE LN2 output'},
        {'name':'ffn_residual_target','direction':'M2C','source':'MPC','target':'R1','active_values':98304,
         'reason':'N32768 to N16384; ordinary mod-switch does not change polynomial degree'},
    ]
    gelu = {'name':'gelu_real_input','direction':'C2M','source':'R5','target':'MPC','active_values':393216,
            'reason':'MPC comparisons/select need actual x; current x_array is all1 placeholder'}
    split_v = {'name':'v_source_separate','direction':'C2M','source':'R2','target':'MPC','active_values':98304,
               'reason':'QK and V have different source levels/scales; no common-scale repair assumed. Base qk_and_v becomes QK-only.'}
    def counts(items):
        counter = Counter(item['direction'] for item in items)
        return {'calls_if_separate':len(items), 'C2M':counter['C2M'], 'M2C':counter['M2C']}

    probe = out/'native_region_probe.cpp'
    probe.write_text(PROBE)
    binary = out/'native_region_probe'
    command = ['c++','-std=c++17','-O2','-pthread',
               '-I'+str(native/'extern/SEAL/native/src'),
               '-I'+str(native/'build-hera/deps/SEAL/native/src'),
               '-I'+str(native/'extern/SEAL/thirdparty/msgsl-src/include'),
               str(probe), str(native/'build-hera/deps/SEAL/lib/libseal-4.1.a'),'-o',str(binary)]
    with (out/'build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    log = subprocess.check_output([str(binary)],text=True)
    (out/'native_region_probe.log').write_text(log)
    regions, transitions, states = [], [], []
    for line in log.splitlines():
        fields = dict(part.split('=',1) for part in line.split(',')[1:])
        for key,value in list(fields.items()):
            if value.isdigit(): fields[key] = int(value)
            elif key == 'scale_bits': fields[key] = float(value)
        (regions if line.startswith('REGION,') else states if line.startswith('STATE,') else transitions).append(fields)
    for transition in transitions:
        if transition['source'] != transition['target']:
            if transition['same_first_parameters'] or transition['target_first_in_source_chain']:
                raise RuntimeError('Cross-region compatibility changed; re-audit required')
    result = {
        'scope':'Native local BLB artifact; normalized-residual interpretation; independently instantiated regions; no cross-region key/ring switching implementation assumed',
        'evidence':evidence, 'regions':regions, 'transitions':transitions, 'public_metadata_trace':states,
        'explicit_base':base, 'missing_region_edges':missing, 'additional_gelu_input':gelu,
        'source_v_split_without_alignment':split_v,
        'scenarios':{
            'artifact_explicit':counts(base),
            'previous_target_only_supplement':counts(base+[x for x in missing if x['direction']=='M2C']),
            'independent_regions_he_residuals_no_extra_batching':counts(base+missing),
            'above_plus_separate_real_gelu_input':counts(base+missing+[gelu]),
            'above_with_qk_v_split_no_gelu_input_fix':counts(base+missing+[split_v]),
            'above_with_qk_v_split_and_separate_real_gelu_input':counts(base+missing+[split_v,gelu]),
        },
        'caveats':[
            '19/20/21 are conditional ledgers, not intrinsic lower bounds or a completed correct inference implementation.',
            'The 19-call ledger assumes QK/V source-scale alignment and correct complex packing. Current half prefix copy excludes the appended V ciphertexts for BERT-base.',
            'Residuals already available as correct MPC shares remove their source C2M, at a potentially additional MPC computation cost.',
            'Same-profile/scale compatible extra tensors may join existing frames; data volume and packing work remain.',
            'Changing region names or matching only N/terminal q does not unify keys, modulus chains, scales or layouts.',
            'GeLU approximation branches and other artifact placeholders still need a complete numerical implementation; 21 is not an upper bound.',
        ]}
    (out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (out/'build_commands.json').write_text(json.dumps([command],indent=2)+'\n')
    inputs=[Path(__file__),source/'ckks_bert.cpp',source/'ckks_bert_basic.cpp',source/'ckks_bert_large_main.cpp',
            source/'linear_ckks.cpp',source/'he.h',source/'mm_ckks_bridge.cpp',
            native/'extern/SEAL/native/src/seal/evaluator.cpp',probe,binary]
    (out/'manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},indent=2)+'\n')
    print(json.dumps(result['scenarios'],indent=2,ensure_ascii=False))
    print(log)


if __name__ == '__main__':
    main()
