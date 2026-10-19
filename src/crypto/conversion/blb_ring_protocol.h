#pragma once

#include "hera/wide_ring.h"
#include "bounded_carry.h"

#include <seal/seal.h>
#include <cstdint>
#include <vector>


namespace hera::conversion_internal::blb {

// Reimplementation of the centered field/ring boundaries described by BLB
// Appendix C.3, SEC-PPDL FixPoint.h and M&M. This layer has no CKKS encoding,
// transport ownership, model metadata or hidden precision changes.
// Both peers must negotiate identical public arguments before calling it.
// PRECONDITION on the reconstructed secret (not individual shares):
// prime: |x| < floor(q/4); binary source: |x| < 2^(input_bits-2).
// Only this bounded protocol is implemented; no generic fallback exists.

// One SCI worker/session. party is its effective SCI role, which may differ
// from the physical server/client role for alternating workers.
class RingProtocol {
public:
    RingProtocol(int party, ConversionOt &ot);

    // Centered Z_q -> signed Z_(2^output_bits). The quarter-range path uses
    // one weighted arithmetic-OR OT under the public range precondition.
    std::vector<WideRingShare> FieldToRing(
        const std::vector<std::uint64_t> &local, std::uint64_t q,
        int output_bits);

    // Exact signed lifting of a binary-ring secret. It does not truncate or
    // change its fixed-point scale. A one-OT OR replaces compare+B2A only
    // under the explicit quarter-range precondition.
    std::vector<WideRingShare> ExtendSigned(
        const std::vector<std::uint64_t> &local, int input_bits,
        int output_bits);

private:
    int party_;
    ConversionOt &ot_;
};

// The LOCAL part of BLB Appendix C.3 ring-to-field, after fresh signed
// extension. For each target RNS prime q_j, subtract 2^bits mod q_j on the
// physical server. It is not valid to call this on arbitrary unrefreshed FFT
// shares and simply assume their integer sum crosses 2^bits exactly once.
std::vector<std::uint64_t> RnsWrapCorrection(
    int bits, const seal::SEALContext::ContextData &target);
seal::Plaintext ProjectExtendedRingToRns(
    const std::vector<WideRingShare> &coefficients, int bits,
    const seal::SEALContext::ContextData &target,
    const std::vector<std::uint64_t> &wrap_correction,
    bool subtract_wrap);

} // namespace hera::conversion_internal::blb
