#ifndef SRESAMPLE_HEX8_PIECES_WARP_CUH
#define SRESAMPLE_HEX8_PIECES_WARP_CUH

#include "sresample_cuda_base.cuh"
#include "sresample_hex8_pieces_kernel.hpp"

// The HEX8 transfer by convex pieces on the device: one warp a pair.
//
// One thread a pair leaves a warp's lanes on different paths (a pair is separated, inside, or
// clipped into a few or many tets; Newton takes more or fewer steps): profiled, 1.4 of 32 lanes
// were active. Here lane 0 does a pair's geometry (its pieces, each pair of pieces as a list of
// tets, sresample_hex8_pieces_kernel.hpp) in shared memory, and the warp's lanes take the list's
// quadrature points 32 at a time, all on the same code. B's entries are summed over the lanes by
// shuffles in a fixed order (deterministic), each lane keeping two of the 64.

namespace sresample {
    namespace intersect {
        namespace cuda {

            static constexpr int hex8_warp_lanes = 32;

            __device__ inline double hex8_warp_sum(double v) {
                for (int o = hex8_warp_lanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                return v;
            }

            __device__ inline int hex8_warp_max(int v) {
                for (int o = hex8_warp_lanes / 2; o > 0; o >>= 1) {
                    const int u = __shfl_xor_sync(0xffffffffu, v, o);
                    v           = u > v ? u : v;
                }
                return v;
            }

            // A warp's pair, in shared memory (raw storage: the members' initializers are not run;
            // lane 0 sets every field it reads).
            struct Hex8WarpPair {
                double           xs[8][3];
                double           xf[8][3];
                long long        ids_s[8];
                long long        ids_f[8];
                Hex8Map          ms;
                Hex8Map          mf;
                Hex8PieceSet     S;
                Hex8PieceSet     F;
                Hex8Piece        ps;
                Hex8Piece        pf;
                const Hex8Piece *sp;
                const Hex8Piece *fp;
                Hex8PieceClip    poly;
                Hex8TetList      L;
                int              cls;
                int              ok;
            };

            // Lane 0: st's nodes (xs: ns of them, xf: nf) relative to the fluid element's centroid,
            // and their pieces in the bases asked for. st.ok: the pair goes by pieces.
            __device__ inline void hex8_warp_setup(Hex8WarpPair &st, const int ns, const int nf, const bool s_tri, const bool f_tri) {
                double c[3] = {0, 0, 0};
                for (int v = 0; v < nf; ++v) {
                    for (int d = 0; d < 3; ++d) c[d] += st.xf[v][d] / nf;
                }
                for (int v = 0; v < 8; ++v) {
                    for (int d = 0; d < 3; ++d) {
                        st.xs[v][d] = v < ns ? st.xs[v][d] - c[d] : 0;
                        st.xf[v][d] = v < nf ? st.xf[v][d] - c[d] : 0;
                    }
                }
                bool ok = s_tri || f_tri;
                if (ok && s_tri) {
                    hex8_map_init(st.xs, &st.ms);
                    ok = hex8_piece_set_trilinear(st.xs, st.ids_s, &st.ms, &st.S);
                } else if (ok) {
                    ok = hex8_piece_set_tets(st.xs, st.ids_s, ns, &st.S);
                }
                if (ok && f_tri) {
                    hex8_map_init(st.xf, &st.mf);
                    ok = hex8_piece_set_trilinear(st.xf, st.ids_f, &st.mf, &st.F);
                } else if (ok) {
                    ok = hex8_piece_set_tets(st.xf, st.ids_f, nf, &st.F);
                }
                st.ok = ok ? 1 : 0;
            }

            // The warp: acc[j] = B entry e = lane + 32 j (e = 8 i + a, i < nf the fluid's, a < ns the
            // solid's) over all pairs of pieces of st. Lane 0 adds the classes to counts; *most: the
            // lane's Newton maximum. False on an overflow (the same on every lane): the pair falls
            // back to the split tets.
            __device__ inline bool hex8_warp_piece_sets(Hex8WarpPair &st,
                                                        const int     lane,
                                                        const bool    six,
                                                        const int     nf,
                                                        const int     ns,
                                                        double        acc[2],
                                                        int           counts[5],
                                                        int          *most) {
                const int nq = hex8_rule_n(six);
                const int ne = 8 * nf;
                for (int m = 0; m < st.F.n; ++m) {
                    for (int k = 0; k < st.S.n; ++k) {
                        if (lane == 0) {
                            st.fp     = st.F.piece(m, &st.pf);
                            st.sp     = st.S.piece(k, &st.ps);
                            st.cls    = hex8_piece_pair_tets(*st.sp, *st.fp, &st.poly, &st.L);
                            st.L.sign = st.sp->sign * st.fp->sign;
                        }
                        __syncwarp();
                        const int cls = st.cls;
                        if (cls == TET4_PAIR_OVERFLOW) return false;
                        if (lane == 0) counts[cls]++;
                        const int P = st.L.n * nq;
                        for (int p0 = 0; p0 < P; p0 += hex8_warp_lanes) {
                            const int p = p0 + lane;
                            double    Ns[8], Nf[8], w = 0;
                            for (int a = 0; a < 8; ++a) Ns[a] = Nf[a] = 0;
                            if (p < P) {
                                double x[3];
                                w             = hex8_tet_list_point(st.L, p / nq, p % nq, six, x);
                                const int its = hex8_piece_basis(*st.sp, x, Ns);
                                const int itf = hex8_piece_basis(*st.fp, x, Nf);
                                *most         = its > *most ? its : *most;
                                *most         = itf > *most ? itf : *most;
                            }
                            for (int e = 0; e < ne; ++e) {
                                const int i = e >> 3, a = e & 7;
                                double    v = a < ns ? w * Nf[i] * Ns[a] : 0;
                                v           = hex8_warp_sum(v);
                                if ((e & (hex8_warp_lanes - 1)) == lane) acc[e / hex8_warp_lanes] += v;
                            }
                        }
                        // Lane 0 overwrites st for the next pair of pieces only after every lane read it.
                        __syncwarp();
                    }
                }
                return true;
            }

        }  // namespace cuda
    }  // namespace intersect
}  // namespace sresample

#endif  // SRESAMPLE_HEX8_PIECES_WARP_CUH
