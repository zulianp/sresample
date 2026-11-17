// The pairs of the HEX8 transfer by convex pieces (sresample_hex8_pieces_kernel.hpp) against the
// same pairs split tet by split tet (hex8_mixed_mass_pair_tets, hex8_fluid_mixed_mass_pair_tets),
// on random overlapping pairs, in the bases the pieces apply to:
//   the coupled volume sum_ia B_ia agrees to round-off (the same region), always (relative to the
//   largest pair's: a pair that only touches has none);
//   B agrees to round-off for parallelepipeds (both rules exact), and to the rule's error for
//   warped hexes and prisms (another fan of the same region);
//   no overflow; Newton converges.
// A hex too distorted for a convex K falls back, and B is then the split tets' exactly. The box
// polyhedron's connectivity: its volume, and that of a clipped box, from its moments.

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sresample_hex8_fluid_kernel.hpp"
#include "sresample_hex8_pieces_kernel.hpp"
#include "sresample_test.hpp"

using namespace sresample;
using namespace sresample::intersect;

namespace {

    unsigned long long rng_state = 0x5DEECE66DULL;
    double             uniform() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
    }

    // A hex of edge h at o, its nodes moved by up to warp * h (prism: (x, z) only, the same on
    // both y levels), sheared by a fixed matrix (not axis-aligned).
    void make_hex(const double o[3], const double h, const double warp, const bool prism, double x[8][3]) {
        const double A[3][3] = {{1.0, 0.15, -0.1}, {0.05, 0.95, 0.12}, {-0.08, 0.1, 1.05}};
        double       m[4][3];
        for (int c = 0; c < 4; ++c) {
            for (int d = 0; d < 3; ++d) m[c][d] = warp * h * (2 * uniform() - 1);
        }
        for (int a = 0; a < 8; ++a) {
            double y[3];
            for (int d = 0; d < 3; ++d) {
                const double move = prism ? (d == 1 ? 0 : m[a % 4][d]) : warp * h * (2 * uniform() - 1);
                y[d]              = h * hex8_corner(a, d) + move;
            }
            for (int d = 0; d < 3; ++d) x[a][d] = o[d] + A[d][0] * y[0] + A[d][1] * y[1] + A[d][2] * y[2];
        }
    }

    void random_ids(long long ids[8]) {
        for (int a = 0; a < 8; ++a) ids[a] = 100 + a;
        for (int a = 7; a > 0; --a) {
            const int       b = (int)(uniform() * (a + 1)) % (a + 1);
            const long long t = ids[a];
            ids[a]            = ids[b];
            ids[b]            = t;
        }
    }

    struct Stats {
        double vol_err{0}, vol_max{0}, b_err{0}, scale{0};
        int    overflow{0}, max_it{0};
    };

    void compare(const double b[8][8], const double r[8][8], const int ni, const int na, Stats *st) {
        double vb = 0, vr = 0, sc = 0;
        for (int i = 0; i < ni; ++i) {
            for (int a = 0; a < na; ++a) {
                vb += b[i][a];
                vr += r[i][a];
                sc = fmax(sc, fabs(r[i][a]));
                st->b_err = fmax(st->b_err, fabs(b[i][a] - r[i][a]));
            }
        }
        st->scale   = fmax(st->scale, sc);
        st->vol_err = fmax(st->vol_err, fabs(vb - vr));
        st->vol_max = fmax(st->vol_max, fabs(vr));
    }

    // kind: 0 parallelepipeds, 1 warped, 2 a warped solid and fluid prisms of a quarter its size.
    // fluid: 0 TRILINEAR hex, 1 SPLIT hex, 2 a TET4 (the HEX8xTET4 pair). solid4: a TET4 solid.
    int run(const int kind, const int fluid, const bool solid4, const int pairs, Stats *st) {
        for (int k = 0; k < pairs; ++k) {
            double    xs[8][3], xf[8][3];
            long long ids_s[8], ids_f[8];
            const double o0[3] = {0, 0, 0};
            make_hex(o0, 1, kind == 0 ? 0 : 0.08, false, xs);
            double o[3];
            if (kind == 2) {
                for (int d = 0; d < 3; ++d) o[d] = -0.2 + 1.15 * uniform();
                make_hex(o, 0.25, 0.08, true, xf);
            } else {
                for (int d = 0; d < 3; ++d) o[d] = 0.4 * (2 * uniform() - 1);
                make_hex(o, 0.8, kind == 0 ? 0 : 0.08, false, xf);
            }
            random_ids(ids_s);
            random_ids(ids_f);
            if (solid4) {
                // The solid: the hex's first split tet.
                int t[6][4];
                hex8_split(hex8_split_pattern(ids_s), t);
                double xt[8][3];
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) xt[v][d] = xs[t[0][v]][d];
                }
                memcpy(xs, xt, sizeof(xt));
            }
            double b[8][8] = {{0}}, r[8][8] = {{0}};
            int    cb[5] = {0, 0, 0, 0, 0}, cr[5] = {0, 0, 0, 0, 0}, itb = 0, itr = 0;
            if (fluid == 2) {
                double tf[4][3], b4[4][8], r4[4][8];
                int    t[6][4];
                hex8_split(hex8_split_pattern(ids_f), t);
                for (int v = 0; v < 4; ++v) {
                    for (int d = 0; d < 3; ++d) tf[v][d] = xf[t[0][v]][d];
                }
                SRESAMPLE_TEST_EQ(hex8_mixed_mass_pair(xs, ids_s, tf, b4, Hex8Basis::TRILINEAR, cb, &itb), 0);
                SRESAMPLE_TEST_EQ(hex8_mixed_mass_pair_tets(xs, ids_s, tf, r4, Hex8Basis::TRILINEAR, cr, &itr), 0);
                for (int i = 0; i < 4; ++i) {
                    for (int a = 0; a < 8; ++a) {
                        b[i][a] = b4[i][a];
                        r[i][a] = r4[i][a];
                    }
                }
                compare(b, r, 4, 8, st);
            } else {
                const int       ns = solid4 ? 4 : 8;
                const Hex8Basis fb = fluid == 1 ? Hex8Basis::SPLIT : Hex8Basis::TRILINEAR;
                SRESAMPLE_TEST_EQ(hex8_fluid_mixed_mass_pair(xs, ids_s, ns, xf, ids_f, b, Hex8Basis::TRILINEAR, fb, cb, &itb), 0);
                SRESAMPLE_TEST_EQ(hex8_fluid_mixed_mass_pair_tets(xs, ids_s, ns, xf, ids_f, r, Hex8Basis::TRILINEAR, fb, cr, &itr), 0);
                compare(b, r, 8, ns, st);
            }
            st->overflow += cb[TET4_PAIR_OVERFLOW];
            st->max_it = itb > st->max_it ? itb : st->max_it;
        }
        return 0;
    }

    int test_against_tets() {
        const char *kinds[3]  = {"parallelepipeds", "warped", "fine prisms"};
        const char *fluids[3] = {"TRILINEAR hex", "SPLIT hex", "TET4"};
        for (int kind = 0; kind < 3; ++kind) {
            for (int fluid = 0; fluid < 3; ++fluid) {
                for (int solid4 = 0; solid4 < 2; ++solid4) {
                    // A TET4 solid only against a TRILINEAR fluid hex (the other cases have no piece).
                    if (solid4 && fluid != 0) continue;
                    Stats st;
                    SRESAMPLE_TEST_EQ(run(kind, fluid, solid4, 400, &st), 0);
                    printf("  %-15s %s solid, %-13s fluid: |dvol|/vol %.2e, max|dB|/max|B| %.2e, Newton <= %d\n", kinds[kind],
                           solid4 ? "TET4" : "HEX8", fluids[fluid], st.vol_err / st.vol_max, st.b_err / st.scale, st.max_it);
                    SRESAMPLE_TEST_ASSERT(st.vol_err < 1e-12 * st.vol_max);
                    SRESAMPLE_TEST_ASSERT(st.b_err < (kind == 0 ? 1e-12 : 1e-3) * st.scale);
                    SRESAMPLE_TEST_EQ(st.overflow, 0);
                    SRESAMPLE_TEST_ASSERT(st.max_it < hex8_newton_max_it);
                }
            }
        }
        return 0;
    }

    // A pair from the LDP24HX pump (a valve plate's hex, its faces folded by 1e-9 of its size, against
    // a fluid prism): the clip of K by the nearly coincident planes of a fold has fan tets of slightly
    // negative volume; summed as |V| they made the volume 4e-7 of the hex too large.
    int test_pump_pair() {
        const double    xs[8][3]  = {{0.01088823131804877, 0.00050048278886786731, 0.00087746029001569651},
                                     {0.010892236420205297, 0.00050048854044097149, 0.00098978219838150483},
                                     {0.011006041791044174, 0.00050048345573521619, 0.00099378433539213675},
                                     {0.011002447781638051, 0.00050047820024129734, 0.00088100630423660589},
                                     {0.010888268163997496, 0.00075000086809289327, 0.00087742842914466141},
                                     {0.010892273805801026, 0.00075000086809289327, 0.00098974219760065377},
                                     {0.011006074737961044, 0.00075000086809512694, 0.00099374521205282244},
                                     {0.011002479837266687, 0.00075000086809427768, 0.00088097473754882006}};
        const double    xf[8][3]  = {{0.011009358209537851, 0.0006430769230769233, 0.00099403135032918259},
                                     {0.010956538543658833, 0.0006430769230769233, 0.00091769311789494145},
                                     {0.010886367645542553, 0.0006430769230769233, 0.00096832475396951552},
                                     {0.010938384644576945, 0.0006430769230769233, 0.0010423779372239123},
                                     {0.011009358209537851, 0.00076000000000000004, 0.00099403135032918259},
                                     {0.010956538543658833, 0.00076000000000000004, 0.00091769311789494145},
                                     {0.010886367645542553, 0.00076000000000000004, 0.00096832475396951552},
                                     {0.010938384644576945, 0.00076000000000000004, 0.0010423779372239123}};
        const long long ids_s[8]  = {155581, 155590, 169568, 169558, 155587, 155593, 169571, 169564};
        const long long ids_f[8]  = {1276601845377989414LL, 1276601845447872691LL, 1276601845009704751LL, 1276602143007632596LL,
                                     1276601767513190630LL, 1276601767555150711LL, 1276601766674071917LL, 1276602220875181588LL};
        double          b[8][8], r[8][8];
        int             cb[5] = {0, 0, 0, 0, 0}, cr[5] = {0, 0, 0, 0, 0}, itb = 0, itr = 0;
        SRESAMPLE_TEST_EQ(hex8_fluid_mixed_mass_pair(xs, ids_s, 8, xf, ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, cb, &itb), 0);
        SRESAMPLE_TEST_EQ(hex8_fluid_mixed_mass_pair_tets(xs, ids_s, 8, xf, ids_f, r, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, cr, &itr), 0);
        Stats st;
        compare(b, r, 8, 8, &st);
        const double vol_s = hex8_split_volume(xs, ids_s);
        printf("  pump pair: |dvol| / vol_s %.2e, max|dB|/max|B| %.2e\n", st.vol_err / vol_s, st.b_err / st.scale);
        SRESAMPLE_TEST_ASSERT(st.vol_err < 1e-12 * vol_s);
        SRESAMPLE_TEST_ASSERT(st.b_err < 1e-3 * st.scale);
        return 0;
    }

    // A hex whose K is not convex (node 6 pushed deep inside): no pieces, the split tets' B exactly.
    int test_fallback() {
        double    xs[8][3], xf[8][3];
        long long ids_s[8], ids_f[8];
        const double o0[3] = {0, 0, 0}, o1[3] = {0.2, 0.15, 0.1};
        make_hex(o0, 1, 0, false, xs);
        make_hex(o1, 0.8, 0, false, xf);
        for (int d = 0; d < 3; ++d) xs[6][d] = 0.3 * xs[6][d] + 0.7 * 0.5 * (xs[0][d] + xs[6][d]);
        random_ids(ids_s);
        random_ids(ids_f);
        Hex8Map      map;
        Hex8PieceSet set;
        hex8_map_init(xs, &map);
        SRESAMPLE_TEST_ASSERT(!hex8_piece_set_trilinear(xs, ids_s, &map, &set));
        double b[8][8], r[8][8];
        int    cb[5] = {0, 0, 0, 0, 0}, cr[5] = {0, 0, 0, 0, 0}, itb = 0, itr = 0;
        hex8_fluid_mixed_mass_pair(xs, ids_s, 8, xf, ids_f, b, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, cb, &itb);
        hex8_fluid_mixed_mass_pair_tets(xs, ids_s, 8, xf, ids_f, r, Hex8Basis::TRILINEAR, Hex8Basis::TRILINEAR, cr, &itr);
        SRESAMPLE_TEST_EQ(memcmp(b, r, sizeof(b)), 0);
        for (int c = 0; c < 5; ++c) SRESAMPLE_TEST_EQ(cb[c], cr[c]);
        return 0;
    }

    // The box polyhedron (r3d_init_box's connectivity): its volume, and a corner cut off.
    int test_box_polyhedron() {
        const double  lo[3] = {-0.5, 0.25, 1.0}, hi[3] = {1.5, 0.75, 4.0};
        Hex8PieceClip p;
        hex8_box_polyhedron(lo, hi, &p);
        double m[10];
        tet4_polyhedron_moments(&p, m);
        SRESAMPLE_TEST_APPROXEQ(m[0], 2.0 * 0.5 * 3.0, 1e-14);
        SRESAMPLE_TEST_APPROXEQ(m[1] / m[0], 0.5, 1e-14);
        // Keep x + y + z <= the lo corner's sum + 1/2: the tet of legs 1/2 at the lo corner.
        const double n[3] = {-1, -1, -1};
        SRESAMPLE_TEST_ASSERT(tet4_clip_plane(&p, lo[0] + lo[1] + lo[2] + 0.5, n));
        tet4_polyhedron_moments(&p, m);
        SRESAMPLE_TEST_APPROXEQ(m[0], 0.5 * 0.5 * 0.5 / 6, 1e-15);
        return 0;
    }

}  // namespace

int main() {
    int err = 0;
    SRESAMPLE_RUN_TEST(test_box_polyhedron);
    SRESAMPLE_RUN_TEST(test_fallback);
    SRESAMPLE_RUN_TEST(test_pump_pair);
    SRESAMPLE_RUN_TEST(test_against_tets);
    return err;
}
