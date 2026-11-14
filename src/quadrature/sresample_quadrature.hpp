#ifndef SRESAMPLE_QUADRATURE_HPP
#define SRESAMPLE_QUADRATURE_HPP

#include "sresample_base.hpp"

// Quadrature rules on reference elements, in double. Each table lives inside a host-device
// function, so one definition serves the host and the device.

namespace sresample {
    namespace quadrature {

        // Triangle {(x, y) : x, y >= 0, x + y <= 1}, 12 points, exact to degree 6; the weights sum to 1.
        // smesh::TRI3 6th order quadrature rule
        static constexpr int tri12_n = 12;

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tri12_w(const int q) {
            constexpr double w[12] = {0.050844906370206816920936809106869,
                                      0.050844906370206816920936809106869,
                                      0.050844906370206816920936809106869,
                                      0.11678627572637936602528961138558,
                                      0.11678627572637936602528961138558,
                                      0.11678627572637936602528961138558,
                                      0.082851075618373575193553456420442,
                                      0.082851075618373575193553456420442,
                                      0.082851075618373575193553456420442,
                                      0.082851075618373575193553456420442,
                                      0.082851075618373575193553456420442,
                                      0.082851075618373575193553456420442};
            return w[q];
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tri12_x(const int q) {
            constexpr double x[12] = {0.063089014491502228340331602870819,
                                      0.063089014491502228340331602870819,
                                      0.87382197101699554331933679425836,
                                      0.24928674517091042129163855310702,
                                      0.24928674517091042129163855310702,
                                      0.50142650965817915741672289378596,
                                      0.053145049844816947353249671631398,
                                      0.053145049844816947353249671631398,
                                      0.31035245103378440541660773395655,
                                      0.31035245103378440541660773395655,
                                      0.63650249912139864723014259441205,
                                      0.63650249912139864723014259441205};
            return x[q];
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tri12_y(const int q) {
            constexpr double y[12] = {0.063089014491502228340331602870819,
                                      0.87382197101699554331933679425836,
                                      0.063089014491502228340331602870819,
                                      0.24928674517091042129163855310702,
                                      0.50142650965817915741672289378596,
                                      0.24928674517091042129163855310702,
                                      0.31035245103378440541660773395655,
                                      0.63650249912139864723014259441205,
                                      0.053145049844816947353249671631398,
                                      0.63650249912139864723014259441205,
                                      0.053145049844816947353249671631398,
                                      0.31035245103378440541660773395655};
            return y[q];
        }

        // [0, 1], 12 Gauss-Legendre points, exact to degree 23; the weights sum to 1.
        static constexpr int line12_n = 12;

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double line12_x(const int q) {
            constexpr double x[12] = {0.009219682876640489244124410106451250612735748291015625000000,
                                      0.047941371814762490100036984586040489375591278076171875000000,
                                      0.115048662902847487199409215463674627244472503662109375000000,
                                      0.206341022856691480580337838546256534755229949951171875000000,
                                      0.316084250500909991199449677878874354064464569091796875000000,
                                      0.437383295744265487847712847724324092268943786621093750000000,
                                      0.562616704255734512152287152275675907731056213378906250000000,
                                      0.683915749499089953289399090863298624753952026367187500000000,
                                      0.793658977143308463908510930195916444063186645507812500000000,
                                      0.884951337097152457289439553278498351573944091796875000000000,
                                      0.952058628185237454388811784156132489442825317382812500000000,
                                      0.990780317123359566267026821151375770568847656250000000000000};
            return x[q];
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double line12_w(const int q) {
            constexpr double w[12] = {0.023587668193255501014604647025407757610082626342773437500000,
                                      0.053469662997658998215833037193078780546784400939941406250000,
                                      0.080039164271672999517726054818922420963644981384277343750000,
                                      0.101583713361533004015946346498822094872593879699707031250000,
                                      0.116746268269177499998789926394238136708736419677734375000000,
                                      0.124573522906701497636738906749087618663907051086425781250000,
                                      0.124573522906701497636738906749087618663907051086425781250000,
                                      0.116746268269177499998789926394238136708736419677734375000000,
                                      0.101583713361533004015946346498822094872593879699707031250000,
                                      0.080039164271672999517726054818922420963644981384277343750000,
                                      0.053469662997658998215833037193078780546784400939941406250000,
                                      0.023587668193255501014604647025407757610082626342773437500000};
            return w[q];
        }

        // [0, 1], 6 Gauss-Legendre points, exact to degree 11, to 10 digits: the tensor rule of the
        // quadrilaterals.
        static constexpr int line6_n = 6;

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double line6_x(const int q) {
            constexpr double x[6] = {0.03376524290, 0.1693953068, 0.3806904070, 0.6193095930, 0.8306046932, 0.9662347571};
            return x[q];
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double line6_w(const int q) {
            constexpr double w[6] = {0.08566224619, 0.1803807865, 0.2339569673, 0.2339569673, 0.1803807865, 0.08566224619};
            return w[q];
        }

        // Tetrahedron {l_k >= 0, sum l_k = 1} in barycentric coordinates, 14 points, exact to degree 5
        // (Walkington's symmetric rule: orbits (a,a,a,1-3a), (c,c,c,1-3c), (e,e,1/2-e,1/2-e)); the
        // weights sum to 1. tet14_l(q, l) gives point q's four barycentric coordinates.
        static constexpr int tet14_n = 14;

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tet14_w(const int q) {
            // Walkington's weights times 6 (they summed to the unit tet's volume, 1/6).
            constexpr double w1 = 6 * 0.01224884051939366, w2 = 6 * 0.01878132095300264, w3 = 6 * 0.007091003462846911;
            return q < 4 ? w1 : (q < 8 ? w2 : w3);
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet14_l(const int q, double l[4]) {
            constexpr double a = 0.0927352503108912, c = 0.3108859192633006, e = 0.4544962958743504;
            // The six pairs of the third orbit: the coordinates that get e.
            constexpr int pair[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
            if (q < 8) {
                const double p = q < 4 ? a : c;
                for (int k = 0; k < 4; ++k) l[k] = p;
                l[q % 4] = 1 - 3 * p;
                return;
            }
            for (int k = 0; k < 4; ++k) l[k] = 0.5 - e;
            l[pair[q - 8][0]] = e;
            l[pair[q - 8][1]] = e;
        }

        // Tetrahedron, 24 points, exact to degree 6 (Keast 1986, positive weights: orbits (a,a,a,1-3a)
        // for three a, and the 12 permutations of (f,f,g,h)); the weights sum to 1. The degree a
        // trilinear times a trilinear basis needs on a parallelepiped.
        static constexpr int tet24_n = 24;

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE double tet24_w(const int q) {
            // Keast's weights times 6.
            constexpr double w[4] = {6 * 0.00665379170969464506, 6 * 0.00167953517588677620, 6 * 0.00922619692394239843,
                                     6 * 0.00803571428571428248};
            return q < 12 ? w[q / 4] : w[3];
        }

        SRESAMPLE_HOST_DEVICE SRESAMPLE_INLINE void tet24_l(const int q, double l[4]) {
            if (q < 12) {
                constexpr double a[3] = {0.214602871259151684, 0.0406739585346113397, 0.322337890142275646};
                const double     p    = a[q / 4];
                for (int k = 0; k < 4; ++k) l[k] = p;
                l[q % 4] = 1 - 3 * p;
                return;
            }
            // The 12 placements of (g, h) in two distinct coordinates (ordered); f in the other two.
            constexpr double f = 0.0636610018750175299, g = 0.269672331458315867, h = 0.603005664791649076;
            constexpr int    gh[12][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {0, 3}, {3, 0}, {1, 2}, {2, 1}, {1, 3}, {3, 1}, {2, 3}, {3, 2}};
            for (int k = 0; k < 4; ++k) l[k] = f;
            l[gh[q - 12][0]] = g;
            l[gh[q - 12][1]] = h;
        }

    }  // namespace quadrature
}  // namespace sresample

#endif  // SRESAMPLE_QUADRATURE_HPP
