using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SrvSurvey.game
{
    /// <summary>
    /// Created with assistance from Claude AI.
    /// 
    /// Time-averaged inverse-square distance between a body and a star: <1/d²>, integrated over every
    /// orbit that separates them. Replaces the closed-form leg combining, which is only exact for two
    /// independent orbits and breaks down once a system has nested binaries (3+ stars).
    ///
    /// Model:
    ///  - A body's position is the vector sum of the Kepler orbits up its parent chain, back to the
    ///    common parent of the body and the star (anything above that cancels out of d).
    ///  - Unrelated orbits get independent phases (uniform in time).
    ///  - Locked pairs - two children of the same parent with the same orbital period, whether stars OR
    ///    barycentres - share ONE phase and sit on opposite sides of their parent:
    ///        r_partner = -(a_partner / a_owner) * r_owner
    ///    This is what makes (A+B)-barycentre vs C behave correctly in a trinary.
    ///  - Phases come from a Kronecker (R_d) low-discrepancy sequence, so the result is deterministic -
    ///    important for the prediction test harness - and converges much faster than random sampling.
    ///
    /// Units: semiMajorAxis only has to be consistent across the system. Angles are degrees.
    /// orbitalPeriod is only compared between bodies, but must be the same unit for all of them (seconds).
    /// </summary>
    internal static class OrbitIntegrator
    {
        // 256 samples is within ~0.4% of a 16k-sample reference on a 5-star test hierarchy, which is far
        // below the gap between genuinely different brightness candidates. 8x cheaper than 2048.
        private const int sampleCount = 256;

        /// <summary> Relative tolerance on orbital period for two siblings to count as a locked pair </summary>
        private const double pairTolerance = 1e-5;

        private struct Kepler
        {
            public double a, e, sqrt1me2;
            public double cO, sO, ci, si, cw, sw;
        }

        /// <summary>
        /// Mean of 1/d² between <paramref name="body"/> and <paramref name="star"/>, where
        /// <paramref name="commonParent"/> is the nearest shared ancestor (null if none is known).
        /// </summary>
        public static double meanInverseSquare(SystemBody body, SystemBody star, SystemBody? commonParent)
        {
            var bodyChain = chain(body, commonParent);
            var starChain = chain(star, commonParent);
            var legCount = bodyChain.Count + starChain.Count;
            if (legCount == 0) return double.PositiveInfinity;

            // One orbit only (e.g. a planet around a lone star): exact closed form, no sampling needed.
            if (legCount == 1)
            {
                var only = bodyChain.Count == 1 ? bodyChain[0] : starChain[0];
                var e = clampEccentricity(only.orbitalEccentricity);
                return 1.0 / (only.semiMajorAxis * only.semiMajorAxis * Math.Sqrt(1 - e * e));
            }

            List<SystemBody> siblings;
            lock (body.system.bodies) { siblings = body.system.bodies.ToList(); }

            // Work out which body "owns" the phase for each leg. Locked partners share their owner.
            var owners = new List<SystemBody>();
            var ownerIdx = new Dictionary<int, int>();
            var parentOf = buildParentMap(siblings);
            var bodyLegs = buildLegs(bodyChain, siblings, parentOf, owners, ownerIdx);
            var starLegs = buildLegs(starChain, siblings, parentOf, owners, ownerIdx);

            var kep = owners.Select(toKepler).ToArray();
            var n = kep.Length;
            var maxA2 = kep.Max(k => k.a * k.a);
            if (maxA2 <= 0) return double.PositiveInfinity;

            // Two independent orbits of near-equal size can pass arbitrarily close, which makes the raw
            // integral blow up. Floor d² so one lucky sample cannot dominate.
            var floor = 1e-6 * maxA2;

            var alpha = kronecker(n);
            var px = new double[n];
            var py = new double[n];
            var pz = new double[n];

            double acc = 0;
            for (var i = 0; i < sampleCount; i++)
            {
                for (var k = 0; k < n; k++)
                {
                    var phase = ((0.5 + i * alpha[k]) % 1.0) * 2 * Math.PI;
                    position(kep[k], phase, out px[k], out py[k], out pz[k]);
                }

                double dx = 0, dy = 0, dz = 0;
                foreach (var (idx, scale) in bodyLegs) { dx += scale * px[idx]; dy += scale * py[idx]; dz += scale * pz[idx]; }
                foreach (var (idx, scale) in starLegs) { dx -= scale * px[idx]; dy -= scale * py[idx]; dz -= scale * pz[idx]; }

                acc += 1.0 / Math.Max(dx * dx + dy * dy + dz * dz, floor);
            }

            return acc / sampleCount;
        }

        /// <summary>
        /// Debug helper: describes the legs, owners and result for one body/star pair. Log it for the
        /// failing case (e.g. B 7 vs A and vs B) to see which orbit data is missing or inconsistent.
        /// </summary>
        public static string describe(SystemBody body, SystemBody star, SystemBody? commonParent)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"{body.name} vs {star.name}: commonParent={(commonParent == null ? "NULL" : commonParent.name + " id " + commonParent.id)}");

            List<SystemBody> siblings;
            lock (body.system.bodies) { siblings = body.system.bodies.ToList(); }
            var parentOf = buildParentMap(siblings);

            void dump(string label, SystemBody start)
            {
                sb.AppendLine($"  {label}: {start.name} (id {start.id}) parentBodies=[{string.Join(", ", start.parentBodies.Select(p => $"{p.name}#{p.id}"))}]");
                foreach (var b in chain(start, commonParent))
                {
                    var o = findOwner(b, siblings, parentOf);
                    sb.AppendLine($"    leg {b.name}#{b.id} (parent #{(parentOf.TryGetValue(b.id, out var pid) ? pid.ToString() : "?")}): a={b.semiMajorAxis:G6} e={b.orbitalEccentricity:G4} P={b.orbitalPeriod:G6} inc={b.orbitalInclination:G4} owner=#{o.id}{(o.id == b.id ? "" : " (mirrored)")}");
                }
                if (chain(start, commonParent).Count == 0) sb.AppendLine("    (no legs - semiMajorAxis missing/0 or body is the common parent)");
            }
            dump("body", body);
            dump("star", star);

            var m = meanInverseSquare(body, star, commonParent);
            sb.AppendLine($"  <1/d^2>={m:G6}  effective distance={(m > 0 && !double.IsInfinity(m) ? Math.Sqrt(1 / m) : double.NaN):G6}");
            return sb.ToString();
        }

        /// <summary>
        /// The bodies that orbit between <paramref name="start"/> and <paramref name="stop"/> (exclusive),
        /// nearest first, skipping anything with no semi-major axis (it contributes no displacement).
        /// </summary>
        private static List<SystemBody> chain(SystemBody start, SystemBody? stop)
        {
            var result = new List<SystemBody>();
            if (stop != null && start.id == stop.id) return result;

            if (start.semiMajorAxis > 0) result.Add(start);
            foreach (var parent in start.parentBodies)
            {
                if (stop != null && parent.id == stop.id) break;
                if (parent.semiMajorAxis > 0) result.Add(parent);
            }
            return result;
        }

        private static List<(int idx, double scale)> buildLegs(List<SystemBody> chain, List<SystemBody> siblings, Dictionary<int, int> parentOf, List<SystemBody> owners, Dictionary<int, int> ownerIdx)
        {
            var legs = new List<(int idx, double scale)>();
            foreach (var b in chain)
            {
                var owner = findOwner(b, siblings, parentOf);
                if (!ownerIdx.TryGetValue(owner.id, out var idx))
                {
                    idx = owners.Count;
                    owners.Add(owner);
                    ownerIdx[owner.id] = idx;
                }

                // owner follows its own orbit; a locked partner mirrors it, scaled by the ratio of axes
                var scale = owner.id == b.id ? 1.0 : (owner.semiMajorAxis > 0 ? -b.semiMajorAxis / owner.semiMajorAxis : 0.0);
                legs.Add((idx, scale));
            }
            return legs;
        }

        /// <summary>
        /// The body whose orbit supplies the phase for <paramref name="b"/>: itself, or - when it has a
        /// locked partner (same parent, same period) - whichever of the pair has the lowest id.
        /// </summary>
        private static SystemBody findOwner(SystemBody b, List<SystemBody> siblings, Dictionary<int, int> parentOf)
        {
            if (!parentOf.TryGetValue(b.id, out var parentId) || b.orbitalPeriod <= 0) return b;

            // Collect every sibling locked to b (same parent, same period).
            var partners = new List<SystemBody>();
            foreach (var other in siblings)
            {
                if (other.id == b.id || other.orbitalPeriod <= 0) continue;
                if (!parentOf.TryGetValue(other.id, out var otherParent) || otherParent != parentId) continue;
                if (Math.Abs(other.orbitalPeriod - b.orbitalPeriod) > pairTolerance * b.orbitalPeriod) continue;
                partners.Add(other);
            }

            // Mirroring is only valid for exactly two bodies around a shared barycentre. With 3+ bodies
            // on the same period the geometry isn't determined by per-body orbit data, so fall back to
            // independent phases rather than guess.
            if (partners.Count != 1) return b;

            return partners[0].id < b.id ? partners[0] : b;
        }

        /// <summary>
        /// body id -> id of its direct parent. A body's own parents list gives this directly, but barycentre
        /// rows (notably from Spansh) carry no parents of their own, so also infer them from the chains of
        /// the bodies below: in a chain [p0, p1, p2...], p0's parent is p1, p1's parent is p2, and so on.
        /// Without this a barycentre is never matched to its locked partner (e.g. star A vs the BC barycentre).
        /// </summary>
        private static Dictionary<int, int> buildParentMap(List<SystemBody> bodies)
        {
            var map = new Dictionary<int, int>();
            foreach (var b in bodies)
            {
                if (b.parents == null || b.parents.Count == 0) continue;
                var ids = b.parents.Select(p => p.id).ToList();
                map.TryAdd(b.id, ids[0]);
                for (var i = 0; i + 1 < ids.Count; i++)
                    map.TryAdd(ids[i], ids[i + 1]);
            }
            return map;
        }

        private static double clampEccentricity(double e) => Math.Min(Math.Max(e, 0), 0.999);

        private static Kepler toKepler(SystemBody b)
        {
            var e = clampEccentricity(b.orbitalEccentricity);
            var inc = b.orbitalInclination * Math.PI / 180;
            var node = b.ascendingNode * Math.PI / 180;
            var peri = b.argOfPeriapsis * Math.PI / 180;
            return new Kepler
            {
                a = b.semiMajorAxis,
                e = e,
                sqrt1me2 = Math.Sqrt(1 - e * e),
                cO = Math.Cos(node),
                sO = Math.Sin(node),
                ci = Math.Cos(inc),
                si = Math.Sin(inc),
                cw = Math.Cos(peri),
                sw = Math.Sin(peri),
            };
        }

        /// <summary> Position relative to the parent at the given mean anomaly (radians) </summary>
        private static void position(in Kepler k, double meanAnomaly, out double x, out double y, out double z)
        {
            var E = k.e < 0.8 ? meanAnomaly : Math.PI;
            for (var it = 0; it < 12; it++)
            {
                var delta = (E - k.e * Math.Sin(E) - meanAnomaly) / (1 - k.e * Math.Cos(E));
                E -= delta;
                if (Math.Abs(delta) < 1e-8) break;
            }

            var ox = k.a * (Math.Cos(E) - k.e);
            var oy = k.a * k.sqrt1me2 * Math.Sin(E);
            x = (k.cO * k.cw - k.sO * k.sw * k.ci) * ox + (-k.cO * k.sw - k.sO * k.cw * k.ci) * oy;
            y = (k.sO * k.cw + k.cO * k.sw * k.ci) * ox + (-k.sO * k.sw + k.cO * k.cw * k.ci) * oy;
            z = (k.sw * k.si) * ox + (k.cw * k.si) * oy;
        }

        /// <summary> Roberts' R_d additive-recurrence constants: a deterministic low-discrepancy sequence in d dimensions </summary>
        private static double[] kronecker(int dims)
        {
            var g = 2.0;
            for (var i = 0; i < 60; i++) g = Math.Pow(1 + g, 1.0 / (dims + 1));

            var alpha = new double[dims];
            for (var j = 0; j < dims; j++) alpha[j] = Math.Pow(1 / g, j + 1);
            return alpha;
        }
    }
}
